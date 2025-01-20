from typing import Any, Dict, Optional, List, Callable
import re
import pandas as pd
from src.api.data_extraction import VariableDefinition
import itertools
from collections import defaultdict
from src.db.database import Database

class ValidationError(Exception):
    pass

class VariableValidator:
    @staticmethod
    def validate_string(value: str) -> str:
        return value

    @staticmethod
    def validate_numeric_float(value: str, allow_percentage=True, min_value=None, max_value=None) -> float:
        if allow_percentage and value.endswith('%'):
            numeric_part = value.rstrip('%')
            val = float(numeric_part) / 100.0
        else:
            val = float(value)

        if (min_value is not None) and (val < min_value):
            raise ValueError("Value below min_value")
        if (max_value is not None) and (val > max_value):
            raise ValueError("Value above max_value")
        return val

    @staticmethod
    def validate_numeric_int(value: str, min_value=None, max_value=None) -> int:
        val = int(value)
        if (min_value is not None) and (val < min_value):
            raise ValueError("Value below min_value")
        if (max_value is not None) and (val > max_value):
            raise ValueError("Value above max_value")
        return val

class ExtractionHandler:
    def __init__(self, variable_definitions, ensemble_id: str, db: Database):
        """
        :param variable_definitions: The subset of variables for this chunk (or all variables if unchunked).
        :param ensemble_id: The ensemble/group identifier.
        :param db: A reference to the database.
        
        This handler can parse the LLM's output for a set of variable definitions,
        validate each extracted value, and optionally handle required fields.
        """
        self.db = db
        self.ensemble_id = ensemble_id

        # Build a dictionary of all variables, keyed by variable name.
        self.all_variables = {}
        for var_def in variable_definitions:
            self.all_variables[var_def.name] = {
                "name": var_def.name,
                "description": var_def.description,
                "data_type": var_def.data_type,
                "formatting_instructions": var_def.formatting_instructions or "",
                "required": var_def.required,
                "csv_column": var_def.csv_column,
                "validation_rules": var_def.validation_rules or {}
            }

        self.prompt_variables = self.all_variables

        # Build validators from all variables so we can handle them in extract_and_validate
        self.validators = self._setup_validators()

    def _setup_validators(self) -> Dict[str, Callable]:
        validators = {}
        for name, var_def in self.all_variables.items():
            data_type = var_def.get("data_type", "string")
            rules = var_def.get("validation_rules", {})

            if data_type == "numeric_float":
                validators[name] = lambda x, r=rules: VariableValidator.validate_numeric_float(
                    x,
                    allow_percentage=r.get("allow_percentage", True),
                    min_value=r.get("min_value"),
                    max_value=r.get("max_value")
                )
            elif data_type == "numeric_int":
                validators[name] = lambda x, r=rules: VariableValidator.validate_numeric_int(
                    x,
                    min_value=r.get("min_value"),
                    max_value=r.get("max_value")
                )
            else:
                validators[name] = VariableValidator.validate_string
        return validators

    def generate_extraction_prompt(self, prompt: str) -> str:
        """
        Uses only self.prompt_variables so the user sees only the chunk's subset.
        """
        variable_instructions = []
        for i, var in enumerate(self.prompt_variables.values(), 1):
            instruction = f"{i}. {var['name']}: {var['description']}"
            if var.get("formatting_instructions"):
                instruction += f". {var['formatting_instructions']}"
            variable_instructions.append(instruction)

        variables_text = "\n".join(variable_instructions)

        return f"""# Variables to Extract
{variables_text}

Please extract these variables and format your response as follows:

[START_EXTRACTIONS]
{chr(10).join(f"{i+1}. {var['name']}: [extracted value]" for i, var in enumerate(self.prompt_variables.values()))}
[END_EXTRACTIONS]

{prompt}
"""

    def extract_and_validate(self, llm_response: str) -> Dict[str, Any]:
        """
        Parses the LLM response for all variables in self.all_variables, ensuring we
        handle any required variables that were not returned. This lets you
        evaluate the union of chunk + prior data if needed.
        """
        pattern = r"\[START_EXTRACTIONS\](.*?)\[END_EXTRACTIONS\]"
        blocks = re.findall(pattern, llm_response, re.DOTALL)
        if not blocks:
            raise ValidationError("No extraction blocks found in response")

        combined_extractions = defaultdict(list)

        for block in blocks:
            lines = block.strip().split("\n")
            for line in lines:
                line_stripped = line.strip()
                if ":" not in line_stripped:
                    continue

                key, value = line_stripped.split(":", 1)
                # Remove numeric prefix if present
                key = re.sub(r'^\d+\.\s*', '', key)
                # remove leading/trailing asterisks or markdown
                key = re.sub(r'[\*]+', '', key).strip()
                
                value = value.strip()

                # Validate against known variables
                if key in self.validators:
                    try:
                        validated_val = self.validators[key](value)
                        combined_extractions[key].append(validated_val)
                    except Exception:
                        # If a single line fails (e.g., invalid numeric), skip
                        continue

        # Merge extracted values
        final_extractions = {}
        for var_name, values in combined_extractions.items():
            if len(values) == 1:
                final_extractions[var_name] = values[0]
            else:
                final_extractions[var_name] = values

        # Handle required fields more gracefully:
        for var_name, var_def in self.all_variables.items():
            if var_def.get("required") and var_name not in final_extractions:
                # Instead of raising an error, store 'na' or some placeholder
                final_extractions[var_name] = "not found"

        return final_extractions
    
    # Optional: Evaluate logic to compare to ground-truth CSV columns
    def evaluate_ensemble_extractions(
        self,
        merged_data: Dict[str, Any],
        dataset_path: str,
        id_column: str,
        db: Database
    ) -> Dict[str, Any]:
        """
        Compare final merged extractions across all prompt_run_ids in the ensemble
        to ground-truth CSV columns in `dataset_path`. Returns a dict with
        "metrics" and "comparisons" keys.
        
        merged_data is the structure from RetrievalTask._assemble_results(ensemble_id)["results"],
        i.e. { doc_id: {"raw_responses": [...], "parsed_extractions": {...} } }.
        """
        df = pd.read_csv(dataset_path)
        df[id_column] = df[id_column].astype(str)

        metrics = {
            "total_articles": 0,
            "total_variables": 0,
            "correct": 0,
            "incorrect": 0,
            "fully_correct_rows": 0
        }
        comparisons_list = []

        # Build a quick doc_id -> row map for the CSV
        row_lookup = {}
        for idx, row in df.iterrows():
            row_id = str(row[id_column])
            row_lookup[row_id] = row

        # Go through each doc_id in merged_data
        for doc_id, doc_content in merged_data.items():
            # doc_content: { "raw_responses": [...], "parsed_extractions": {...} }
            parsed_extractions = doc_content.get("parsed_extractions", {})
            if doc_id not in row_lookup:
                # doc_id not found in CSV, skip
                continue

            metrics["total_articles"] += 1
            row_fully_correct = True
            csv_row = row_lookup[doc_id]

            # For each variable in self.all_variables
            for var_name, var_def in self.all_variables.items():
                if var_name not in parsed_extractions:
                    # Not extracted
                    continue

                # If there's no csv_column or no matching column in the CSV row, skip
                csv_col = var_def.get("csv_column")
                if not csv_col or csv_col not in csv_row:
                    continue

                extracted_val = str(parsed_extractions[var_name])
                ground_val = str(csv_row[csv_col])

                # Split on commas if multiple
                ex_list = [x.strip() for x in extracted_val.split(",") if x.strip()]
                gt_list = [x.strip() for x in ground_val.split(",") if x.strip()]

                # We'll do the same permutation-based comparison logic
                min_len = min(len(ex_list), len(gt_list))

                best_permutation_matches = []
                best_match_count = -1

                import itertools
                for perm in itertools.permutations(gt_list):
                    match_list = []
                    match_count = 0
                    for i in range(min_len):
                        ex_val = ex_list[i]
                        perm_val = perm[i]
                        matched = self._compare_single_val(
                            ex_val,
                            perm_val,
                            var_def.get("data_type", "string"),
                            var_def.get("validation_rules", {})
                        )
                        if matched:
                            match_count += 1
                        match_list.append((ex_val, perm_val, matched))

                    if match_count > best_match_count:
                        best_match_count = match_count
                        best_permutation_matches = match_list

                # Record the best matches
                for ex_val, best_gt_val, matched in best_permutation_matches:
                    metrics["total_variables"] += 1
                    if matched:
                        metrics["correct"] += 1
                    else:
                        metrics["incorrect"] += 1
                        row_fully_correct = False

                    comparisons_list.append({
                        "article_id": doc_id,
                        "variable_name": var_name,
                        "extracted_value": ex_val,
                        "ground_value": best_gt_val,
                        "matched": matched
                    })

                # leftover extracted items
                if len(ex_list) > min_len:
                    leftover = ex_list[min_len:]
                    for ex_val in leftover:
                        metrics["total_variables"] += 1
                        metrics["incorrect"] += 1
                        row_fully_correct = False
                        comparisons_list.append({
                            "article_id": doc_id,
                            "variable_name": var_name,
                            "extracted_value": ex_val,
                            "ground_value": "(no ground value)",
                            "matched": False
                        })

                # leftover ground items
                if len(gt_list) > min_len:
                    leftover = gt_list[min_len:]
                    for gt_val in leftover:
                        metrics["total_variables"] += 1
                        metrics["incorrect"] += 1
                        row_fully_correct = False
                        comparisons_list.append({
                            "article_id": doc_id,
                            "variable_name": var_name,
                            "extracted_value": "(no extracted value)",
                            "ground_value": gt_val,
                            "matched": False
                        })

            if row_fully_correct:
                metrics["fully_correct_rows"] += 1

        return {
            "metrics": metrics,
            "comparisons": comparisons_list
        }


    def _compare_single_val(
        self,
        extracted_val: str,
        ground_val: str,
        data_type: str,
        validation_rules: Dict[str, Any]
    ) -> bool:
        """
        Compare two single values. If numeric, apply tolerance; if string, direct match.
        """
        if data_type in ["numeric_float", "numeric_int"]:
            try:
                tolerance = float(validation_rules.get("tolerance", 0.0))
                # Remove percent sign if present
                ex_clean = extracted_val.replace("%", "").strip()
                gt_clean = ground_val.replace("%", "").strip()
                ex_num = float(ex_clean)
                gt_num = float(gt_clean)
                return abs(ex_num - gt_num) <= tolerance
            except ValueError:
                return False
        else:
            # String or other: direct equality with any needed strip
            return extracted_val.strip() == ground_val.strip()
