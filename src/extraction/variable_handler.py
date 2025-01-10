from typing import Any, Dict, Optional, List, Callable
import re
import pandas as pd
from src.api.data_extraction import VariableDefinition

class ValidationError(Exception):
    pass

class VariableValidator:
    @staticmethod
    def validate_string(value: str) -> str:
        return value.strip()
    
    @staticmethod
    def validate_numeric_float(
        value: str,
        allow_percentage: bool = True,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        """
        Basic numeric validation with minimal processing
        Format requirements should be handled via prompting
        """
        try:
            # Remove percentage signs if present
            if "%" in value and allow_percentage:
                value = value.replace("%", "")
                parsed_value = float(value) / 100
            else:
                parsed_value = float(value)
                
            # Only check extreme bounds if specified
            if min_value is not None and parsed_value < min_value:
                raise ValidationError(f"Value {parsed_value} is below minimum {min_value}")
            if max_value is not None and parsed_value > max_value:
                raise ValidationError(f"Value {parsed_value} is above maximum {max_value}")
                
            return parsed_value
        except ValueError:
            raise ValidationError(f"Could not convert '{value}' to a number")
    
    @staticmethod
    def validate_numeric_int(
        value: str,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> int:
        """
        Validate and return an integer value.
        """
        try:
            parsed_value = int(value)
            
            # Only check extreme bounds if specified
            if min_value is not None and parsed_value < min_value:
                raise ValidationError(f"Value {parsed_value} is below minimum {min_value}")
            if max_value is not None and parsed_value > max_value:
                raise ValidationError(f"Value {parsed_value} is above maximum {max_value}")
                
            return parsed_value
        except ValueError:
            raise ValidationError(f"Could not convert '{value}' to an integer")

class ExtractionHandler:
    def __init__(self, variables: List[VariableDefinition]):
        self.variables = {var.name: var for var in variables}
        self.validators = self._setup_validators()
        
    def generate_extraction_prompt(self, prompt: str) -> str:
        """Generate the formatted prompt with variable instructions"""
        variable_instructions = []
        for i, var in enumerate(self.variables.values(), 1):
            instruction = f"{i}. {var.name}: {var.description}"
            if var.formatting_instructions:
                instruction += f". {var.formatting_instructions}"
            variable_instructions.append(instruction)
            
        variables_text = "\n".join(variable_instructions)
        
        return f"""

# Variables to Extract
{variables_text}

Please extract these variables and format your response as follows:

[START_EXTRACTIONS]
{chr(10).join(f'{i+1}. {var.name}: [extracted value]' for i, var in enumerate(self.variables.values()))}
[END_EXTRACTIONS]

{prompt}
"""

    def _setup_validators(self) -> Dict[str, Callable]:
        validators = {}
        for name, var in self.variables.items():
            if var.data_type == "numeric_float":
                rules = var.validation_rules or {}
                validators[name] = lambda x, r=rules: VariableValidator.validate_numeric_float(
                    x,
                    allow_percentage=r.get("allow_percentage", True),
                    min_value=r.get("min_value"),
                    max_value=r.get("max_value")
                )
            elif var.data_type == "numeric_int":
                rules = var.validation_rules or {}
                validators[name] = lambda x, r=rules: VariableValidator.validate_numeric_int(
                    x,
                    min_value=r.get("min_value"),
                    max_value=r.get("max_value")
                )
            else:
                # default to string validator
                validators[name] = VariableValidator.validate_string
        return validators

    def extract_and_validate(self, llm_response: str) -> Dict[str, Any]:
        """
        Extract and perform basic validation on LLM response
        """
        # Extract data between markers
        pattern = r"\[START_EXTRACTIONS\](.*?)\[END_EXTRACTIONS\]"
        match = re.search(pattern, llm_response, re.DOTALL)
        if not match:
            raise ValidationError("No extraction block found in response")
            
        extraction_text = match.group(1).strip()
        
        # Parse extractions into dictionary
        extractions = {}
        for line in extraction_text.split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            # NEW: remove leading "<digits>. " if present
            key = re.sub(r'^\d+\.\s*', '', key)
            
            if key in self.validators:
                try:
                    extractions[key] = self.validators[key](value)
                except Exception as e:
                    raise ValidationError(f"Validation failed for {key}: {str(e)}")
        
        # Check for required fields
        for var_name, var_def in self.variables.items():
            if var_def.required and var_name not in extractions:
                raise ValidationError(f"Required field {var_name} missing from extraction")
                
        return extractions
    
    # Optional: Evaluate logic to compare to ground-truth CSV columns
    def evaluate_extractions(self, ensemble_id: str, dataset_path: str, id_column: str, db) -> Dict[str, Any]:
        """
        For each row in the CSV dataset:
          1) Retrieve the extracted variables 
          2) Split each variable's extraction and CSV ground truth by commas
          3) Combine them into tuples so that the i-th item from each variable is paired with the i-th item from each other variable
          4) Compare ignoring broad ordering (optional). If they mismatch, mark incorrect.
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

        # We’ll store details for each matched item here:
        comparisons_list = []

        for _, row in df.iterrows():
            article_id = row[id_column]
            var_extractions = db.retrieve_variable_extractions(ensemble_id, article_id)
            if not var_extractions:
                # Skip if no extractions
                continue

            metrics["total_articles"] += 1

            # Keep track of whether this entire row was fully correct
            row_fully_correct = True

            # Gather all variables for which we have a CSV column and an extracted value
            # e.g. { var_name: {"extracted": [...], "ground": [...], "data_type": ..., "rules": ...} }
            row_vars = {}
            for var_name, var_def in self.variables.items():
                if var_name in var_extractions and var_def.csv_column and var_def.csv_column in row:
                    extracted_val = str(var_extractions[var_name])
                    ground_val = str(row[var_def.csv_column])
                    row_vars[var_name] = {
                        "extracted": [x.strip() for x in extracted_val.split(",") if x.strip()],
                        "ground": [x.strip() for x in ground_val.split(",") if x.strip()],
                        "data_type": var_def.data_type,
                        "rules": var_def.validation_rules or {}
                    }

            # Skip if no overlapping variables in this row
            if not row_vars:
                continue

            # Instead of skipping rows at any mismatch, do partial matching
            for var_name, info in row_vars.items():
                ex_list = info["extracted"]
                gt_list = info["ground"]
                data_type = info["data_type"]
                rules = info["rules"]

                # Compare up to the smaller length
                min_len = min(len(ex_list), len(gt_list))
                for i in range(min_len):
                    metrics["total_variables"] += 1
                    ex_val = ex_list[i]
                    gt_val = gt_list[i]
                    matched = self._compare_single_val(ex_val, gt_val, data_type, rules)
                    if matched:
                        metrics["correct"] += 1
                    else:
                        metrics["incorrect"] += 1
                        row_fully_correct = False

                    comparisons_list.append({
                        "article_id": article_id,
                        "variable_name": var_name,
                        "extracted_value": ex_val,
                        "ground_value": gt_val,
                        "matched": matched
                    })

                # Handle leftover extracted items
                if len(ex_list) > min_len:
                    leftover = ex_list[min_len:]
                    for ex_val in leftover:
                        metrics["total_variables"] += 1
                        metrics["incorrect"] += 1
                        row_fully_correct = False
                        comparisons_list.append({
                            "article_id": article_id,
                            "variable_name": var_name,
                            "extracted_value": ex_val,
                            "ground_value": "(no ground value)",
                            "matched": False
                        })

                # Handle leftover ground items
                if len(gt_list) > min_len:
                    leftover = gt_list[min_len:]
                    for gt_val in leftover:
                        metrics["total_variables"] += 1
                        metrics["incorrect"] += 1
                        row_fully_correct = False
                        comparisons_list.append({
                            "article_id": article_id,
                            "variable_name": var_name,
                            "extracted_value": "(no extracted value)",
                            "ground_value": gt_val,
                            "matched": False
                        })

            if row_fully_correct:
                metrics["fully_correct_rows"] += 1

        # Return both metrics and comparisons
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