from typing import Dict, Any, List
from src.interfaces.factory import get_interface
from src.extraction.variable_handler import ExtractionHandler
from src.api.data_extraction import VariableDefinition

class AIEvaluator:
    def __init__(self, variable_definitions: List[VariableDefinition], ensemble_id: str):
        self.variable_definitions = variable_definitions
        self.ensemble_id = ensemble_id
        self.model = get_interface("gpt-4-0125-preview")  # Using GPT-4 Turbo

    def _generate_evaluation_prompt(self, var_def: VariableDefinition, extracted_val: str, ground_val: str, context: Dict[str, str]) -> str:
        return f"""Given the following context and variable definition, determine if the extracted value matches the ground truth value.

Context:
Preprompt: {context.get('preprompt', 'N/A')}
Variable Information: {context.get('variable_info', 'N/A')}

Variable Definition:
Name: {var_def.name}
Description: {var_def.description}
Data Type: {var_def.data_type}
Formatting Instructions: {var_def.formatting_instructions or 'N/A'}

Comparison:
Extracted Value: {extracted_val}
Ground Truth Value: {ground_val}

Are these values equivalent? Consider meaning of words, formatting variations, and numerical equivalence where appropriate.
Please think through your answer carefully as if you were a human reviewer. If you're uncertain about the match, treat it as not matching.

Return your response in the following JSON format:
{{
    "matched": boolean,  // true if values match, false if values do not match
    "reasoning": string  // detailed explanation of your decision
}}"""

    async def evaluate_single_variable(self, var_def: VariableDefinition, extracted_val: str, ground_val: str, context: Dict[str, str]) -> Dict[str, Any]:
        prompt = self._generate_evaluation_prompt(var_def, extracted_val, ground_val, context)
        
        response = await self.model.client.chat.completions.create(
            model=self.model.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={ "type": "json_object" }
        )

        try:
            import json
            response_json = json.loads(response.choices[0].message.content)
            
            return {
                "matched": response_json["matched"],
                "explanation": response_json["reasoning"],
                "extracted_value": extracted_val,
                "ground_value": ground_val
            }
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback in case of JSON parsing error
            return {
                "matched": False,
                "explanation": f"Error parsing GPT response: {str(e)}",
                "extracted_value": extracted_val,
                "ground_value": ground_val
            }

    async def evaluate_extractions(self, 
        merged_data: Dict[str, Any],
        dataset_path: str,
        id_column: str,
        context: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Evaluate extractions using GPT-4 for semantic comparison.
        """
        import pandas as pd
        print(f"Loading CSV from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Case-insensitive column matching for ID
        column_map = {col.lower(): col for col in df.columns}
        actual_id_column = column_map.get(id_column.lower())
        
        if not actual_id_column:
            available_columns = ", ".join(df.columns)
            print(f"ID column '{id_column}' not found. Available columns: {available_columns}")
            return {
                "status": "error",
                "message": f"ID column '{id_column}' not found in CSV. Available columns: {available_columns}",
                "metrics": {},
                "comparisons": []
            }

        # Convert ID column to string type
        df[actual_id_column] = df[actual_id_column].astype(str)
        print(f"Found {len(df)} rows in CSV")

        metrics = {
            "total_articles": 0,
            "total_variables": 0,
            "correct": 0,
            "incorrect": 0,
            "fully_correct_rows": 0
        }
        comparisons_list = []

        # Build doc_id -> row map for CSV
        row_lookup = {str(row[actual_id_column]): row for idx, row in df.iterrows()}
        print(f"CSV IDs: {list(row_lookup.keys())[:5]}...")

        # Debug merged data structure
        print(f"Merged data keys: {list(merged_data.keys())}")
        if "results" in merged_data:
            print(f"Document IDs in results: {list(merged_data['results'].keys())[:5]}...")
            for doc_id, content in list(merged_data['results'].items())[:2]:
                print(f"\nExample document {doc_id}:")
                print(f"Keys in content: {list(content.keys())}")
                if "parsed_extractions" in content:
                    print(f"Extracted variables: {list(content['parsed_extractions'].keys())}")

        # Process each document
        for doc_id, doc_content in merged_data.items():
            parsed_extractions = doc_content.get("parsed_extractions", {})
            if doc_id not in row_lookup:
                print(f"Document ID {doc_id} not found in CSV")
                continue

            metrics["total_articles"] += 1
            row_fully_correct = True
            csv_row = row_lookup[doc_id]

            # Evaluate each variable
            for var_def in self.variable_definitions:
                if var_def.name not in parsed_extractions:
                    print(f"Variable {var_def.name} not found in extractions for doc {doc_id}")
                    continue

                # Case-insensitive column matching for variable columns too
                csv_col = var_def.csv_column
                if csv_col:
                    actual_csv_col = column_map.get(csv_col.lower())
                    if not actual_csv_col:
                        print(f"CSV column {csv_col} not found for variable {var_def.name}")
                        continue
                else:
                    continue

                extracted_val = str(parsed_extractions[var_def.name])
                ground_val = str(csv_row[actual_csv_col])
                print(f"\nComparing for {var_def.name}:")
                print(f"Extracted: {extracted_val}")
                print(f"Ground truth: {ground_val}")

                # Use GPT-4 to evaluate
                evaluation = await self.evaluate_single_variable(
                    var_def,
                    extracted_val,
                    ground_val,
                    context
                )

                metrics["total_variables"] += 1
                if evaluation["matched"]:
                    metrics["correct"] += 1
                else:
                    metrics["incorrect"] += 1
                    row_fully_correct = False

                comparisons_list.append({
                    "article_id": doc_id,
                    "variable_name": var_def.name,
                    "extracted_value": extracted_val,
                    "ground_value": ground_val,
                    "matched": evaluation["matched"],
                    "explanation": evaluation["explanation"]
                })

            if row_fully_correct:
                metrics["fully_correct_rows"] += 1

        print(f"\nFinal metrics: {metrics}")
        print(f"Total comparisons: {len(comparisons_list)}")

        return {
            "status": "success",
            "metrics": metrics,
            "comparisons": comparisons_list
        } 