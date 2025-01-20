from typing import Dict, List, Any, Optional
from src.db.database import Database
from src.interfaces.interface import BatchResponseInterface
from src.interfaces.factory import get_interface
from src.srma_prompts.result import Result
from src.extraction.variable_handler import ExtractionHandler, VariableDefinition

def chunk_list(items: List[str], chunk_size: int = 500) -> List[List[str]]:
    """
    Splits a list of items into smaller chunks of a specified size to avoid SQULite's 999 parameter limit

    Example:
        >>> chunk_list(['a','b','c','d','e'], 2)
        [['a','b'], ['c','d'], ['e']]
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

class RetrievalTask:
    def __init__(self):
        self.db = Database()
        self.variables = None  # Add variables as needed

    async def run(self, prompt_run_id: str) -> Dict[str, Any]:
        """
        Retrieves and processes extraction results for a given prompt_run_id.

        Steps:
            1. Check if retrieval for the prompt_run_id is already done.
            2. If done, assemble and return the merged results for the entire ensemble.
            3. If pending, perform retrieval from the model, store results, mark as done,
               and then assemble and return the merged results for the entire ensemble.

        Args:
            prompt_run_id (str): The ID of the prompt run to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the ensemble_id, status, and merged results.
        """
        if self.db.is_retrieval_done(prompt_run_id):
            print(f"Retrieval already done for prompt_run_id = {prompt_run_id}.")
            # Retrieve batch_info to get ensemble_id
            batch_info = self.db.retrieve_batch_info(prompt_run_id)
            if batch_info:
                ensemble_id = batch_info["ensemble_id"]
                # Assemble and return results for the entire ensemble
                return await self._assemble_results(ensemble_id)
            else:
                return {
                    "ensemble_id": None,
                    "status": "no_data",
                    "results": {}
                }

        # Retrieve batch information
        batch_info = self.db.retrieve_batch_info(prompt_run_id)
        if not batch_info:
            print(f"No batch_info found for prompt_run_id = {prompt_run_id}.")
            return {
                "ensemble_id": None,
                "status": "no_data",
                "results": {}
            }

        ensemble_id = batch_info["ensemble_id"]
        batch_ids: List[str] = batch_info["batch_ids"]
        model: str = batch_info["model"]

        # Set variables if they are stored in batch_info; otherwise, keep self.variables
        self.variables = self.variables or batch_info["variables"]

        # Retrieve and download from the model interface
        interface = get_interface(model)
        batches = await interface.retrieve_batches(batch_ids)
        batches_results = await interface.download_batches(batches)

        # Store results
        await self._process_results(
            prompt_run_id=prompt_run_id,
            sub_ensemble_id=batch_info["sub_ensemble_id"],
            ensemble_id=ensemble_id,
            batches_results=batches_results
        )
        self.db.mark_retrieved(prompt_run_id)

        # Assemble and return results for the entire ensemble
        return await self._assemble_results(ensemble_id)

    async def _process_results(
        self,
        prompt_run_id: str,
        sub_ensemble_id: str,
        ensemble_id: str,
        batches_results: List[List[BatchResponseInterface]]
    ) -> None:
        """
        Parses and stores extraction results from the model into the database.

        Args:
            prompt_run_id (str): The ID of the prompt run.
            sub_ensemble_id (str): The ID of the sub-ensemble.
            ensemble_id (str): The ID of the ensemble.
            batches_results (List[List[BatchResponseInterface]]): The raw responses from the model.
        """
        # retrieve.py, near `_process_results(...)`
        master_vars = self.db.retrieve_master_variables(ensemble_id) or []
        all_defs = [VariableDefinition(**mv) for mv in master_vars]

        extraction_handler = ExtractionHandler(
            variable_definitions=all_defs,
            ensemble_id=ensemble_id,
            db=self.db,
        )


        for batch_list in batches_results:
            for br in batch_list:
                article_id = br.id
                self.db.store_extraction_result(
                    prompt_run_id=prompt_run_id,
                    sub_ensemble_id=sub_ensemble_id,
                    ensemble_id=ensemble_id,
                    article_id=article_id,
                    extraction_text=br.response
                )

                # Here we call extraction_handler.extract, which will use master_variables
                try:
                    extractions = extraction_handler.extract_and_validate(br.response)
                    self.db.store_multiple_variable_extractions(
                        prompt_run_id=prompt_run_id,
                        sub_ensemble_id=sub_ensemble_id,
                        ensemble_id=ensemble_id,
                        article_id=article_id,
                        extractions=extractions
                    )
                except Exception as e:
                    print(f"Extraction parse failed for article_id={article_id}: {str(e)}")

    async def _assemble_results(self, ensemble_id: str) -> Dict[str, Any]:
        """
        Assembles and merges extraction results across all prompt_run_ids within an ensemble.

        Steps:
            1. Retrieve all prompt_run_ids associated with the ensemble_id.
            2. Retrieve all extraction_results and extracted_variables for these prompt_run_ids in bulk (with chunking if necessary).
            3. Merge the raw responses and parsed extractions for each document.
            4. Return a dictionary containing the ensemble_id, status, and merged results.

        Args:
            ensemble_id (str): The ID of the ensemble to assemble results for.

        Returns:
            Dict[str, Any]: A dictionary containing the ensemble_id, status, and merged results.
        """
        prompt_runs = self.db.find_prompt_runs_for_ensemble(ensemble_id)
        if not prompt_runs:
            return {
                "ensemble_id": ensemble_id,
                "status": "no_data",
                "results": {}
            }

        # Retrieve all doc_ids across prompt_runs
        doc_ids_union = set()
        for pr in prompt_runs:
            batch_info = self.db.retrieve_batch_info(pr)
            if batch_info and batch_info["doc_ids"]:
                doc_ids_union.update(batch_info["doc_ids"])

        if not doc_ids_union:
            return {
                "ensemble_id": ensemble_id,
                "status": "no_data",
                "results": {}
            }

        # Prepare placeholders and chunking to avoid parameter limit issues
        # e.g. chunk_size=500 or 999 depending on your database
        run_chunks = chunk_list(prompt_runs, 500)
        doc_chunks = chunk_list(list(doc_ids_union), 500)

        # We'll accumulate data across partial queries
        extraction_dict: Dict[str, List[str]] = {}
        extracted_vars_dict: Dict[str, Dict[str, str]] = {}

        # For each chunk combination, do partial retrieval
        for run_chunk in run_chunks:
            for doc_chunk in doc_chunks:
                placeholders_runs = ', '.join(['?'] * len(run_chunk))
                placeholders_docs = ', '.join(['?'] * len(doc_chunk))

                # 1) Bulk retrieval of extraction_results
                extraction_query = f"""
                    SELECT prompt_run_id, article_id, extraction_text
                    FROM extraction_results
                    WHERE prompt_run_id IN ({placeholders_runs})
                    AND article_id IN ({placeholders_docs})
                """
                extraction_params = run_chunk + doc_chunk
                self.db.cursor.execute(extraction_query, extraction_params)
                extraction_rows = self.db.cursor.fetchall()

                for row in extraction_rows:
                    article_id = row["article_id"]
                    extraction_text = row["extraction_text"]
                    extraction_dict.setdefault(article_id, []).append(extraction_text)

                # 2) Bulk retrieval of extracted_variables
                extracted_vars_query = f"""
                    SELECT prompt_run_id, article_id, variable_name, extracted_value
                    FROM extracted_variables
                    WHERE prompt_run_id IN ({placeholders_runs})
                    AND article_id IN ({placeholders_docs})
                """
                self.db.cursor.execute(extracted_vars_query, extraction_params)
                extracted_vars_rows = self.db.cursor.fetchall()

                for row in extracted_vars_rows:
                    article_id = row["article_id"]
                    var_name = row["variable_name"]
                    var_value = row["extracted_value"]
                   # Instead of overwriting unconditionally, let's only overwrite if the new value is better:
                    current_val = extracted_vars_dict.setdefault(article_id, {}).get(var_name)

                    # If current_val is empty (None or not found) and the new var_value is a real value, store it.
                    # Or if current_val was never set, store whatever we have now.
                    if (not current_val or current_val == "not found") and var_value != "not found":
                        extracted_vars_dict[article_id][var_name] = var_value
                    elif not current_val:
                        # If the current_val is None (key doesn't exist yet), store the new one (even if "not found")
                        extracted_vars_dict[article_id][var_name] = var_value


        # Now unify the final results for each doc
        final_results: Dict[str, Any] = {}
        for doc_id in doc_ids_union:
            combined_raw_responses = extraction_dict.get(doc_id, [])
            combined_extractions = extracted_vars_dict.get(doc_id, {})
            final_results[doc_id] = {
                "raw_responses": combined_raw_responses,
                "parsed_extractions": combined_extractions
            }

        return {
            "ensemble_id": ensemble_id,
            "status": "complete",
            "results": final_results
        }

    def _get_doc_ids(self, ensemble_id: str) -> List[str]:
        """
        Returns a list of all document/article IDs associated with the given ensemble_id.
        """
        self.db.cursor.execute(
            """
            SELECT DISTINCT article_id
            FROM extraction_results
            WHERE ensemble_id = ?
            """,
            (ensemble_id,)
        )
        rows = self.db.cursor.fetchall()
        return [row["article_id"] for row in rows]

    def _gather_extraction_text(self, ensemble_id: str) -> Dict[str, List[str]]:
        """
        Returns a dict mapping each article_id -> a list of raw extraction texts.
        """
        self.db.cursor.execute(
            """
            SELECT article_id, extraction_text
            FROM extraction_results
            WHERE ensemble_id = ?
            """,
            (ensemble_id,)
        )
        rows = self.db.cursor.fetchall()

        result_map: Dict[str, List[str]] = {}
        for row in rows:
            article_id = row["article_id"]
            text = row["extraction_text"]
            result_map.setdefault(article_id, []).append(text)
        return result_map

    def _gather_extracted_vars(self, ensemble_id: str) -> Dict[str, Dict[str, str]]:
        """
        Returns a dict mapping each article_id -> { variable_name: extracted_value }
        Overwrites duplicate variables if they appear multiple times.
        """
        self.db.cursor.execute(
            """
            SELECT article_id, variable_name, extracted_value
            FROM extracted_variables
            WHERE ensemble_id = ?
            """,
            (ensemble_id,)
        )
        rows = self.db.cursor.fetchall()

        vars_map: Dict[str, Dict[str, str]] = {}
        for row in rows:
            article_id = row["article_id"]
            var_name = row["variable_name"]
            var_value = row["extracted_value"]
            if article_id not in vars_map:
                vars_map[article_id] = {}
            vars_map[article_id][var_name] = var_value
        return vars_map
