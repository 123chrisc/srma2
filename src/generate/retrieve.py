from typing import Dict, List, Any
from src.db.database import Database
from src.interfaces.interface import BatchResponseInterface
from src.interfaces.factory import get_interface
from src.srma_prompts.result import Result
from src.extraction.variable_handler import ExtractionHandler


class RetrievalTask:
    def __init__(self):
        self.db = Database()
        self.variables = None  # Add variables as needed

    async def run(self, ensemble_id: str):
        """
        1) Check if we've already done retrieval for this ensemble_id (avoid duplicate cost).
        2) If retrieval is pending, actually download from the model.
        3) If retrieval is done, skip model calls and just return results from DB.
        """
        # Check if retrieval was already performed
        if self.db.is_retrieval_done(ensemble_id):
            print(f"Retrieval already done for ensemble_id = {ensemble_id}.")
            return self._assemble_results(ensemble_id)

        # Otherwise, do the retrieval
        batch_info = self.db.retrieve_batch_info(ensemble_id)
        if not batch_info:
            return []  

        batch_ids: List[str] = batch_info["batch_ids"]
        model: str = batch_info["model"]

        # Set variables if they are stored in batch_info; otherwise, keep self.variables
        self.variables = self.variables or batch_info["variables"]

        # (A) Retrieve and download from the model interface
        interface = get_interface(model)
        batches = await interface.retrieve_batches(batch_ids)
        batches_results = await interface.download_batches(batches)

        # (B) Process the results (extract + store)
        processed_results = await self.process_results(ensemble_id, batches_results)
        self.db.mark_retrieved(ensemble_id)
        return processed_results

    async def process_results(
        self,
        ensemble_id: str,
        batches_results: List[List[BatchResponseInterface]]
    ) -> List[List[Result]]:
        if not self.variables:
            raise ValueError("Variables must be set before processing results")
        
        extraction_handler = ExtractionHandler(self.variables)
        ensemble: List[List[Result]] = []
        
        for batch_results in batches_results:
            batch_processed = []
            for result in batch_results:
                result_obj = Result(result.response, result.id)
                self.db.store_extraction_result(
                    ensemble_id=ensemble_id,
                    article_id=result.id,
                    extraction_text=result.response
                )
                try:
                    extractions = result_obj.extract(extraction_handler)
                    self.db.store_multiple_variable_extractions(
                        ensemble_id=ensemble_id,
                        article_id=result.id,
                        extractions=extractions
                    )
                except Exception as e:
                    print(f"Failed to parse extractions for {result.id}: {str(e)}")
                batch_processed.append(result_obj)
            ensemble.append(batch_processed)
        return ensemble

    def _assemble_results(self, ensemble_id: str) -> List[List[Result]]:
        # Optional: load from DB to build a list of lists of Result objects
        return []
