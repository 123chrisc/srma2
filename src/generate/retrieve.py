from typing import Dict, List
from src.db.database import Database
from src.interfaces.interface import BatchResponseInterface
from src.interfaces.factory import get_interface
from src.srma_prompts.result import Result


class RetrievalTask:
    def __init__(self):
        self.db = Database()

    async def run(self, ensemble_id: str):
        ensemble = self.db.retrieve_batch_ids(ensemble_id)
        batch_ids: List[str] = ensemble["batch_ids"]
        model: str = ensemble["model"]
        batches_results = await self.retrieve_batches(batch_ids, model)
        processed_results = await self.process_results(ensemble_id, batches_results)
        results = await self.assess_results(processed_results)

        return processed_results

    # Run for all the batches associated with an ensemble
    async def retrieve_batches(self, batch_ids: List[str], model: str):
        interface = get_interface(model)
        batches_results: List[List[BatchResponseInterface]] = []

        batches = await interface.retrieve_batches(batch_ids)
        batches_results = await interface.download_batches(batches)
        return batches_results

    async def process_results(
        self, ensemble_id: str, batches_results: List[List[BatchResponseInterface]]
    ) -> List[List[Result]]:
        answers: Dict[str, str] = {}

        answers.update(
            self.db.retrieve_multiple_abstract_answers(
                ensemble_id, list(map(lambda x: x.id, batches_results[0]))
            )
        )

        ensemble: List[List[Result]] = []
        for batch_results in batches_results:
            ret = [Result(result.response, answers) for result in batch_results]
            ensemble.append(ret)
        return ensemble

    async def assess_results(self, results: List[List[Result]]):
        # Implement number correct/wrong/specificity etc.
        pass
