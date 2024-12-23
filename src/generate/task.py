from typing import List
import uuid
from src.api.data_extraction import DataExtractionRequest
from src.db.database import Database
from src.interfaces.interface import Interface
from src.interfaces.factory import get_interface
from src.srma_prompts.prompt import Prompt


class GenerationTask:

    model: Interface

    def __init__(self):
        self.db = Database()

    async def run(self, request: DataExtractionRequest):
        self.from_request(request)

        # these need to be replaced with actual papers/abstracts
        abstracts = ["abstract example 1", "abstract example 2"]
        abstract_ids = ["1", "2"]
        abstract_answers = ["answer 1", "answer 2"]

        # save abstract answers to the database. Once we retrieve the batch results, we'll assess the results
        self.db.store_multiple_abstract_answers(
            self.ensemble_id, abstract_ids, abstract_answers
        )

        prompts = self.generate_prompts(abstracts, abstract_ids)
        batch_ids = await self.generate_batches(prompts)

        # store batch ids to the database
        self.db.store_batch_ids(self.ensemble_id, batch_ids, self.model.model)
        print("Batch ids: {}, ensemble id: {}".format(batch_ids, self.ensemble_id))

    def from_request(self, request: DataExtractionRequest):
        self.model = get_interface(request.model)
        self.preprompt = request.preprompt
        self.prompt = request.prompt
        self.exclude_samples = request.exclude_samples
        self.include_samples = request.include_samples

        self.include_dataset = request.include_dataset
        self.exclude_dataset = request.exclude_dataset

        self.seed = request.seed

        self.ensemble = request.ensemble
        self.ensemble_threshold = request.ensemble_threshold
        self.ensemble_id = str(uuid.uuid4())[:8]

    def generate_prompt(self, abstract: str, abstract_id: str):
        return Prompt(abstract_id, self.preprompt, self.prompt, abstract)

    def generate_prompts(self, abstracts: List[str], abstract_ids: List[str]):
        return [
            self.generate_prompt(abstract, abstract_id)
            for abstract, abstract_id in zip(abstracts, abstract_ids)
        ]

    async def generate_batch(self, prompts: List[Prompt], ensemble_id: str):
        return await self.model.create_batch(
            prompts=[prompt.get_prompt() for prompt in prompts],
            prompts_ids=[prompt.abstract_id for prompt in prompts],
            ensemble_id=ensemble_id,
            seed=self.seed,
        )

    async def generate_batches(self, prompts: List[Prompt]):
        return [
            await self.generate_batch(prompts, self.ensemble_id)
            for _ in range(self.ensemble)
        ]
