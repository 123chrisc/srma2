from typing import List
import uuid
from src.api.data_extraction import DataExtractionRequest
from src.db.database import Database
from src.interfaces.interface import Interface
from src.interfaces.factory import get_interface
from src.srma_prompts.prompt import Prompt
import pandas as pd

class GenerationTask:

    model: Interface

    def __init__(self):
        self.db = Database()

    async def run(self, request: DataExtractionRequest):
        self.from_request(request)

        df = pd.read_csv(request.dataset_path)
        content = df[request.content_column].tolist()
        article_ids = df[request.id_column].astype(str).tolist()

        # Generate prompts
        prompts = self.generate_prompts(content, article_ids)
        batch_ids = await self.generate_batches(prompts)

        # Store everything in the DB
        self.db.store_batch_info(
            ensemble_id=self.ensemble_id,
            batch_ids=batch_ids,
            model=self.model.model,
            variables=request.variables,
            doc_ids=article_ids,            # store CSV article IDs
            dataset_path=request.dataset_path,
            preprompt=request.preprompt,
            prompt=request.prompt
        )
        print(f"Batch ids: {batch_ids}, ensemble id: {self.ensemble_id}")
        return self.ensemble_id

    def from_request(self, request: DataExtractionRequest):
        self.model = get_interface(request.model)
        self.ensemble = request.ensemble
        self.ensemble_threshold = request.ensemble_threshold
        self.preprompt = request.preprompt
        self.prompt = request.prompt

        self.variables = request.variables

        self.seed = request.seed
        self.ensemble = request.ensemble
        self.ensemble_threshold = request.ensemble_threshold
        self.ensemble_id = str(uuid.uuid4())[:8]

    def generate_prompts(self, content: List[str], article_ids: List[str]) -> List[Prompt]:
        """Generate prompts for each article"""
        prompts = []
        for article_id, article_text in zip(article_ids, content):
            prompt = Prompt(
                abstract_id=article_id,
                preprompt=self.preprompt,
                prompt=self.prompt,
                content=article_text,
            )
            prompts.append(prompt)
        return prompts

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
