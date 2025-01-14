import asyncio
from typing import List, Dict
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
        
        # Generate batches concurrently
        batch_tasks = [
            self.generate_batch(prompts, self.ensemble_id)
            for _ in range(self.ensemble)
        ]
        
        # Wait for all batches to complete or fail
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Filter out failures and collect valid batch_ids
        batch_ids = []
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"Batch generation failed: {str(result)}")
            else:
                batch_ids.append(result)

        if not batch_ids:
            raise ValueError("No valid batch IDs were generated")

        # Store everything in the DB with status
        self.db.store_batch_info(
            ensemble_id=self.ensemble_id,
            batch_ids=batch_ids,
            model=self.model.model,
            variables=request.variables,
            doc_ids=article_ids,
            dataset_path=request.dataset_path,
            preprompt=request.preprompt,
            prompt=request.prompt
        )
        
        print(f"Successfully generated {len(batch_ids)} batches")
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
        """Generate a single batch with validation and retries"""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                batch_id = await self.model.create_batch(
                    prompts=[prompt.get_prompt() for prompt in prompts],
                    prompts_ids=[prompt.abstract_id for prompt in prompts],
                    ensemble_id=ensemble_id,
                    seed=self.seed,
                )
                
                # Validate file_id
                if not batch_id or not isinstance(batch_id, str) or len(batch_id.strip()) == 0:
                    raise ValueError(f"Invalid batch_id received from OpenAI: {batch_id}")
                
                return batch_id
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise  # Re-raise the last exception
                print(f"Batch creation attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
