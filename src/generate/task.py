import asyncio
from typing import List, Dict, Any
import uuid
from src.api.data_extraction import DataExtractionRequest
from src.db.database import Database
from src.interfaces.interface import Interface
from src.interfaces.factory import get_interface
from src.srma_prompts.prompt import Prompt
import pandas as pd
from src.extraction.variable_handler import ExtractionHandler, VariableDefinition
import json

def chunk_variables(variables: List[VariableDefinition], max_vars_per_chunk: int) -> List[List[VariableDefinition]]:
    chunks = []
    for i in range(0, len(variables), max_vars_per_chunk):
        chunks.append(variables[i : i + max_vars_per_chunk])
    return chunks

class GenerationTask:
    model: Interface

    def __init__(self):
        self.db = Database()
        # We define these as instance variables to store them if needed
        self.ensemble_id: str = ""
        self.sub_ensemble_id: str = ""
        self.model = None
        self.variables = []

    async def run(self, request: DataExtractionRequest) -> Dict[str, Any]:
        """
        Creates one or more prompt_run_ids (each referencing sub_ensemble_id + ensemble_id)
        based on chunking logic. Submits them to the LLM for batch creation, then
        stores the batch info in the DB.

        Steps:
          1) Derive (or reuse) ensemble_id and sub_ensemble_id from request or generate new.
          2) Optionally chunk variables if needed.
          3) For each chunk, create a new prompt_run_id, generate prompts, create a batch for the LLM.
          4) Store batch info in the DB, referencing prompt_run_id, sub_ensemble_id, ensemble_id, etc.
          5) Return a dictionary with "ensemble_id", "created_prompt_runs", ...
        """
        # Initialize from request
        self.from_request(request)

        # If user-supplied ensemble_id/sub_ensemble_id is not provided, generate new shorter ones
        self.ensemble_id = str(uuid.uuid4())[:8]
        self.sub_ensemble_id = str(uuid.uuid4())[:8]

        # Store all variables as JSON for later reference (optional)
        all_vars_json = json.dumps([v.dict() for v in self.variables])
        self.db.store_master_variables(self.ensemble_id, all_vars_json)

        # Now actually chunk the variables from the request
        max_vars_per_chunk = 50
        variable_chunks = chunk_variables(self.variables, max_vars_per_chunk)

        # Load the dataset content
        df = pd.read_csv(request.dataset_path)
        content = df[request.content_column].tolist()
        article_ids = df[request.id_column].astype(str).tolist()

        # Create a prompt_run_id for each chunk, generate prompts, and store
        created_prompt_runs = []
        chunk_index = 0

        for chunk_vars in variable_chunks:
            chunk_index += 1
            prompt_run_id = str(uuid.uuid4())[:8]

            extraction_handler = ExtractionHandler(
                variable_definitions=chunk_vars,
                ensemble_id=self.ensemble_id,
                db=self.db,
            )

            # Build out the final chunk-specific prompt
            chunk_prompt = extraction_handler.generate_extraction_prompt(self.prompt)

            # Create Prompt objects for each article
            chunk_prompts = self.generate_prompts(content, article_ids, self.preprompt, chunk_prompt)

            # Example: run some async job submission
            batch_ids = []
            for _ in range(self.ensemble):
                try:
                    batch_id = await self.generate_batch(chunk_prompts, prompt_run_id)
                    batch_ids.append(batch_id)
                except Exception as exc:
                    print(f"Batch generation failed for chunk {chunk_index}: {exc}")

            if not batch_ids:
                raise ValueError(f"No valid batch IDs were generated for chunk {chunk_index}.")

            # Store info about this chunk in the DB
            self.db.store_batch_info(
                prompt_run_id=prompt_run_id,
                sub_ensemble_id=self.sub_ensemble_id,
                ensemble_id=self.ensemble_id,
                batch_ids=batch_ids,
                model=self.model.model,
                variables=chunk_vars,
                doc_ids=article_ids,
                dataset_path=request.dataset_path,
                preprompt=self.preprompt,
                prompt=chunk_prompt
            )

            created_prompt_runs.append({
                "prompt_run_id": prompt_run_id,
                "batch_ids": batch_ids
            })
            print(f"[CHUNK {chunk_index}] Stored batch_info for prompt_run_id={prompt_run_id}")

        # Return overall metadata
        return {
            "ensemble_id": self.ensemble_id,
            "sub_ensemble_id": self.sub_ensemble_id,
            "created_prompt_runs": created_prompt_runs
        }

    def from_request(self, request: DataExtractionRequest):
        """
        Initializes the model interface and other instance attributes
        from the request object.
        """
        self.model = get_interface(request.model)
        self.ensemble = request.ensemble
        self.ensemble_threshold = request.ensemble_threshold
        self.preprompt = request.preprompt
        self.prompt = request.prompt
        self.variables = request.variables
        self.seed = request.seed
        # self.ensemble_id is set later
        # self.sub_ensemble_id is set later

    def generate_prompts(self, content: List[str], article_ids: List[str],
                         preprompt: str, prompt_text: str) -> List[Prompt]:
        """
        Generate Prompt objects for each article. We combine the user's
        preprompt, prompt_text, and article content into a single string.
        """
        prompts = []
        for article_id, article_text in zip(article_ids, content):
            p = Prompt(
                abstract_id=article_id,
                preprompt=preprompt,
                prompt=prompt_text,
                content=article_text
            )
            prompts.append(p)
        return prompts

    async def generate_batch(self, prompts: List[Prompt], prompt_run_id: str) -> str:
        """
        Generate a single batch with validation and retries. We
        pass 'ensemble_id' here as it will be used in the model interface calls.
        The model is set in from_request().
        """
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                batch_id = await self.model.create_batch(
                    prompts=[prompt.get_prompt() for prompt in prompts],
                    prompts_ids=[prompt.abstract_id for prompt in prompts],
                    prompt_run_id=prompt_run_id,  # might use prompt_run_id or any ID you prefer
                    seed=self.seed,
                )

                # Validate file_id
                if not batch_id or not isinstance(batch_id, str) or len(batch_id.strip()) == 0:
                    raise ValueError(f"Invalid batch_id received from LLM: {batch_id}")

                return batch_id

            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise
                print(f"Batch creation attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(retry_delay * (attempt + 1))

        # Should never reach here due to the raise above, but just in case
        raise RuntimeError("Failed to create batch after max retries.")
