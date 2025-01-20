import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from openai import AsyncOpenAI
from openai.types.batch import Batch

from src.interfaces.interface import (
    MAX_COMPLETION_TOKENS,
    OPENAI_MODELS,
    BatchInterface,
    BatchResponseInterface,
    Interface,
)


class OpenAIInterface(Interface):
    client: AsyncOpenAI
    available_models = OPENAI_MODELS

    def __init__(self, model: str | None = None):
        super().__init__(model)
        self.client = AsyncOpenAI()

    # prompt_ids should be a list of ints mapped to the prompts
    async def create_batch(
        self, prompts: List[str], prompts_ids: List[str], prompt_run_id: str, seed: int
    ):
        lines: List[Dict[str, Any]] = []
        for i, prompt in enumerate(prompts):
            lines.append(
                {
                    "custom_id": prompts_ids[i],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "messages": [{"role": "user", "content": prompt}],
                        "model": self.model,
                        "max_tokens": MAX_COMPLETION_TOKENS,
                        "seed": seed,
                    },
                }
            )
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".jsonl", prefix="generation_task_"
        ) as f:
            jsonl = "\n".join([json.dumps(line) for line in lines])
            f.write(jsonl.encode())
            f.flush()

            response = await self.client.files.create(
                file=Path(f.name), purpose="batch"
            )
            input_file_id = response.id
            batch = await self.client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"prompt_run_id": str(prompt_run_id)},
            )
            print("Batch created", batch.id)
            return batch.id

    def to_batch(self, batch: Batch):
        return BatchInterface(
            id=batch.id,
            status=batch.status,
            created_at=batch.created_at,
            completed_at=batch.completed_at or 0,
            output_file_id=batch.output_file_id or "",
            metadata=batch.metadata or {},
        )

    async def retrieve_batches(self, batch_ids: List[str]):
        batches = await asyncio.gather(
            *[self.client.batches.retrieve(batch_id) for batch_id in batch_ids]
        )
        return [self.to_batch(batch) for batch in batches]

    def to_result(self, result: Dict[str, Any]):
        return BatchResponseInterface(
            id=result["custom_id"],
            response=result["response"]["body"]["choices"][0]["message"]["content"],
        )

    async def download_batches(self, batches: List[BatchInterface]):
        files = await asyncio.gather(
            *[self.client.files.content(batch.output_file_id) for batch in batches]
        )
        batches_results = list(
            map(lambda x: [json.loads(line) for line in x.iter_lines()], files)
        )
        return [
            [self.to_result(batch_result) for batch_result in batch_results]
            for batch_results in batches_results
        ]
