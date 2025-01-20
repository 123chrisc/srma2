from typing import List

from pydantic import BaseModel

MAX_COMPLETION_TOKENS = 8192
# 16384 is the max for GPT-4o
# 8192 is the max for Claude 3.5 Sonnet

OPENAI_MODELS = set(
    [
        "o1-2024-12-17",
        "gpt-4-0125-preview",
        "gpt-4-turbo-2024-04-09",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
    ]
)

ANTHROPIC_MODELS = set(["claude-3-5-sonnet-20241022"])


class BatchInterface(BaseModel):
    id: str
    status: str
    created_at: int
    completed_at: int | None
    output_file_id: str
    metadata: object | dict[str, str]


class BatchResponseInterface(BaseModel):
    id: str
    response: str


class Interface:
    model: str

    available_models: set[str] = set()

    def __init__(self, model: str | None):
        if model:
            self.model = model
            self.verify_model()

    def verify_model(self) -> None:
        if not self.available_models:
            raise ValueError("No available models")
        if self.model not in self.available_models:
            raise ValueError(f"Model {self.model} is not available")

    async def create_batch(
        self, prompts: List[str], prompts_ids: List[str], prompt_run_id: str, seed: int
    ) -> str:
        raise NotImplementedError("create_batch is not implemented for this model")

    async def retrieve_batches(self, batch_ids: List[str]) -> List[BatchInterface]:
        raise NotImplementedError("retrieve_batches is not implemented for this model")

    async def download_batches(
        self, batches: List[BatchInterface]
    ) -> List[List[BatchResponseInterface]]:
        raise NotImplementedError("download_batches is not implemented for this model")
