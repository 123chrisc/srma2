from typing import List
from anthropic import Anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request
from anthropic.types.beta.messages.beta_message_batch import BetaMessageBatch
from anthropic.types.beta.messages.beta_message_batch_individual_response import (
    BetaMessageBatchIndividualResponse,
)
from src.interfaces.interface import (
    ANTHROPIC_MODELS,
    MAX_COMPLETION_TOKENS,
    BatchInterface,
    BatchResponseInterface,
    Interface,
)


class AnthropicInterface(Interface):
    client: Anthropic
    available_models = ANTHROPIC_MODELS

    def __init__(self, model: str | None = None):
        super().__init__(model)
        self.client = Anthropic()

    # prompt_ids should be a list of ints mapped to the prompts
    async def create_batch(
        self, prompts: List[str], prompts_ids: List[str], ensemble_id: str, seed: int
    ):
        requests: List[Request] = []
        for i, prompt in enumerate(prompts):
            requests.append(
                Request(
                    custom_id=str(prompts_ids[i]),
                    params=MessageCreateParamsNonStreaming(
                        model=self.model,
                        max_tokens=MAX_COMPLETION_TOKENS,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                )
            )

        batch = self.client.beta.messages.batches.create(
            requests=requests,
        )
        print("Batch created", batch.id)
        return batch.id

    def to_batch(self, batch: BetaMessageBatch):
        return BatchInterface(
            id=batch.id,
            status=batch.processing_status,
            created_at=int(batch.created_at.timestamp()),
            completed_at=int(batch.ended_at.timestamp()) if batch.ended_at else None,
            output_file_id=batch.id,
            metadata={},
        )

    async def retrieve_batches(self, batch_ids: List[str]) -> List[BatchInterface]:
        batches = [
            self.client.beta.messages.batches.retrieve(batch_id)
            for batch_id in batch_ids
        ]
        return [self.to_batch(batch) for batch in batches]

    def to_result(self, result: BetaMessageBatchIndividualResponse):
        return BatchResponseInterface(
            id=result.custom_id, response=result.result.message.content[0].text  # type: ignore
        )

    async def download_batches(self, batches: List[BatchInterface]):
        batches_results = [
            [line for line in self.client.beta.messages.batches.results(batch.id)]
            for batch in batches
        ]
        return [
            [self.to_result(batch_result) for batch_result in batch_results]
            for batch_results in batches_results
        ]
