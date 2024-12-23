from typing import Optional
from pydantic import BaseModel


class DataExtractionRequest(BaseModel):
    # abstract_in_investigation: Optional[str] = None
    # abstract_in_investigation_actual_value: Optional[str] = None

    model: str = "gemini-pro"
    preprompt: str
    prompt: str

    include_samples: int
    exclude_samples: int

    # gpt_cot_id: Optional[str] = None
    # few_shot_exclude: int = 0
    # few_shot_include: int = 0

    include_dataset: str
    exclude_dataset: str

    seed: int = 1
    ensemble: int = 1
    ensemble_threshold: Optional[int] = None
