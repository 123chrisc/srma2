from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class VariableDefinition(BaseModel):
    name: str
    description: str
    data_type: str  # "string" or "numeric_int" or "numeric_float"
    formatting_instructions: Optional[str] = None
    required: bool = True
    csv_column: Optional[str] = None
    # numeric-specific rules if wanted
    validation_rules: Optional[Dict[str, Any]] = None


class DataExtractionRequest(BaseModel):
    model: str = "gpt-4o-2024-11-20"
    preprompt: str
    prompt: str
    variables: List[VariableDefinition]
    dataset_path: str  # Path to CSV file
    content_column: str  # Column name for content to analyze
    id_column: str  # Column name for article IDs
    seed: int = 1
    ensemble: int = 1
    ensemble_threshold: Optional[int] = None