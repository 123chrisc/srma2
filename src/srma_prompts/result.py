from typing import Dict, Any
from src.extraction.variable_handler import ExtractionHandler

class Result:
    def __init__(self, response: str, article_id: str):
        self.response = response
        self.article_id = article_id
        self.extractions = {}

    def extract(self, extraction_handler: ExtractionHandler) -> Dict[str, Any]:
        """
        Parse the variables when explicitly called
        """
        self.extractions = extraction_handler.extract_and_validate(self.response)
        return self.extractions