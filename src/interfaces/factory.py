# factory function
from src.interfaces.interface import Interface
from src.interfaces.openai import OpenAIInterface
from src.interfaces.anthropic import AnthropicInterface
from src.interfaces.interface import OPENAI_MODELS, ANTHROPIC_MODELS


def get_interface(model: str) -> Interface:
    if model in OPENAI_MODELS:
        return OpenAIInterface(model)
    elif model in ANTHROPIC_MODELS:
        return AnthropicInterface(model)
    raise ValueError(f"Model {model} is not available")
