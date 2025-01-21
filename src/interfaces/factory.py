# factory function
from src.interfaces.interface import Interface, OPENAI_MODELS, ANTHROPIC_MODELS, OPENAI_models_01
from src.interfaces.openai import OpenAIInterface
from src.interfaces.openai_01 import O1Interface
from src.interfaces.anthropic import AnthropicInterface


def get_interface(model: str) -> Interface:
    """Get the appropriate interface for the model"""
    if model in OPENAI_MODELS:
        return OpenAIInterface(model)
    elif model in ANTHROPIC_MODELS:
        return AnthropicInterface(model)
    elif model in OPENAI_models_01:
        return O1Interface(model)
    raise ValueError(f"Model {model} is not available. Available models: {OPENAI_MODELS | ANTHROPIC_MODELS | OPENAI_models_01}")
