"""LLM layer: provider-agnostic protocol + thin native adapters.

Structured output via tool-use (the LLM emits the Program AST as a `plan` tool
call). Mock enables offline closed-loop/CI. No LangChain — raw SDKs behind a
Protocol. Does NOT import engine.
"""
import os

from . import providers
from .protocol import Directive, Feedback, LLMProvider, TurnInput, TurnOutput
from .providers import AnthropicProvider, MockProvider, OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "Directive",
    "Feedback",
    "LLMProvider",
    "MockProvider",
    "OpenAIProvider",
    "TurnInput",
    "TurnOutput",
    "make_provider",
    "providers",
]


def make_provider(name: str, **kwargs) -> LLMProvider:
    """Build a provider by name: mock | anthropic | claude | openai | deepseek."""
    key = name.lower()
    if key == "mock":
        return MockProvider(**kwargs)
    if key in ("anthropic", "claude"):
        return AnthropicProvider(**kwargs)
    if key == "openai":
        return OpenAIProvider(**kwargs)
    if key == "deepseek":
        kwargs.setdefault("base_url", "https://api.deepseek.com")
        kwargs.setdefault("model", "deepseek-chat")
        kwargs.setdefault("api_key", os.getenv("DEEPSEEK_API_KEY"))
        return OpenAIProvider(**kwargs)
    raise ValueError(f"unknown provider {name!r}")
