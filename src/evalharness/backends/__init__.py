"""Judge backends: a real Anthropic-powered judge and a deterministic mock."""

from .base import JudgeBackend, JudgeRequest
from .mock import MockBackend

__all__ = ["JudgeBackend", "JudgeRequest", "MockBackend", "create_backend"]


def create_backend(name: str) -> JudgeBackend:
    """Factory used by the CLI. Imports the Anthropic backend lazily so the
    mock path (tests, CI) never needs the SDK to be importable or a key set."""
    if name == "mock":
        return MockBackend()
    if name == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend()
    raise ValueError(f"unknown backend {name!r}; expected 'mock' or 'anthropic'")
