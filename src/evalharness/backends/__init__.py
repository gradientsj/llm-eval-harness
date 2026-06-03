"""Judge backends: a real Anthropic-powered judge and a deterministic
lexical-overlap baseline."""

from .base import JudgeBackend, JudgeRequest
from .lexical import LexicalBackend

__all__ = ["JudgeBackend", "JudgeRequest", "LexicalBackend", "create_backend"]


def create_backend(name: str) -> JudgeBackend:
    """Factory used by the CLI. Imports the Anthropic backend lazily so the
    lexical path (tests, CI) never needs the SDK to be importable or a key
    set."""
    if name == "lexical":
        return LexicalBackend()
    if name == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend()
    raise ValueError(f"unknown backend {name!r}; expected 'lexical' or 'anthropic'")
