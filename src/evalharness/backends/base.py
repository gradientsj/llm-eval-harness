"""Backend interface for the judge.

A backend receives the fully rendered prompts plus the structured fields they
were rendered from. Real LLM backends use the prompts; the deterministic
lexical baseline uses the structured fields directly so it never has to parse
prompt text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class JudgeRequest:
    system: str
    user: str
    # Structured fields the prompts were rendered from.
    context: str
    question: str
    answer: str


@runtime_checkable
class JudgeBackend(Protocol):
    name: str

    def complete(self, request: JudgeRequest) -> str:
        """Return the raw model output (expected to contain one JSON object)."""
        ...
