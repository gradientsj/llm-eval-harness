"""Anthropic-powered judge backend.

Uses temperature 0 and a fixed model snapshot so judge scores are as
repeatable as the API allows. Model is overridable via EVALHARNESS_JUDGE_MODEL
so judge-model upgrades are an explicit, calibration-gated change rather than
a silent drift.
"""

from __future__ import annotations

import os

from .base import JudgeRequest

DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        import anthropic

        self.model = model or os.environ.get("EVALHARNESS_JUDGE_MODEL") or DEFAULT_MODEL
        # Reads ANTHROPIC_API_KEY from the environment.
        self._client = anthropic.Anthropic()

    def complete(self, request: JudgeRequest) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=0.0,
            system=request.system,
            messages=[{"role": "user", "content": request.user}],
        )
        return "".join(block.text for block in message.content if block.type == "text")
