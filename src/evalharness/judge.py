"""LLM-as-judge: prompt rendering, output parsing, and scoring.

The judge emits a single JSON object per example. Parsing is deliberately
strict about content (all rubric fields present, scores in range) but lenient
about packaging (code fences, leading/trailing prose), because real models
occasionally wrap their JSON despite instructions. A parse failure triggers
one retry with a terser instruction before raising.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .backends.base import JudgeBackend, JudgeRequest
from .dataset import Example
from .rubric import DIMENSIONS, MAX_SCORE, MIN_SCORE, RUBRIC_TEXT


class JudgeParseError(ValueError):
    """Raised when the judge's output cannot be parsed into a valid score."""


@dataclass(frozen=True)
class JudgeScore:
    groundedness: int
    relevance: int
    coherence: int
    overall_pass: bool
    rationale: str

    def dimension(self, name: str) -> int:
        return int(getattr(self, name))

    def to_dict(self) -> dict:
        return {
            "groundedness": self.groundedness,
            "relevance": self.relevance,
            "coherence": self.coherence,
            "overall_pass": self.overall_pass,
            "rationale": self.rationale,
        }


SYSTEM_PROMPT = (
    "You are a strict, consistent evaluation judge. Apply the rubric below "
    "exactly as written. Do not reward style over substance.\n\n" + RUBRIC_TEXT
)

USER_TEMPLATE = """\
Score the candidate answer against the rubric.

## Context
{context}

## Question
{question}

## Candidate answer
{answer}

Respond with ONLY a JSON object on a single line, no other text:
{{"groundedness": <1-5>, "relevance": <1-5>, "coherence": <1-5>, \
"overall_pass": <true|false>, "rationale": "<one or two sentences>"}}"""

RETRY_SUFFIX = "\n\nYour previous output could not be parsed. Output ONLY the JSON object."

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def render_request(example: Example, retry: bool = False) -> JudgeRequest:
    user = USER_TEMPLATE.format(
        context=example.context,
        question=example.question,
        answer=example.candidate_answer,
    )
    if retry:
        user += RETRY_SUFFIX
    return JudgeRequest(
        system=SYSTEM_PROMPT,
        user=user,
        context=example.context,
        question=example.question,
        answer=example.candidate_answer,
    )


def parse_judge_output(text: str) -> JudgeScore:
    match = _JSON_RE.search(text)
    if match is None:
        raise JudgeParseError(f"no JSON object found in judge output: {text!r}")
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"invalid JSON in judge output: {exc}: {text!r}") from exc

    for dim in DIMENSIONS:
        value = raw.get(dim)
        if not isinstance(value, int) or isinstance(value, bool):
            raise JudgeParseError(f"judge field {dim!r} must be an int, got {value!r}")
        if not (MIN_SCORE <= value <= MAX_SCORE):
            raise JudgeParseError(f"judge field {dim!r} out of range: {value}")
    if not isinstance(raw.get("overall_pass"), bool):
        raise JudgeParseError(
            f"judge field 'overall_pass' must be a bool, got {raw.get('overall_pass')!r}"
        )

    return JudgeScore(
        groundedness=raw["groundedness"],
        relevance=raw["relevance"],
        coherence=raw["coherence"],
        overall_pass=raw["overall_pass"],
        rationale=str(raw.get("rationale", "")),
    )


class Judge:
    def __init__(self, backend: JudgeBackend) -> None:
        self.backend = backend

    def score_example(self, example: Example) -> JudgeScore:
        raw = self.backend.complete(render_request(example))
        try:
            return parse_judge_output(raw)
        except JudgeParseError:
            raw_retry = self.backend.complete(render_request(example, retry=True))
            return parse_judge_output(raw_retry)

    def score_dataset(
        self, examples: list[Example], progress: bool = False
    ) -> list[JudgeScore]:
        scores: list[JudgeScore] = []
        for i, example in enumerate(examples, start=1):
            scores.append(self.score_example(example))
            if progress:
                print(f"  scored {i}/{len(examples)}: {example.id}")
        return scores
