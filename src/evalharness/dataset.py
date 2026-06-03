"""Benchmark dataset loading and validation.

Records live in JSONL. Human labels are optional per-record: the calibration
flow requires them (judge vs. human agreement), the regression-gate flow does
not (it only aggregates judge scores over a candidate answer set).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .rubric import DIMENSIONS, MAX_SCORE, MIN_SCORE


class DatasetError(ValueError):
    """Raised when a dataset file fails validation."""


@dataclass(frozen=True)
class HumanLabel:
    groundedness: int
    relevance: int
    coherence: int
    overall_pass: bool

    def dimension(self, name: str) -> int:
        return int(getattr(self, name))


@dataclass(frozen=True)
class Example:
    id: str
    domain: str
    context: str
    question: str
    candidate_answer: str
    human: HumanLabel | None
    human_rationale: str
    tags: tuple[str, ...]


def _parse_label(raw: dict, example_id: str) -> HumanLabel:
    for dim in DIMENSIONS:
        value = raw.get(dim)
        if not isinstance(value, int) or not (MIN_SCORE <= value <= MAX_SCORE):
            raise DatasetError(
                f"{example_id}: human.{dim} must be an int in "
                f"[{MIN_SCORE}, {MAX_SCORE}], got {value!r}"
            )
    if not isinstance(raw.get("overall_pass"), bool):
        raise DatasetError(f"{example_id}: human.overall_pass must be a bool")
    return HumanLabel(
        groundedness=raw["groundedness"],
        relevance=raw["relevance"],
        coherence=raw["coherence"],
        overall_pass=raw["overall_pass"],
    )


def load_dataset(path: str | Path) -> list[Example]:
    path = Path(path)
    examples: list[Example] = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_no}: invalid JSON ({exc})") from exc

            example_id = raw.get("id")
            if not example_id or not isinstance(example_id, str):
                raise DatasetError(f"{path}:{line_no}: missing string 'id'")
            if example_id in seen_ids:
                raise DatasetError(f"{path}:{line_no}: duplicate id {example_id!r}")
            seen_ids.add(example_id)

            for field in ("context", "question", "candidate_answer"):
                if not isinstance(raw.get(field), str) or not raw[field].strip():
                    raise DatasetError(f"{example_id}: missing or empty '{field}'")

            human = _parse_label(raw["human"], example_id) if raw.get("human") else None

            examples.append(
                Example(
                    id=example_id,
                    domain=str(raw.get("domain", "")),
                    context=raw["context"],
                    question=raw["question"],
                    candidate_answer=raw["candidate_answer"],
                    human=human,
                    human_rationale=str(raw.get("human_rationale", "")),
                    tags=tuple(raw.get("tags", [])),
                )
            )

    if not examples:
        raise DatasetError(f"{path}: dataset is empty")
    return examples


def require_labels(examples: list[Example]) -> None:
    """Raise unless every example carries a human label (needed for calibration)."""
    missing = [e.id for e in examples if e.human is None]
    if missing:
        raise DatasetError(
            f"calibration requires human labels on every example; missing on: {missing}"
        )
