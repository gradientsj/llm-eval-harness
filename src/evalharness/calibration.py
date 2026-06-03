"""Judge-vs-human calibration: run the judge over a labeled benchmark and
quantify agreement per rubric dimension, plus a ranked disagreement list for
failure analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from .dataset import Example, require_labels
from .judge import Judge, JudgeScore
from .metrics import (
    accuracy,
    cohen_kappa,
    confusion_matrix,
    exact_and_within_one,
    mae,
    pearson,
    precision_recall_f1,
    quadratic_weighted_kappa,
    spearman,
)
from .rubric import DIMENSIONS, MAX_SCORE, MIN_SCORE


@dataclass(frozen=True)
class OrdinalAgreement:
    dimension: str
    qwk: float
    spearman: float
    pearson: float
    mae: float
    exact: float
    within_one: float

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "qwk": self.qwk,
            "spearman": self.spearman,
            "pearson": self.pearson,
            "mae": self.mae,
            "exact": self.exact,
            "within_one": self.within_one,
        }


@dataclass(frozen=True)
class PassAgreement:
    accuracy: float
    kappa: float
    precision: float
    recall: float
    f1: float
    # Rows/cols: [human fail, human pass] x [judge fail, judge pass]
    confusion: list[list[int]]

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "kappa": self.kappa,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion": self.confusion,
        }


@dataclass(frozen=True)
class Disagreement:
    example: Example
    human_score: int
    judge_score: JudgeScore
    dimension: str
    delta: int  # judge - human on `dimension`
    pass_flip: bool


@dataclass
class CalibrationResult:
    backend_name: str
    n_examples: int
    ordinal: dict[str, OrdinalAgreement] = field(default_factory=dict)
    overall_pass: PassAgreement | None = None
    disagreements: list[Disagreement] = field(default_factory=list)
    judge_scores: list[JudgeScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend_name,
            "n_examples": self.n_examples,
            "ordinal": {k: v.to_dict() for k, v in self.ordinal.items()},
            "overall_pass": self.overall_pass.to_dict() if self.overall_pass else None,
        }


def run_calibration(
    examples: list[Example], judge: Judge, progress: bool = False
) -> CalibrationResult:
    require_labels(examples)
    scores = judge.score_dataset(examples, progress=progress)
    return compute_calibration(examples, scores, judge.backend.name)


def compute_calibration(
    examples: list[Example], scores: list[JudgeScore], backend_name: str
) -> CalibrationResult:
    if len(examples) != len(scores):
        raise ValueError("examples and scores must be paired")
    require_labels(examples)

    result = CalibrationResult(backend_name=backend_name, n_examples=len(examples))
    result.judge_scores = list(scores)

    for dim in DIMENSIONS:
        human = [e.human.dimension(dim) for e in examples]
        judge_vals = [s.dimension(dim) for s in scores]
        exact, within = exact_and_within_one(human, judge_vals)
        result.ordinal[dim] = OrdinalAgreement(
            dimension=dim,
            qwk=quadratic_weighted_kappa(human, judge_vals, MIN_SCORE, MAX_SCORE),
            spearman=spearman(human, judge_vals),
            pearson=pearson(human, judge_vals),
            mae=mae(human, judge_vals),
            exact=exact,
            within_one=within,
        )

    human_pass = [e.human.overall_pass for e in examples]
    judge_pass = [s.overall_pass for s in scores]
    precision, recall, f1 = precision_recall_f1(human_pass, judge_pass)
    result.overall_pass = PassAgreement(
        accuracy=accuracy(human_pass, judge_pass),
        kappa=cohen_kappa(human_pass, judge_pass),
        precision=precision,
        recall=recall,
        f1=f1,
        confusion=confusion_matrix(human_pass, judge_pass, [False, True]),
    )

    result.disagreements = _rank_disagreements(examples, scores)
    return result


def _rank_disagreements(
    examples: list[Example], scores: list[JudgeScore]
) -> list[Disagreement]:
    """Worst judge-human disagreements first. Severity = largest ordinal delta
    across dimensions, with ship/no-ship flips breaking ties upward."""
    items: list[Disagreement] = []
    for example, score in zip(examples, scores, strict=True):
        worst_dim = max(
            DIMENSIONS,
            key=lambda d: abs(score.dimension(d) - example.human.dimension(d)),
        )
        delta = score.dimension(worst_dim) - example.human.dimension(worst_dim)
        pass_flip = score.overall_pass != example.human.overall_pass
        if delta != 0 or pass_flip:
            items.append(
                Disagreement(
                    example=example,
                    human_score=example.human.dimension(worst_dim),
                    judge_score=score,
                    dimension=worst_dim,
                    delta=delta,
                    pass_flip=pass_flip,
                )
            )
    items.sort(key=lambda d: (abs(d.delta), d.pass_flip), reverse=True)
    return items
