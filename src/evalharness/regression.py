"""Two-tier quality gating.

Tier 1 — calibration gate: is the judge still trustworthy? Judge-vs-human
agreement must stay above thresholds. A judge that has drifted from human
judgment must not be allowed to gate releases, so this tier runs first.

Tier 2 — regression gate: did the system under test get worse? Aggregate
judge scores over a candidate answer set are compared against a frozen
baseline with explicit tolerances.

NaN statistics fail gates by design: "we cannot demonstrate calibration"
should never read as a pass.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .calibration import CalibrationResult
from .judge import JudgeScore
from .metrics import mean
from .rubric import DIMENSIONS


@dataclass(frozen=True)
class GateConfig:
    # Tier 1: minimum acceptable judge-human agreement. Initialized just below
    # the measured agreement of the lexical baseline (kappa 0.400,
    # groundedness QWK 0.256 on the seed benchmark) so the floor is honest:
    # any judge that cannot beat lexical overlap fails. After switching to a
    # stronger judge backend, ratchet these up to just below its measured
    # agreement so regressions in judge quality are caught.
    min_pass_kappa: float = 0.30
    min_groundedness_qwk: float = 0.20
    # Tier 2: maximum acceptable drop vs. the frozen baseline.
    max_pass_rate_drop: float = 0.02
    max_groundedness_drop: float = 0.15  # mean groundedness, 1-5 scale


@dataclass(frozen=True)
class GateCheck:
    name: str
    value: float
    threshold: float
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class GateReport:
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [c.to_dict() for c in self.checks]}


def aggregate_scores(scores: list[JudgeScore]) -> dict[str, float]:
    """Aggregate judge scores into the system-quality numbers the regression
    gate compares. pass_rate is the headline; per-dimension means localize a
    drop when one happens."""
    out = {"pass_rate": mean([1.0 if s.overall_pass else 0.0 for s in scores])}
    for dim in DIMENSIONS:
        out[f"mean_{dim}"] = mean([float(s.dimension(dim)) for s in scores])
    return out


def check_calibration_gate(
    result: CalibrationResult, config: GateConfig | None = None
) -> GateReport:
    config = config or GateConfig()
    report = GateReport()

    kappa = result.overall_pass.kappa if result.overall_pass else float("nan")
    report.checks.append(
        _at_least(
            "judge-human kappa (overall_pass)",
            kappa,
            config.min_pass_kappa,
            "Cohen's kappa between judge and human ship/no-ship verdicts",
        )
    )
    qwk = result.ordinal["groundedness"].qwk if "groundedness" in result.ordinal else float("nan")
    report.checks.append(
        _at_least(
            "judge-human QWK (groundedness)",
            qwk,
            config.min_groundedness_qwk,
            "quadratic-weighted kappa on the groundedness dimension",
        )
    )
    return report


def check_regression_gate(
    current: dict[str, float],
    baseline: dict[str, float],
    config: GateConfig | None = None,
) -> GateReport:
    config = config or GateConfig()
    report = GateReport()

    report.checks.append(
        _max_drop(
            "pass rate",
            current.get("pass_rate", float("nan")),
            baseline.get("pass_rate", float("nan")),
            config.max_pass_rate_drop,
        )
    )
    report.checks.append(
        _max_drop(
            "mean groundedness",
            current.get("mean_groundedness", float("nan")),
            baseline.get("mean_groundedness", float("nan")),
            config.max_groundedness_drop,
        )
    )
    return report


def write_baseline(path: str | Path, backend_name: str, dataset_name: str,
                   aggregate: dict[str, float]) -> None:
    payload = {
        "backend": backend_name,
        "dataset": dataset_name,
        "aggregate": aggregate,
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _at_least(name: str, value: float, threshold: float, detail: str) -> GateCheck:
    passed = (not math.isnan(value)) and value >= threshold
    return GateCheck(name=name, value=value, threshold=threshold, passed=passed, detail=detail)


def _max_drop(name: str, current: float, baseline: float, tolerance: float) -> GateCheck:
    drop = baseline - current
    passed = (not math.isnan(drop)) and drop <= tolerance
    return GateCheck(
        name=name,
        value=current,
        threshold=baseline - tolerance,
        passed=passed,
        detail=f"baseline {baseline:.3f}, current {current:.3f}, "
        f"allowed drop {tolerance:.3f}",
    )
