import math

from evalharness.calibration import CalibrationResult, OrdinalAgreement, PassAgreement
from evalharness.judge import JudgeScore
from evalharness.regression import (
    GateConfig,
    aggregate_scores,
    check_calibration_gate,
    check_regression_gate,
    load_baseline,
    write_baseline,
)


def _scores(passes: list[bool], groundedness: int = 4) -> list[JudgeScore]:
    return [JudgeScore(groundedness, 4, 4, p, "") for p in passes]


def test_aggregate_scores():
    agg = aggregate_scores(_scores([True, True, False, False]))
    assert agg["pass_rate"] == 0.5
    assert agg["mean_groundedness"] == 4.0


def test_regression_gate_passes_when_equal():
    agg = aggregate_scores(_scores([True, False]))
    assert check_regression_gate(agg, agg).passed


def test_regression_gate_fails_on_pass_rate_drop():
    baseline = aggregate_scores(_scores([True] * 10))
    current = aggregate_scores(_scores([True] * 7 + [False] * 3))
    report = check_regression_gate(current, baseline)
    assert not report.passed
    failed = [c for c in report.checks if not c.passed]
    assert any("pass rate" in c.name for c in failed)


def test_regression_gate_tolerates_small_drop():
    baseline = {"pass_rate": 0.80, "mean_groundedness": 4.0}
    current = {"pass_rate": 0.79, "mean_groundedness": 3.95}
    config = GateConfig(max_pass_rate_drop=0.02, max_groundedness_drop=0.15)
    assert check_regression_gate(current, baseline, config).passed


def _calibration_result(kappa: float, qwk: float) -> CalibrationResult:
    result = CalibrationResult(backend_name="x", n_examples=10)
    result.ordinal["groundedness"] = OrdinalAgreement(
        "groundedness", qwk, 0.5, 0.5, 1.0, 0.5, 0.9
    )
    result.overall_pass = PassAgreement(0.8, kappa, 0.8, 0.8, 0.8, [[4, 1], [1, 4]])
    return result


def test_calibration_gate_thresholds():
    config = GateConfig(min_pass_kappa=0.30, min_groundedness_qwk=0.30)
    assert check_calibration_gate(_calibration_result(0.5, 0.5), config).passed
    assert not check_calibration_gate(_calibration_result(0.1, 0.5), config).passed
    assert not check_calibration_gate(_calibration_result(0.5, 0.1), config).passed


def test_calibration_gate_fails_on_nan():
    nan = float("nan")
    assert not check_calibration_gate(_calibration_result(nan, 0.5)).passed
    assert math.isnan(_calibration_result(nan, 0.5).overall_pass.kappa)


def test_baseline_round_trip(tmp_path):
    path = tmp_path / "baseline.json"
    agg = {"pass_rate": 0.6, "mean_groundedness": 3.9}
    write_baseline(path, "lexical", "data/grounded_qa.jsonl", agg)
    loaded = load_baseline(path)
    assert loaded["backend"] == "lexical"
    assert loaded["aggregate"] == agg
