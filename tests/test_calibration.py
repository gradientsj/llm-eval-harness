import json

import pytest

from evalharness.backends import LexicalBackend
from evalharness.calibration import compute_calibration, run_calibration
from evalharness.dataset import load_dataset
from evalharness.judge import Judge, JudgeScore


class EchoHumanBackend:
    """A 'perfect' judge that returns each example's human label, by matching
    on the answer text embedded in the request. Used to verify the agreement
    math tops out at 1.0 when judge and human are identical."""

    name = "echo"

    def __init__(self, examples):
        self._by_answer = {e.candidate_answer: e.human for e in examples}

    def complete(self, request):
        label = self._by_answer[request.answer]
        return json.dumps(
            {
                "groundedness": label.groundedness,
                "relevance": label.relevance,
                "coherence": label.coherence,
                "overall_pass": label.overall_pass,
                "rationale": "echo of human label",
            }
        )


@pytest.fixture(scope="module")
def examples():
    return load_dataset("data/grounded_qa.jsonl")


def test_perfect_judge_scores_perfect_agreement(examples):
    result = run_calibration(examples, Judge(EchoHumanBackend(examples)))
    assert result.overall_pass.accuracy == pytest.approx(1.0)
    assert result.overall_pass.kappa == pytest.approx(1.0)
    for dim in ("groundedness", "relevance", "coherence"):
        assert result.ordinal[dim].qwk == pytest.approx(1.0)
        assert result.ordinal[dim].mae == pytest.approx(0.0)
    assert result.disagreements == []


def test_mock_judge_end_to_end(examples):
    result = run_calibration(examples, Judge(LexicalBackend()))
    assert result.n_examples == 30
    assert set(result.ordinal) == {"groundedness", "relevance", "coherence"}
    # The lexical baseline is designed to be imperfect: traps must produce
    # disagreements.
    assert result.disagreements
    # Confusion matrix cells must sum to n.
    assert sum(sum(row) for row in result.overall_pass.confusion) == 30


def test_disagreements_ranked_worst_first(examples):
    result = run_calibration(examples, Judge(LexicalBackend()))
    deltas = [abs(d.delta) for d in result.disagreements]
    assert deltas == sorted(deltas, reverse=True)


def test_paired_length_mismatch_raises(examples):
    scores = [JudgeScore(5, 5, 5, True, "")]
    with pytest.raises(ValueError):
        compute_calibration(examples, scores, "x")
