import json

import pytest

from evalharness.dataset import Example, HumanLabel
from evalharness.judge import (
    Judge,
    JudgeParseError,
    parse_judge_output,
    render_request,
)

EXAMPLE = Example(
    id="t-1",
    domain="test",
    context="The cafe opens at 8am and closes at 4pm.",
    question="When does the cafe open?",
    candidate_answer="It opens at 8am.",
    human=HumanLabel(5, 5, 5, True),
    human_rationale="",
    tags=(),
)

VALID = (
    '{"groundedness": 5, "relevance": 5, "coherence": 4, '
    '"overall_pass": true, "rationale": "ok"}'
)


def test_parse_clean_json():
    score = parse_judge_output(VALID)
    assert score.groundedness == 5
    assert score.coherence == 4
    assert score.overall_pass is True


def test_parse_fenced_json():
    assert parse_judge_output(f"```json\n{VALID}\n```").relevance == 5


def test_parse_prose_wrapped_json():
    assert parse_judge_output(f"Here is my verdict:\n{VALID}\nThanks!").groundedness == 5


def test_parse_rejects_out_of_range():
    bad = VALID.replace('"groundedness": 5', '"groundedness": 7')
    with pytest.raises(JudgeParseError, match="out of range"):
        parse_judge_output(bad)


def test_parse_rejects_bool_for_dimension():
    bad = VALID.replace('"groundedness": 5', '"groundedness": true')
    with pytest.raises(JudgeParseError):
        parse_judge_output(bad)


def test_parse_rejects_missing_pass():
    bad = VALID.replace('"overall_pass": true, ', "")
    with pytest.raises(JudgeParseError, match="overall_pass"):
        parse_judge_output(bad)


def test_parse_rejects_no_json():
    with pytest.raises(JudgeParseError, match="no JSON"):
        parse_judge_output("I think it is fine.")


class FlakyBackend:
    """Garbage on first call, valid JSON on the retry."""

    name = "flaky"

    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            return "sorry, as an AI I cannot produce JSON"
        return VALID


def test_judge_retries_once_on_parse_failure():
    backend = FlakyBackend()
    score = Judge(backend).score_example(EXAMPLE)
    assert backend.calls == 2
    assert score.overall_pass is True


def test_render_request_includes_fields_and_rubric():
    request = render_request(EXAMPLE)
    assert EXAMPLE.context in request.user
    assert EXAMPLE.question in request.user
    assert EXAMPLE.candidate_answer in request.user
    assert "groundedness" in request.system  # rubric embedded


def test_mock_backend_deterministic_and_directionally_sane():
    from evalharness.backends import MockBackend

    backend = MockBackend()
    request = render_request(EXAMPLE)
    assert backend.complete(request) == backend.complete(request)

    grounded = json.loads(backend.complete(request))
    fabricated_example = Example(
        id="t-2",
        domain="test",
        context=EXAMPLE.context,
        question=EXAMPLE.question,
        candidate_answer="The owner imports rare beans from a volcanic farm in Iceland.",
        human=None,
        human_rationale="",
        tags=(),
    )
    fabricated = json.loads(backend.complete(render_request(fabricated_example)))
    assert grounded["groundedness"] > fabricated["groundedness"]


def test_mock_backend_is_blind_to_negation():
    """Documents the designed failure mode: a negation flip built from context
    vocabulary gets high lexical groundedness. The calibration report exists
    to quantify exactly this."""
    from evalharness.backends import MockBackend

    flipped = Example(
        id="t-3",
        domain="test",
        context="The cafe is not open on Sundays, and it never serves dinner.",
        question="Is the cafe open on Sundays?",
        candidate_answer="Yes, the cafe is open on Sundays and serves dinner.",
        human=None,
        human_rationale="",
        tags=(),
    )
    verdict = json.loads(MockBackend().complete(render_request(flipped)))
    assert verdict["groundedness"] >= 4
