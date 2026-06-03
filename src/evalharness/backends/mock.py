"""Deterministic lexical-overlap judge.

This is intentionally a *weak but honest* baseline judge: it scores
groundedness by content-word overlap between answer and context, and
relevance by overlap between question and answer. It exists for three
reasons:

1. CI and tests run the full pipeline (prompt -> backend -> JSON -> parse ->
   metrics) with zero secrets and full determinism.
2. It is the control arm for calibration: the README's central claim is that
   lexical overlap cannot measure groundedness (negation, fact recombination,
   and correct refusals all break it), and the calibration report quantifies
   exactly how. An LLM judge has to beat these numbers to justify its cost.
3. It returns raw JSON text like a real model would, so the same parsing path
   is exercised end to end.
"""

from __future__ import annotations

import json
import re

from .base import JudgeRequest

# Small stopword list. Note that negators ("not", "no", "never") are treated
# as stopwords, exactly like classic lexical-overlap metrics — this is one of
# the designed failure modes the calibration report surfaces.
_STOPWORDS = frozenset(
    """a an the is are was were be been being to of in on at for with and or
    not no never it its this that these those as by from has have had does do
    did don't doesn't didn't isn't aren't he she they them their his her you
    your i we us our but if than then so about also can could will would may
    might what which who whom when where why how does""".split()
)

_WORD_RE = re.compile(r"[a-z0-9']+")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _overlap(subset_of: set[str], words: set[str]) -> float:
    if not words:
        return 0.0
    return len(words & subset_of) / len(words)


def _bucket(value: float, thresholds: tuple[float, float, float, float]) -> int:
    """Map a 0-1 score onto 1-5 using descending thresholds for 5,4,3,2."""
    t5, t4, t3, t2 = thresholds
    if value >= t5:
        return 5
    if value >= t4:
        return 4
    if value >= t3:
        return 3
    if value >= t2:
        return 2
    return 1


class MockBackend:
    name = "mock"

    def complete(self, request: JudgeRequest) -> str:
        answer_words = _content_words(request.answer)
        context_words = _content_words(request.context)
        question_words = _content_words(request.question)

        # Groundedness proxy: how much of the answer's vocabulary appears in
        # the context. Blind to negation, blind to recombined facts, and it
        # punishes correct refusals (their vocabulary is meta-language).
        context_overlap = _overlap(context_words, answer_words)
        groundedness = _bucket(context_overlap, (0.80, 0.65, 0.45, 0.25))

        # Relevance proxy: how much of the question's vocabulary the answer
        # engages with.
        question_overlap = _overlap(answer_words, question_words)
        relevance = _bucket(question_overlap, (0.65, 0.45, 0.30, 0.15))

        coherence = self._coherence(request.answer)
        overall_pass = groundedness >= 4 and relevance >= 3 and coherence >= 3

        payload = {
            "groundedness": groundedness,
            "relevance": relevance,
            "coherence": coherence,
            "overall_pass": overall_pass,
            "rationale": (
                f"lexical heuristic: context-overlap={context_overlap:.2f}, "
                f"question-overlap={question_overlap:.2f}"
            ),
        }
        return json.dumps(payload)

    @staticmethod
    def _coherence(answer: str) -> int:
        text = answer.strip()
        if not text:
            return 1
        letters = sum(c.isalpha() for c in text)
        if letters / max(len(text), 1) < 0.5:
            return 1
        ends_cleanly = text[-1] in ".!?"
        words = text.split()
        if len(words) < 4:
            return 2
        if not ends_cleanly:
            return 3
        # Heavy immediate-repetition reads as degenerate decoding.
        pairs = zip(words, words[1:], strict=False)
        repeats = sum(1 for a, b in pairs if a.lower() == b.lower())
        if repeats >= 2:
            return 2
        return 5 if len(words) >= 8 else 4
