"""The scoring rubric — single source of truth for humans and the judge.

Judge-human calibration is only meaningful when both score against the same
instructions, so this module is canonical: the judge prompt embeds RUBRIC_TEXT
verbatim, and data/ANNOTATION_GUIDELINES.md quotes the same text. If you edit
the rubric, you must re-annotate (or re-verify) the human labels and re-run
calibration — the CI calibration gate exists to catch silent drift here.
"""

# Ordinal dimensions scored 1-5 by both humans and the judge.
DIMENSIONS: tuple[str, ...] = ("groundedness", "relevance", "coherence")

MIN_SCORE = 1
MAX_SCORE = 5

RUBRIC_TEXT = """\
You are scoring a candidate answer to a question about a provided context
passage. Score ONLY against the context — outside knowledge must not rescue
an answer that the context does not support.

Score each dimension as an integer from 1 to 5.

groundedness — Is every claim in the answer supported by the context?
  5: Every claim is directly supported by the context; or the answer
     correctly states that the context does not contain the information.
  4: All material claims are supported; at most trivial unsupported phrasing.
  3: Mostly supported, but at least one material unsupported claim.
  2: Several unsupported claims, or a claim that contradicts the context.
  1: Largely fabricated, or contradicts the context on the main point.

relevance — Does the answer address the question that was asked?
  5: Directly and completely answers the question.
  4: Answers the question with minor omissions or digressions.
  3: Partially answers; a significant part of the question is unaddressed.
  2: Mostly off-target; touches the topic but not the question.
  1: Does not address the question.

coherence — Is the answer well-formed, readable prose?
  5: Clear, fluent, well-organized.
  4: Minor awkwardness; fully understandable.
  3: Noticeable issues (fragments, repetition) but the meaning survives.
  2: Hard to follow.
  1: Garbled or self-contradictory.

overall_pass (boolean) — Would you ship this answer to an end user?
  true requires groundedness >= 4 AND relevance >= 3 AND coherence >= 3.
  An answer that correctly declines because the context lacks the requested
  information is shippable: groundedness 5, relevance scored on how clearly
  the limitation is communicated.
"""
