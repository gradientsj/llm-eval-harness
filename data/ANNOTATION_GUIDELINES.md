# Annotation guidelines — grounded QA benchmark

These are the instructions a human annotator follows to label `grounded_qa.jsonl`.
They quote the **same rubric the judge prompt uses** (`src/evalharness/rubric.py`).
That is deliberate: judge-human calibration is only meaningful when both sides
score against identical instructions. If the rubric changes, existing labels must
be re-verified and calibration re-run — the CI calibration gate exists to catch
exactly this kind of silent drift.

## Procedure

For each record, read the **context**, the **question**, then the
**candidate answer**. Score only against the context: outside knowledge must not
rescue an answer the context does not support.

1. Score `groundedness`, `relevance`, and `coherence` as integers 1–5 using the
   anchors below.
2. Derive `overall_pass` mechanically: `true` iff groundedness ≥ 4 AND
   relevance ≥ 3 AND coherence ≥ 3.
3. Write a one-sentence `human_rationale` naming the deciding factor.
4. Add `tags` from the taxonomy below describing the answer's failure (or
   success) mode.

## Rubric anchors

**groundedness — is every claim in the answer supported by the context?**

| Score | Anchor |
|---|---|
| 5 | Every claim directly supported; or the answer correctly states the context lacks the information |
| 4 | All material claims supported; at most trivial unsupported phrasing |
| 3 | Mostly supported, but at least one material unsupported claim |
| 2 | Several unsupported claims, or a claim that contradicts the context |
| 1 | Largely fabricated, or contradicts the context on the main point |

**relevance — does the answer address the question that was asked?**

| Score | Anchor |
|---|---|
| 5 | Directly and completely answers the question |
| 4 | Answers with minor omissions or digressions |
| 3 | Partially answers; a significant part unaddressed |
| 2 | Mostly off-target; touches the topic but not the question |
| 1 | Does not address the question |

**coherence — is the answer well-formed, readable prose?**

| Score | Anchor |
|---|---|
| 5 | Clear, fluent, well-organized |
| 4 | Minor awkwardness; fully understandable |
| 3 | Noticeable issues but the meaning survives |
| 2 | Hard to follow |
| 1 | Garbled or self-contradictory |

Note that relevance and groundedness are independent: a confident wrong answer
that addresses the question directly is high-relevance, low-groundedness. An
answer that correctly declines because the context lacks the information is
groundedness 5 and shippable — inventing an answer is the failure mode, not
admitting the gap.

## Tag taxonomy

| Tag | Meaning |
|---|---|
| `faithful` | Fully supported answer |
| `paraphrase` | Supported, but reworded rather than quoted |
| `partial_support` | Core claim supported, with unsupported additions |
| `unsupported_addition` | Specific invented detail attached to a true claim |
| `contradiction` | States the opposite of the context |
| `negation_flip` | Contradiction made of *only* context vocabulary (lexical-overlap trap) |
| `fact_recombination` | Real facts from the context attached to the wrong entity |
| `correct_refusal` | Correctly states the context lacks the information |
| `off_topic` | Fluent text that ignores the question |
| `incoherent` | Word salad / degenerate repetition |
| `fluent_hallucination` | Well-written, entirely fabricated answer |

## Provenance and limitations

This is a 30-example **seed benchmark labeled by a single annotator** (the
repository author) for the purpose of demonstrating the calibration machinery.
The composition is deliberately adversarial: roughly a third of the examples are
constructed traps (negation flips, fact recombinations, correct refusals,
incoherent-but-on-topic strings) where shallow scoring heuristics and careless
judges are known to disagree with humans.

A production version of this benchmark would need: (1) at least three
independent annotators per example with inter-annotator agreement reported
(e.g., Krippendorff's alpha) and adjudication for disagreements; (2) examples
sampled from real system traffic rather than authored; (3) periodic refresh to
track distribution shift; and (4) a held-out split so the judge prompt is not
tuned against the same examples used to certify it.
