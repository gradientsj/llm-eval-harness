# llm-eval-harness

[![eval-gate](https://github.com/gradientsj/llm-eval-harness/actions/workflows/eval-gate.yml/badge.svg)](https://github.com/gradientsj/llm-eval-harness/actions/workflows/eval-gate.yml)

**An LLM-as-judge evaluation harness with human calibration and two-tier CI
quality gating, demonstrated on grounded QA.**

Evaluating open-ended LLM output is an unsolved problem with immediate product
consequences: you cannot ship what you cannot measure, and you cannot trust an
auto-judge you have never calibrated against humans. This repo is a small,
fully tested, end-to-end implementation of the loop that makes auto-judges
trustworthy:

```
documented rubric ──> LLM judge ──> agreement vs. human labels ──> calibration gate
                          │                                             │
                          └──> aggregate quality vs. frozen baseline ──> regression gate
                                                                  (both run in CI)
```

Three design commitments, all unusual enough to be worth stating up front:

1. **The judge is gated on its agreement with humans, not assumed correct.**
   Judge↔human agreement (Cohen's κ, quadratic-weighted κ, Spearman, MAE) is
   computed on a labeled benchmark and enforced in CI: if a prompt or model
   change degrades agreement below threshold, the build fails. A judge you
   can't trust must not gate releases.
2. **A deterministic lexical judge is the control arm.** The `lexical`
   backend scores groundedness by content-word overlap — the strongest thing
   you can do without a model. It runs the entire pipeline in CI with zero
   secrets, and it sets the floor an LLM judge has to beat to justify its
   cost.
3. **The benchmark is adversarially composed.** A third of the examples are
   constructed traps — negation flips, fact recombinations, correct refusals,
   incoherent-but-on-topic strings — where shallow scoring is known to
   disagree with humans. The failure-analysis report quantifies exactly where
   and why.

## Results: the lexical baseline, measured

Judge↔human agreement of the `lexical` (word-overlap) baseline judge on the
30-example seed benchmark (`uv run evalharness calibrate --backend lexical`):

| Dimension | QWK | Spearman | MAE | Exact | Within ±1 |
|---|---|---|---|---|---|
| groundedness | **0.256** | 0.305 | 1.30 | 0.47 | 0.67 |
| relevance | 0.505 | 0.380 | 0.83 | 0.43 | 0.80 |
| coherence | 0.725 | 0.667 | 0.23 | 0.83 | 0.93 |

Ship/no-ship (overall_pass): **accuracy 0.70, Cohen's κ 0.400, F1 0.73.**

The gradient is the finding: **coherence is nearly solvable with surface
features, relevance is partially solvable, groundedness is not.** The
[failure analysis](reports/calibration_report_lexical.md) shows the three
mechanisms, each a deliberate slice of the benchmark:

- **Negation flips** (`gqa-018`: human 1, judge 5) — *"Yes, his gritty trough
  mix is suitable for ericaceous species"* scores 0.86 lexical overlap against
  a context that says it is **not** suitable. Overlap is blind to polarity.
- **Fact recombination** (`gqa-021`: human 1, judge 5) — real facts from the
  context attached to the wrong entity. Every word is "grounded"; the claim is
  false.
- **Correct refusals** (`gqa-022`: human 5, judge 1) — *"The profile does not
  say how many subscribers the channel has"* is the **right** answer with
  near-zero context overlap. Lexical scoring punishes exactly the behavior
  groundedness training tries to encourage.

These numbers are the floor for the real judge: an LLM backend
(`--backend anthropic`) has to beat κ 0.40 / QWK 0.26 to be worth its latency
and cost — and the harness is how you check it does.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). No API key needed for the default
lexical backend.

```bash
uv sync --extra dev
uv run pytest                                          # 45 tests
uv run evalharness calibrate --backend lexical --check-gates
uv run evalharness baseline  --backend lexical            # freeze the baseline
uv run evalharness gate      --backend lexical            # regression-gate it
```

To run the real judge, set `ANTHROPIC_API_KEY` (copy `.env.example` to `.env`),
then:

```bash
uv run evalharness calibrate --backend anthropic --check-gates
```

This makes ~30 calls (plus at most one parse-retry each) at temperature 0
against a pinned model snapshot (`claude-sonnet-4-6` by default, overridable
via `EVALHARNESS_JUDGE_MODEL`).

## The two-tier CI gate

[`eval-gate.yml`](.github/workflows/eval-gate.yml) enforces both gates on
every push and PR, after lint and unit tests:

**Tier 1 — calibration gate.** `calibrate --check-gates` fails the build if
judge↔human agreement drops below thresholds (κ ≥ 0.30 on ship/no-ship,
QWK ≥ 0.20 on groundedness — initialized just below the lexical baseline's
measured agreement, to be ratcheted up once the LLM judge's numbers are in).
This catches rubric edits, judge-prompt regressions, and judge-model drift.

**Tier 2 — regression gate.** `gate` re-scores a candidate answer set and
compares aggregates against the frozen [baseline](data/baseline_scores.json)
with explicit tolerances (pass rate −0.02, mean groundedness −0.15). The gate
only runs if Tier 1 passed: a regression verdict from an uncalibrated judge is
noise.

The repo includes a simulated bad release —
[`grounded_qa_regressed.jsonl`](data/grounded_qa_regressed.jsonl), the same 30
questions with 8 answers replaced by fluent fabrications — to demonstrate the
gate catching a real quality drop:

```
$ uv run evalharness gate --backend lexical --candidates data/grounded_qa_regressed.jsonl
  [FAIL] pass rate: baseline 0.600, current 0.333, allowed drop 0.020
  [FAIL] mean groundedness: baseline 3.733, current 2.767, allowed drop 0.150
Regression gate FAILED - candidate quality dropped below baseline tolerance.   (exit 1)
```

Note the candidate file carries no human labels — gating fresh model output
must not require annotation, only the (separately calibrated) judge.

## Benchmark construction

30 grounded-QA examples over short creator-profile contexts, labeled against
the same rubric the judge prompt embeds (single source of truth:
[`rubric.py`](src/evalharness/rubric.py); annotator instructions:
[`ANNOTATION_GUIDELINES.md`](data/ANNOTATION_GUIDELINES.md)). Composition is
deliberately adversarial rather than uniformly easy:

| Slice | n | Designed to test |
|---|---|---|
| faithful / paraphrase | 12 | the easy case; judges must not over-fail |
| partial support | 5 | one invented detail attached to true claims |
| negation flip / fact recombination | 4 | groundedness beyond lexical overlap |
| correct refusal | 3 | refusing must be rewarded, not punished |
| off-topic (fluent) | 3 | relevance vs. fluency |
| incoherent word salad | 2 | coherence vs. token overlap |
| fluent hallucination | 1 | confident fabrication |

Labels follow the rubric mechanically (`overall_pass` ⇔ groundedness ≥ 4 ∧
relevance ≥ 3 ∧ coherence ≥ 3), which is itself enforced by a unit test — the
dataset cannot drift from its own labeling rule without CI noticing.

**Provenance:** this is a seed benchmark labeled by a single
annotator (me) to demonstrate the calibration machinery. The production
version needs ≥3 annotators per example with inter-annotator agreement
reported, examples sampled from real traffic, and a held-out split so the
judge prompt is never tuned on the examples that certify it. See
[ANNOTATION_GUIDELINES.md](data/ANNOTATION_GUIDELINES.md) for the full list of
limitations.

## Design decisions

- **NaN fails gates.** Undefined statistics (zero variance, degenerate
  marginals) return `nan`, never a silently-wrong 0.0 — and `nan` fails the
  gate, because "cannot demonstrate calibration" must not read as a pass.
- **Metrics implemented from scratch, tested against hand-computed values.**
  ~150 lines of pure Python instead of a scipy dependency: the agreement math
  is the product here, so it should be auditable.
- **Reports carry no timestamps.** Identical inputs produce byte-identical
  reports, so a report diff in a PR shows real changes only.
- **Strict-content, lenient-packaging parsing.** The judge must produce all
  rubric fields in range (booleans are rejected where ints are required), but
  code fences and surrounding prose are tolerated, with one terse retry —
  matching how real models actually misbehave.
- **Backend mismatch is a hard error.** Scores from different judges are not
  comparable; the regression gate refuses to compare a `lexical` run against
  an `anthropic` baseline rather than emitting a meaningless verdict.

## What I'd do next

In rough priority order:

1. **Run the Anthropic judge and ratchet the gates** — measure κ/QWK for
   `claude-sonnet-4-6`, publish the comparison table against the lexical
   floor, and raise the calibration thresholds to just below its numbers.
2. **Uncertainty on small-n agreement** — bootstrap CIs on κ/QWK; with n=30 a
   single flipped example moves κ by ~0.07, and the gates should know that.
3. **Judge robustness checks** — position/length bias probes, self-consistency
   across temperature-0 re-runs over time, and an ensemble-of-rubrics variant.
4. **Scale the annotation** — multi-annotator labels with Krippendorff's
   α, adjudication, and a held-out certification split.
5. **Close the loop on a real system** — point the harness at an actual
   RAG/agent pipeline so the regression gate scores live candidate outputs per
   release rather than a fixture file.

## Repository layout

```
src/evalharness/
  rubric.py               # the rubric (single source of truth, embedded in judge prompt)
  dataset.py              # JSONL loading + validation; labels optional for gating
  judge.py                # prompt rendering, strict parsing, one-retry scoring
  backends/
    lexical.py            # deterministic word-overlap judge (CI control arm)
    anthropic_backend.py  # temperature-0 Anthropic judge, pinned model
  metrics.py              # κ, QWK, Spearman/Pearson, MAE, confusion — pure Python
  calibration.py          # agreement computation + ranked disagreements
  regression.py           # two-tier gates, frozen baselines
  report.py               # deterministic markdown reports
  cli.py                  # calibrate / baseline / gate
data/
  grounded_qa.jsonl            # 30 labeled examples (composition above)
  grounded_qa_regressed.jsonl  # simulated bad release for the gate demo
  baseline_scores.json         # frozen aggregates (regression-gate reference)
  ANNOTATION_GUIDELINES.md     # human labeling instructions (same rubric)
reports/
  calibration_report_lexical.md  # committed example output, reproducible in CI
tests/                         # 45 tests: metrics math, parsing, E2E, gates
```

## License

MIT
