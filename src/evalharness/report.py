"""Markdown calibration report with ranked failure analysis.

Reports are written to be self-contained: a reader who has never seen this
repository should be able to understand what was measured, how to read each
statistic, and why it was chosen. Reports intentionally carry no timestamp: a
re-run on identical inputs must produce an identical file, so report diffs in
PRs show real changes only.
"""

from __future__ import annotations

import math

from .calibration import CalibrationResult, Disagreement
from .regression import GateReport
from .rubric import DIMENSIONS

# Shown under "This run" so the report explains its own judge.
BACKEND_DESCRIPTIONS = {
    "lexical": (
        "a deterministic word-overlap baseline that makes no model calls. It "
        "scores groundedness by counting how much of the answer's vocabulary "
        "appears in the context passage, so it runs in CI for free and it "
        "sets the agreement floor that a real LLM judge must beat to justify "
        "its cost. Its scores are intentionally imperfect: the failure "
        "analysis at the bottom shows exactly where surface word overlap "
        "diverges from human judgment."
    ),
    "anthropic": (
        "an LLM judge (Anthropic Claude, pinned model snapshot, temperature "
        "0) that scores each answer against the rubric embedded in its "
        "system prompt."
    ),
}

BACKGROUND = """\
## What this report is

This harness evaluates LLM answers to grounded QA tasks: given a context
passage and a question, was the model's answer actually supported by the
context, on topic, and readable? Checking answers at scale requires an
automatic judge, but an automatic judge is only usable if it agrees with
human judgment. Otherwise every number it produces is noise.

This report is that check. The judge scored a benchmark of answers that
humans had already labeled against the same rubric, and the tables below
quantify how closely the judge's scores track the human ones, dimension by
dimension. The same measurement runs in CI: if agreement falls below the
thresholds in the gate table, the build fails and this judge is not trusted
to evaluate anything until the regression is understood.

Each answer is scored on three 1-5 dimensions (groundedness: is every claim
supported by the context; relevance: does it answer the question; coherence:
is it readable prose) plus a binary ship/no-ship verdict (overall_pass)."""

METRICS_GUIDE = """\
## How to read the metrics

No single number captures agreement, so the tables report several statistics
that cover each other's blind spots.

For the 1-5 dimensions (judge score vs. human score, per dimension):

- **QWK (quadratic-weighted kappa)** is the headline statistic. It measures
  agreement corrected for chance, and it penalizes disagreements by their
  squared distance, so scoring a 1 as a 5 costs far more than scoring a 4 as
  a 5. It is the standard statistic for ordinal human-rating agreement
  (essay scoring, annotation studies). 1.0 is perfect, 0 is chance level,
  negative is worse than chance.
- **Spearman** is rank correlation: do the judge and the humans put the
  answers in the same order, even if their absolute scores differ? A judge
  that is consistently harsher than humans but ranks identically still gets
  a high Spearman. That is the property that matters most when the judge is
  used to compare two model versions.
- **Pearson** is linear correlation of the raw scores. Read it together
  with MAE: high Pearson with high MAE means the judge tracks humans but
  with a systematic offset.
- **MAE (mean absolute error)** is the average distance from the human
  score, in rubric points: 0.5 means the judge is half a point off on
  average. It is the most intuitive number here, but it is blind to chance
  agreement, which is why it does not gate anything on its own.
- **Exact / Within +/-1** are the simplest views: the fraction of examples
  where the judge matched the human score exactly, or within one point.
  Easy to read, but inflated whenever the labels cluster on a few values.

For the binary ship/no-ship verdict:

- **Accuracy** is the fraction of matching verdicts. It is reported because
  everyone asks for it, but it is inflated by class imbalance, which is why
  it does not gate anything.
- **Cohen's kappa** corrects accuracy for chance: 0 means no better than
  guessing the base rates, 1 means perfect agreement. This is the headline
  binary statistic and one of the two numbers the calibration gate enforces.
- **Precision / Recall / F1** (treating "ship" as the positive class) split
  the errors by their cost: precision asks "when the judge said ship, how
  often did humans agree?" (low precision means bad answers reach users);
  recall asks "of the answers humans would ship, how many did the judge
  pass?" (low recall means good answers get blocked). F1 combines the two.
- The **confusion matrix** shows where the errors actually live, so a low
  kappa can be traced to its failure direction."""


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown(
    result: CalibrationResult,
    dataset_name: str,
    gate_report: GateReport | None = None,
    max_disagreements: int = 8,
) -> str:
    backend_note = BACKEND_DESCRIPTIONS.get(
        result.backend_name, f"a judge backend named `{result.backend_name}`"
    )

    lines: list[str] = []
    lines.append("# Judge calibration report")
    lines.append("")
    lines.append(BACKGROUND)
    lines.append("")
    lines.append("## This run")
    lines.append("")
    lines.append(f"- **Judge backend:** `{result.backend_name}` - {backend_note}")
    lines.append(
        f"- **Benchmark:** `{dataset_name}` ({result.n_examples} examples with "
        "human labels; labeling procedure and rubric in "
        "`data/ANNOTATION_GUIDELINES.md`)"
    )
    lines.append("")
    lines.append(METRICS_GUIDE)
    lines.append("")

    lines.append("## Ordinal agreement (judge vs. human, per dimension)")
    lines.append("")
    lines.append("| Dimension | QWK | Spearman | Pearson | MAE | Exact | Within +/-1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for dim in DIMENSIONS:
        o = result.ordinal[dim]
        lines.append(
            f"| {dim} | {_fmt(o.qwk)} | {_fmt(o.spearman)} | {_fmt(o.pearson)} "
            f"| {_fmt(o.mae, 2)} | {_fmt(o.exact, 2)} | {_fmt(o.within_one, 2)} |"
        )
    lines.append("")

    p = result.overall_pass
    lines.append("## Ship/no-ship agreement (overall_pass)")
    lines.append("")
    lines.append("| Accuracy | Cohen's kappa | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| {_fmt(p.accuracy, 2)} | {_fmt(p.kappa)} | {_fmt(p.precision, 2)} "
        f"| {_fmt(p.recall, 2)} | {_fmt(p.f1, 2)} |"
    )
    lines.append("")
    lines.append("Confusion matrix (rows = human, columns = judge):")
    lines.append("")
    lines.append("|  | judge: fail | judge: pass |")
    lines.append("|---|---|---|")
    lines.append(f"| **human: fail** | {p.confusion[0][0]} | {p.confusion[0][1]} |")
    lines.append(f"| **human: pass** | {p.confusion[1][0]} | {p.confusion[1][1]} |")
    lines.append("")

    if gate_report is not None:
        lines.append("## Calibration gate")
        lines.append("")
        lines.append(
            "The two agreement statistics the CI build enforces. If either "
            "falls below its threshold the build fails, because a judge that "
            "no longer agrees with humans must not be trusted to gate "
            "releases."
        )
        lines.append("")
        lines.append("| Check | Value | Threshold | Result |")
        lines.append("|---|---|---|---|")
        for check in gate_report.checks:
            verdict = "PASS" if check.passed else "FAIL"
            lines.append(
                f"| {check.name} | {_fmt(check.value)} | >= {_fmt(check.threshold)} "
                f"| {verdict} |"
            )
        lines.append("")

    lines.append("## Failure analysis: largest judge-human disagreements")
    lines.append("")
    lines.append(
        "The examples below are where the judge's score differs most from the "
        "human label, worst first. Each entry shows both rationales so the "
        "mechanism of the disagreement is visible, not just its size."
    )
    lines.append("")
    if not result.disagreements:
        lines.append("No disagreements - judge matches human labels exactly.")
    else:
        shown = min(max_disagreements, len(result.disagreements))
        lines.append(
            f"{len(result.disagreements)} of {result.n_examples} examples have at "
            f"least one disagreement; the worst {shown} are shown."
        )
        lines.append("")
        for d in result.disagreements[:max_disagreements]:
            lines.extend(_render_disagreement(d))
    lines.append("")
    return "\n".join(lines)


def _render_disagreement(d: Disagreement) -> list[str]:
    e = d.example
    tags = ", ".join(e.tags) if e.tags else "-"
    flip = " | **ship/no-ship flip**" if d.pass_flip else ""
    judge_dim = d.judge_score.dimension(d.dimension)
    return [
        f"### `{e.id}` - {d.dimension}: human {d.human_score} vs judge {judge_dim}"
        f" (delta {d.delta:+d}){flip}",
        "",
        f"- **Tags:** {tags}",
        f"- **Question:** {e.question}",
        f"- **Answer:** {e.candidate_answer}",
        f"- **Human rationale:** {e.human_rationale or '-'}",
        f"- **Judge rationale:** {d.judge_score.rationale or '-'}",
        "",
    ]
