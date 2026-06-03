"""Markdown calibration report with ranked failure analysis.

Reports intentionally carry no timestamp: a re-run on identical inputs must
produce an identical file, so report diffs in PRs show real changes only.
"""

from __future__ import annotations

import math

from .calibration import CalibrationResult, Disagreement
from .regression import GateReport
from .rubric import DIMENSIONS


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
    lines: list[str] = []
    lines.append("# Judge calibration report")
    lines.append("")
    lines.append(f"- **Judge backend:** `{result.backend_name}`")
    lines.append(f"- **Benchmark:** `{dataset_name}` ({result.n_examples} examples)")
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
