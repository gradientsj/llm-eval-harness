"""Command-line interface.

Commands:
  calibrate  Run the judge over the labeled benchmark, write the calibration
             report, and (with --check-gates) enforce the calibration gate.
  baseline   Freeze aggregate judge scores for the regression gate.
  gate       Run the judge over a candidate answer set and compare aggregates
             against the frozen baseline.

Exit codes: 0 success / gates pass, 1 a gate failed, 2 usage or data error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .backends import create_backend
from .calibration import run_calibration
from .dataset import DatasetError, load_dataset
from .judge import Judge
from .regression import (
    aggregate_scores,
    check_calibration_gate,
    check_regression_gate,
    load_baseline,
    write_baseline,
)
from .report import render_markdown

DEFAULT_DATASET = "data/grounded_qa.jsonl"
DEFAULT_BASELINE = "data/baseline_scores.json"


def _load_dotenv() -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides the real env."""
    env_file = Path(".env")
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=["lexical", "anthropic"],
        default="lexical",
        help="judge backend (default: lexical - deterministic word-overlap "
        "baseline, no credentials needed)",
    )


def cmd_calibrate(args: argparse.Namespace) -> int:
    examples = load_dataset(args.dataset)
    judge = Judge(create_backend(args.backend))
    print(f"Calibrating {args.backend} judge on {len(examples)} labeled examples...")
    result = run_calibration(examples, judge, progress=args.backend != "lexical")

    gate_report = check_calibration_gate(result)
    out_path = Path(args.out or f"reports/calibration_report_{args.backend}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_markdown(result, dataset_name=args.dataset, gate_report=gate_report),
        encoding="utf-8",
    )
    metrics_path = out_path.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(result.to_dict() | {"gate": gate_report.to_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Report written to {out_path} (metrics: {metrics_path})")

    for check in gate_report.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: "
              f"{check.value:.3f} (>= {check.threshold:.3f})")
    if args.check_gates and not gate_report.passed:
        print("Calibration gate FAILED - judge does not agree with humans well "
              "enough to be trusted as a gate.", file=sys.stderr)
        return 1
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    examples = load_dataset(args.dataset)
    judge = Judge(create_backend(args.backend))
    print(f"Scoring {len(examples)} examples with {args.backend} judge for baseline...")
    scores = judge.score_dataset(examples, progress=args.backend != "lexical")
    aggregate = aggregate_scores(scores)
    write_baseline(args.out, args.backend, args.dataset, aggregate)
    print(f"Baseline written to {args.out}: "
          + ", ".join(f"{k}={v:.3f}" for k, v in aggregate.items()))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    examples = load_dataset(args.candidates)
    judge = Judge(create_backend(args.backend))
    print(f"Scoring {len(examples)} candidate answers with {args.backend} judge...")
    scores = judge.score_dataset(examples, progress=args.backend != "lexical")
    current = aggregate_scores(scores)

    baseline = load_baseline(args.baseline)
    if baseline.get("backend") != args.backend:
        print(
            f"error: baseline was frozen with backend "
            f"{baseline.get('backend')!r} but gate is running with "
            f"{args.backend!r}; scores are not comparable. Re-freeze with "
            f"'evalharness baseline --backend {args.backend}'.",
            file=sys.stderr,
        )
        return 2

    report = check_regression_gate(current, baseline["aggregate"])
    for check in report.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    if not report.passed:
        print("Regression gate FAILED - candidate quality dropped below baseline "
              "tolerance.", file=sys.stderr)
        return 1
    print("Regression gate passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalharness",
        description="LLM-as-judge evaluation with human calibration and CI gating.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="measure judge-vs-human agreement")
    _add_common(p_cal)
    p_cal.add_argument("--dataset", default=DEFAULT_DATASET)
    p_cal.add_argument("--out", default=None, help="report path (default: reports/)")
    p_cal.add_argument(
        "--check-gates",
        action="store_true",
        help="exit 1 if judge-human agreement is below gate thresholds",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    p_base = sub.add_parser("baseline", help="freeze aggregate scores for the gate")
    _add_common(p_base)
    p_base.add_argument("--dataset", default=DEFAULT_DATASET)
    p_base.add_argument("--out", default=DEFAULT_BASELINE)
    p_base.set_defaults(func=cmd_baseline)

    p_gate = sub.add_parser("gate", help="regression-gate a candidate answer set")
    _add_common(p_gate)
    p_gate.add_argument(
        "--candidates",
        default=DEFAULT_DATASET,
        help="JSONL of candidate answers to score (human labels optional)",
    )
    p_gate.add_argument("--baseline", default=DEFAULT_BASELINE)
    p_gate.set_defaults(func=cmd_gate)

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"file not found: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
