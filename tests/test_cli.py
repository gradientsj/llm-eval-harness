from evalharness.cli import main


def test_calibrate_smoke(tmp_path):
    out = tmp_path / "report.md"
    code = main(
        ["calibrate", "--backend", "lexical", "--out", str(out), "--check-gates"]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "# Judge calibration report" in text
    assert "Failure analysis" in text
    assert out.with_suffix(".json").exists()


def test_baseline_then_gate_passes(tmp_path):
    baseline = tmp_path / "baseline.json"
    assert main(["baseline", "--backend", "lexical", "--out", str(baseline)]) == 0
    assert main(["gate", "--backend", "lexical", "--baseline", str(baseline)]) == 0


def test_gate_fails_on_regressed_candidates(tmp_path):
    baseline = tmp_path / "baseline.json"
    assert main(["baseline", "--backend", "lexical", "--out", str(baseline)]) == 0
    code = main(
        [
            "gate",
            "--backend",
            "lexical",
            "--baseline",
            str(baseline),
            "--candidates",
            "data/grounded_qa_regressed.jsonl",
        ]
    )
    assert code == 1


def test_gate_rejects_backend_mismatch(tmp_path):
    baseline = tmp_path / "baseline.json"
    assert main(["baseline", "--backend", "lexical", "--out", str(baseline)]) == 0
    text = baseline.read_text(encoding="utf-8").replace('"lexical"', '"anthropic"', 1)
    baseline.write_text(text, encoding="utf-8")
    assert main(["gate", "--backend", "lexical", "--baseline", str(baseline)]) == 2
