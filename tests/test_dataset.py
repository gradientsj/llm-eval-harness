import pytest

from evalharness.dataset import DatasetError, load_dataset, require_labels

MAIN = "data/grounded_qa.jsonl"
REGRESSED = "data/grounded_qa_regressed.jsonl"


def test_main_dataset_loads_fully_labeled():
    examples = load_dataset(MAIN)
    assert len(examples) == 30
    assert all(e.human is not None for e in examples)
    assert len({e.id for e in examples}) == 30
    require_labels(examples)  # must not raise


def test_main_dataset_labels_follow_pass_rule():
    # overall_pass must be mechanically derivable from the rubric thresholds.
    for e in load_dataset(MAIN):
        derived = (
            e.human.groundedness >= 4 and e.human.relevance >= 3 and e.human.coherence >= 3
        )
        assert e.human.overall_pass == derived, f"{e.id}: label violates pass rule"


def test_regressed_dataset_is_unlabeled_but_aligned():
    main = load_dataset(MAIN)
    regressed = load_dataset(REGRESSED)
    assert [e.id for e in regressed] == [e.id for e in main]
    assert all(e.human is None for e in regressed)
    with pytest.raises(DatasetError):
        require_labels(regressed)


def test_duplicate_ids_rejected(tmp_path):
    record = (
        '{"id": "x", "context": "c", "question": "q", "candidate_answer": "a"}'
    )
    path = tmp_path / "dup.jsonl"
    path.write_text(record + "\n" + record + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(path)


def test_out_of_range_label_rejected(tmp_path):
    record = (
        '{"id": "x", "context": "c", "question": "q", "candidate_answer": "a", '
        '"human": {"groundedness": 6, "relevance": 5, "coherence": 5, "overall_pass": true}}'
    )
    path = tmp_path / "bad.jsonl"
    path.write_text(record + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="groundedness"):
        load_dataset(path)
