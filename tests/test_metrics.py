import math

import pytest

from evalharness.metrics import (
    accuracy,
    cohen_kappa,
    confusion_matrix,
    exact_and_within_one,
    mae,
    mean,
    pearson,
    precision_recall_f1,
    quadratic_weighted_kappa,
    spearman,
)


def test_mean_and_mae():
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert math.isnan(mean([]))
    assert mae([1, 2], [2, 4]) == pytest.approx(1.5)


def test_accuracy():
    assert accuracy([1, 2, 3], [1, 2, 4]) == pytest.approx(2 / 3)


def test_exact_and_within_one():
    exact, within = exact_and_within_one([1, 3, 5], [1, 4, 1])
    assert exact == pytest.approx(1 / 3)
    assert within == pytest.approx(2 / 3)


def test_confusion_matrix():
    matrix = confusion_matrix([False, False, True], [False, True, True], [False, True])
    assert matrix == [[1, 1], [0, 1]]


def test_precision_recall_f1():
    # tp=1, fp=1, fn=1
    p, r, f1 = precision_recall_f1([True, True, False], [True, False, True])
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_cohen_kappa_hand_computed():
    # po = 0.75; marginals give pe = 0.5; kappa = 0.5.
    y_true = [True, True, False, False]
    y_pred = [True, False, False, False]
    assert cohen_kappa(y_true, y_pred) == pytest.approx(0.5)


def test_cohen_kappa_constant_raters_is_nan():
    assert math.isnan(cohen_kappa([True, True], [True, True]))


def test_qwk_perfect_agreement():
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1, 5) == pytest.approx(1.0)


def test_qwk_independent_ratings_is_zero():
    # 2x2 balanced independence: observed disagreement equals expected.
    assert quadratic_weighted_kappa([1, 1, 2, 2], [1, 2, 1, 2], 1, 2) == pytest.approx(0.0)


def test_qwk_penalizes_distance():
    near = quadratic_weighted_kappa([1, 2, 3, 4, 5], [2, 3, 4, 5, 5], 1, 5)
    far = quadratic_weighted_kappa([1, 2, 3, 4, 5], [5, 4, 5, 1, 1], 1, 5)
    assert near > far


def test_pearson_known_values():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)
    assert math.isnan(pearson([1, 1, 1], [1, 2, 3]))


def test_spearman_monotonic_nonlinear():
    a = [1, 2, 3, 4]
    b = [1, 4, 9, 16]
    assert spearman(a, b) == pytest.approx(1.0)
    assert pearson(a, b) < 1.0


def test_spearman_handles_ties():
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy([1], [1, 2])
