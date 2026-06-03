"""Agreement statistics, implemented in pure Python.

These are deliberately implemented from scratch rather than imported from
scipy/sklearn: the formulas are small, the implementations are unit-tested
against hand-computed values, and keeping them legible makes the calibration
math auditable - which is the point of a calibration harness.

Conventions: undefined statistics (zero variance, empty input) return
float('nan') rather than a silently-wrong 0.0. Display code renders nan as
"n/a"; gate code treats nan as a failure, because "we cannot demonstrate
calibration" should not pass a calibration gate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def mae(a: Sequence[float], b: Sequence[float]) -> float:
    _check_paired(a, b)
    if not a:
        return float("nan")
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    _check_paired(y_true, y_pred)
    if not y_true:
        return float("nan")
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def exact_and_within_one(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[float, float]:
    """Exact-match rate and |delta| <= 1 rate, the two most readable ordinal stats."""
    _check_paired(y_true, y_pred)
    if not y_true:
        return float("nan"), float("nan")
    exact = sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)
    within = sum(abs(t - p) <= 1 for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)
    return exact, within


def confusion_matrix(
    y_true: Sequence, y_pred: Sequence, labels: Sequence
) -> list[list[int]]:
    """Rows = true label, columns = predicted label, in `labels` order."""
    _check_paired(y_true, y_pred)
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[index[t]][index[p]] += 1
    return matrix


def precision_recall_f1(
    y_true: Sequence[bool], y_pred: Sequence[bool]
) -> tuple[float, float, float]:
    """Binary precision/recall/F1 with True as the positive class."""
    _check_paired(y_true, y_pred)
    tp = sum(t and p for t, p in zip(y_true, y_pred, strict=True))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred, strict=True))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred, strict=True))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def cohen_kappa(y_true: Sequence, y_pred: Sequence) -> float:
    """Unweighted Cohen's kappa: agreement corrected for chance."""
    _check_paired(y_true, y_pred)
    n = len(y_true)
    if n == 0:
        return float("nan")
    labels = sorted(set(y_true) | set(y_pred), key=str)
    po = accuracy(y_true, y_pred)
    pe = 0.0
    for label in labels:
        pe += (
            sum(t == label for t in y_true) / n
        ) * (
            sum(p == label for p in y_pred) / n
        )
    if pe == 1.0:
        # Both raters constant: perfect observed agreement is uninformative.
        return float("nan")
    return (po - pe) / (1.0 - pe)


def quadratic_weighted_kappa(
    y_true: Sequence[int], y_pred: Sequence[int], min_rating: int, max_rating: int
) -> float:
    """Quadratic-weighted kappa - the standard agreement statistic for ordinal
    ratings (penalizes a 5-vs-1 disagreement more than a 5-vs-4)."""
    _check_paired(y_true, y_pred)
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max_rating - min_rating + 1
    observed = [[0.0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred, strict=True):
        observed[t - min_rating][p - min_rating] += 1

    hist_true = [sum(t == r for t in y_true) for r in range(min_rating, max_rating + 1)]
    hist_pred = [sum(p == r for p in y_pred) for r in range(min_rating, max_rating + 1)]

    numerator = 0.0
    denominator = 0.0
    for i in range(k):
        for j in range(k):
            weight = ((i - j) ** 2) / ((k - 1) ** 2)
            expected = hist_true[i] * hist_pred[j] / n
            numerator += weight * observed[i][j]
            denominator += weight * expected
    if denominator == 0:
        return float("nan")
    return 1.0 - numerator / denominator


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    _check_paired(a, b)
    n = len(a)
    if n < 2:
        return float("nan")
    mean_a, mean_b = mean(a), mean(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a == 0 or var_b == 0:
        return float("nan")
    return cov / math.sqrt(var_a * var_b)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""
    _check_paired(a, b)
    return pearson(_average_ranks(a), _average_ranks(b))


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        # Average rank for the tie group spanning positions i..j (1-based ranks).
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _check_paired(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"paired sequences must match in length: {len(a)} != {len(b)}")
