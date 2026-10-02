"""Synthetic regression tests for strict, competition-aligned PQ behaviour."""

import numpy as np

from solar_filament.pipeline import compute_pq, match_instances, relabel_sequential


def two_instances() -> np.ndarray:
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[7:10, 7:10] = 2
    return labels


def test_perfect_match() -> None:
    truth = two_instances()
    result = compute_pq(truth, truth)
    assert result["pq"] == 1 and result["tp"] == 2 and result["fp"] == result["fn"] == 0


def test_false_positive_and_false_negative() -> None:
    truth = two_instances()
    false_positive = truth.copy(); false_positive[1:3, 8:10] = 3
    assert compute_pq(false_positive, truth)["fp"] == 1
    false_negative = truth.copy(); false_negative[7:10, 7:10] = 0
    assert compute_pq(false_negative, truth)["fn"] == 1


def test_merge_and_split_diagnostics() -> None:
    truth = two_instances()
    merged = np.where(truth > 0, 1, 0)
    assert compute_pq(merged, truth)["many_to_one"] >= 1
    one = np.zeros((12, 12), dtype=np.int32); one[2:8, 2:8] = 1
    split = one.copy(); split[2:8, 5] = 0; split[2:8, 6] = 2
    result = compute_pq(split, one)
    assert result["one_to_many"] >= 1 and result["pq"] < 1


def test_strict_iou_threshold() -> None:
    assert match_instances(np.array([[0.5]]), 0.5) == []
    assert match_instances(np.array([[0.500001]]), 0.5) == [(0, 0, 0.500001)]


def test_relabel_preserves_adjacent_watershed_instances() -> None:
    # Labels 7 and 19 deliberately touch. A binary re-label would merge them; sequential remapping must not.
    watershed_labels = np.array([[0, 7, 7, 19, 19]], dtype=np.int32)
    assert np.array_equal(relabel_sequential(watershed_labels), np.array([[0, 1, 1, 2, 2]], dtype=np.int32))
