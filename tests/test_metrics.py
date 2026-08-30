import pytest

np = pytest.importorskip("numpy")

from hotc_tracker.evaluation import ope_metrics


def test_perfect_boxes_have_perfect_ope_metrics() -> None:
    boxes = np.array([[0, 0, 10, 10], [4, 3, 8, 6]], dtype=float)
    result = ope_metrics(boxes, boxes)
    assert result["mean_iou"] == 1.0
    assert result["precision_at_20"] == 1.0
    assert result["success_auc"] == 1.0


def test_metrics_reject_mismatched_shape() -> None:
    with pytest.raises(ValueError):
        ope_metrics(np.zeros((2, 4)), np.zeros((3, 4)))
