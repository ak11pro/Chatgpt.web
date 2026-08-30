"""Metrics for xywh boxes, matching HOTC one-pass tracking evaluation needs."""

from __future__ import annotations

import numpy as np


def _iou(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    p2 = pred[:, :2] + np.maximum(pred[:, 2:], 0)
    t2 = truth[:, :2] + np.maximum(truth[:, 2:], 0)
    start = np.maximum(pred[:, :2], truth[:, :2])
    end = np.minimum(p2, t2)
    intersection = np.prod(np.maximum(end - start, 0), axis=1)
    union = np.prod(np.maximum(pred[:, 2:], 0), axis=1) + np.prod(np.maximum(truth[:, 2:], 0), axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def ope_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Return mean IoU, precision@20, and success AUC over thresholds [0, 1]."""
    pred = np.asarray(predictions, dtype=float)
    truth = np.asarray(ground_truth, dtype=float)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError("predictions and ground_truth must both have shape (n, 4)")
    valid = np.isfinite(pred).all(axis=1) & np.isfinite(truth).all(axis=1)
    if not valid.any():
        raise ValueError("no valid box pairs")
    pred, truth = pred[valid], truth[valid]
    centers_pred = pred[:, :2] + pred[:, 2:] / 2
    centers_truth = truth[:, :2] + truth[:, 2:] / 2
    center_error = np.linalg.norm(centers_pred - centers_truth, axis=1)
    ious = _iou(pred, truth)
    thresholds = np.linspace(0, 1, 101)
    success = np.mean(ious[:, None] >= thresholds[None, :])
    return {"mean_iou": float(ious.mean()), "precision_at_20": float((center_error <= 20).mean()), "success_auc": float(success), "frames": float(len(pred))}

