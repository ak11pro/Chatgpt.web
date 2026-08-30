"""Deterministic HSI-to-RGB adapters for RGB-only spatial trackers."""

from __future__ import annotations

import numpy as np


def validate_cube(cube: np.ndarray, expected_bands: int | None = None) -> np.ndarray:
    cube = np.asarray(cube)
    if cube.ndim != 3 or cube.shape[-1] < 3:
        raise ValueError("cube must have shape (height, width, bands>=3)")
    if expected_bands is not None and cube.shape[-1] != expected_bands:
        raise ValueError(f"expected {expected_bands} bands, got {cube.shape[-1]}")
    return cube.astype(np.float32, copy=False)


def robust_band_rgb(cube: np.ndarray, bands: tuple[int, int, int] | None = None) -> np.ndarray:
    """Map three evenly spaced bands to uint8 RGB via per-band percentiles."""
    cube = validate_cube(cube)
    if bands is None:
        bands = tuple(np.linspace(0, cube.shape[-1] - 1, 3, dtype=int).tolist())
    if len(bands) != 3 or min(bands) < 0 or max(bands) >= cube.shape[-1]:
        raise ValueError("bands must contain three valid band indices")
    selected = cube[..., list(bands)]
    low = np.percentile(selected, 1, axis=(0, 1), keepdims=True)
    high = np.percentile(selected, 99, axis=(0, 1), keepdims=True)
    normalized = np.clip((selected - low) / np.maximum(high - low, 1e-6), 0, 1)
    return np.rint(normalized * 255).astype(np.uint8)


def pca_rgb(cube: np.ndarray) -> np.ndarray:
    """Project a frame's spectral vectors to three standardized PCA components."""
    cube = validate_cube(cube)
    height, width, bands = cube.shape
    values = cube.reshape(-1, bands)
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ components[:3].T
    return robust_band_rgb(projected.reshape(height, width, 3), bands=(0, 1, 2))

