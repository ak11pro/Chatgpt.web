"""Explicit interface that prevents trackers from silently using labels as frames."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FrameSource(ABC):
    """Read a single HSI frame as a `(height, width, bands)` array."""

    @abstractmethod
    def frame(self, sequence: str, frame_index: int) -> np.ndarray:
        """Return raw HSI values; implementations must raise if unavailable."""

