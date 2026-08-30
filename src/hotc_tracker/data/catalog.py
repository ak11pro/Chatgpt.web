"""Safe discovery of HOTC annotations and optional raw-frame mounts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ANNOTATION_NAMES = ("2026training.csv",)
SUBMISSION_NAMES = ("sample_submisson.csv", "sample_submission.csv")
FRAME_SUFFIXES = frozenset({".npy", ".npz", ".mat", ".h5", ".hdf5", ".tif", ".tiff"})


@dataclass(frozen=True)
class DataCatalog:
    """Discovered inputs; paths are optional because Kaggle mounts vary."""

    roots: tuple[str, ...]
    training_csv: str | None
    sample_submission_csv: str | None
    frame_files: tuple[str, ...]

    @property
    def frames_available(self) -> bool:
        return bool(self.frame_files)

    def as_dict(self) -> dict[str, object]:
        return asdict(self) | {"frames_available": self.frames_available}


def _existing_roots(roots: Iterable[str | Path] | None) -> list[Path]:
    requested = list(roots or [])
    if not requested:
        requested = ["/kaggle/input/competitions", "/kaggle/input/datasets", "/kaggle/input"]
    unique: list[Path] = []
    for value in requested:
        path = Path(value).expanduser()
        if path.is_dir() and path not in unique:
            unique.append(path)
    return unique


def _find_named(roots: list[Path], names: tuple[str, ...]) -> Path | None:
    candidates = [path for root in roots for name in names for path in root.rglob(name)]
    return min(candidates, key=lambda path: (len(path.parts), str(path))) if candidates else None


def discover_catalog(roots: Iterable[str | Path] | None = None, frame_limit: int = 32) -> DataCatalog:
    """Discover known files without assuming a fixed Kaggle Dataset slug.

    `frame_limit` bounds the audit's file list and avoids serializing a massive
    raw-video mount. A nonempty list proves only that candidate frame files were
    mounted; a concrete loader must still validate shape and sensor metadata.
    """
    if frame_limit < 1:
        raise ValueError("frame_limit must be positive")
    existing = _existing_roots(roots)
    training = _find_named(existing, ANNOTATION_NAMES)
    sample = _find_named(existing, SUBMISSION_NAMES)
    frames: list[Path] = []
    for root in existing:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES:
                frames.append(path)
                if len(frames) >= frame_limit:
                    break
        if len(frames) >= frame_limit:
            break
    return DataCatalog(
        roots=tuple(str(path) for path in existing),
        training_csv=str(training) if training else None,
        sample_submission_csv=str(sample) if sample else None,
        frame_files=tuple(str(path) for path in frames),
    )

