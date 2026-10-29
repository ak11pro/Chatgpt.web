"""Competition-compliant training and inference pipeline for MAGFiLO COCO data.

Predictions are always derived from model logits.  This module intentionally has
no test-image lookup tables or precomputed RLE payloads.
"""

from __future__ import annotations

import json
import logging
import platform
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .disk import Disk, detect_disk, flatten

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """All training/inference choices recorded alongside every experiment."""

    seed: int = 42
    image_size: int = 2048
    train_tile_size: int = 768
    tile_overlap: float = 0.35
    inference_batch_size: int = 4
    weighted_tile_blending: bool = True
    in_channels: int = 2
    encoder: str = "resnet34"
    pretrained_encoder: bool = True
    epochs: int = 50
    batch_size: int = 2
    gradient_accumulation: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    use_amp: bool = True
    val_fraction: float = 0.2
    threshold: float = 0.40
    min_component_area: int = 50
    morphology_kernel: int = 0
    watershed: bool = True
    watershed_peak_distance: int = 18
    boundary_mode: str = "guidance"  # disabled, guidance, separator
    boundary_threshold: float = 0.65
    boundary_head: bool = True
    use_tta: bool = False
    tta_mode: str = "light"
    use_solar_disk_mask: bool = True
    debug_mode: bool = False
    max_train_images: int | None = None
    samples_per_epoch: int | None = 1600
    mask_cache_dir: str = "/kaggle/working/solar_filament_mask_cache_v2"
    positive_probability: float = 0.50
    hard_negative_probability: float = 0.25
    random_probability: float = 0.25
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "bce": 0.35, "dice": 0.45, "tversky": 0.20, "boundary": 0.20,
    })


CONFIG = PipelineConfig()


@dataclass(frozen=True)
class CompetitionData:
    root: Path
    annotation_json: Path
    train_images: Path
    test_images: Path


@dataclass
class InstanceRecord:
    """One annotator-specific filament target; identifiers are opaque strings."""

    image_id: str
    annotation_id: str
    bbox: tuple[float, float, float, float]
    area: float
    segmentation: Any


@dataclass
class CocoIndex:
    """COCO entries remain separate even when they share one physical JPEG."""

    images: dict[str, dict[str, Any]]
    annotations_by_image: dict[str, list[InstanceRecord]]
    categories: list[dict[str, Any]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_competition_data(root: str | Path = "/kaggle/input") -> CompetitionData:
    """Discover MAGFiLO files by content rather than a fixed Kaggle slug."""
    root_path = Path(root)
    json_files = sorted(root_path.rglob("*.json"))
    candidates = [p for p in json_files if "annot" in p.name.lower() or "coco" in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(f"No COCO annotation JSON found under {root_path}")
    annotation_json = candidates[0]
    image_dirs = [p for p in root_path.rglob("*") if p.is_dir() and "image" in p.name.lower()]
    train_dirs = [p for p in image_dirs if "train" in str(p).lower()]
    test_dirs = [p for p in image_dirs if "test" in str(p).lower()]
    if not train_dirs or not test_dirs:
        raise FileNotFoundError("Could not identify both train_images and test_images directories")
    data = CompetitionData(root_path, annotation_json, train_dirs[0], test_dirs[0])
    index = load_coco_annotations(data.annotation_json)
    sample = next(iter(index.images.values()))
    test_count = len(list_image_files(data.test_images))
    areas = [a.area for records in index.annotations_by_image.values() for a in records]
    print(json.dumps({
        "training_images": len(index.images), "annotations": len(areas), "test_images": test_count,
        "sample_dimensions": [sample.get("width"), sample.get("height")],
        "filament_instances": len(areas),
        "instance_area": ({"min": min(areas), "p1": float(np.percentile(areas, 1)), "p5": float(np.percentile(areas, 5)),
                           "p25": float(np.percentile(areas, 25)), "median": float(np.median(areas)),
                           "p75": float(np.percentile(areas, 75)), "p95": float(np.percentile(areas, 95)), "max": max(areas)} if areas else {}),
        "categories": index.categories,
    }, indent=2, default=str))
    return data


def list_image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})


def get_physical_observation_key(image_meta: dict[str, Any]) -> str:
    """Physical JPEG identity, deliberately distinct from the COCO annotation entry ID."""
    return Path(str(image_meta["file_name"])).name


def get_annotation_entry_key(image_meta: dict[str, Any]) -> str:
    """Opaque COCO image-entry identity; never coerce it to an integer."""
    image_id = image_meta["id"]
    if not isinstance(image_id, str):
        raise TypeError(f"MAGFiLO image id must be a string, got {type(image_id).__name__}")
    return image_id


def group_by_physical_observation(index: CocoIndex) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for image_id, image_meta in index.images.items():
        grouped[get_physical_observation_key(image_meta)].append(image_id)
    return dict(grouped)


def load_coco_annotations(annotation_json: str | Path) -> CocoIndex:
    """Load every COCO instance without collapsing per-filament identity."""
    with Path(annotation_json).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not raw.get("images") or not raw.get("annotations"):
        raise ValueError("COCO JSON must include non-empty images and annotations lists")
    images: dict[str, dict[str, Any]] = {}
    for image in raw["images"]:
        image_id = get_annotation_entry_key(image)
        if image_id in images:
            raise ValueError(f"Duplicate COCO image id: {image_id}")
        if int(image["width"]) != 2048 or int(image["height"]) != 2048:
            raise ValueError(f"MAGFiLO image {image_id} must be 2048x2048, got {image['width']}x{image['height']}")
        images[image_id] = image
    grouped: dict[str, list[InstanceRecord]] = defaultdict(list)
    for ann in raw["annotations"]:
        if not isinstance(ann.get("id"), str) or not isinstance(ann.get("image_id"), str):
            raise TypeError("MAGFiLO annotation id and image_id must be strings")
        if ann["image_id"] not in images:
            raise ValueError(f"Annotation {ann['id']} references missing image id {ann['image_id']}")
        if ann.get("iscrowd", 0) != 0:
            raise ValueError(f"MAGFiLO annotation {ann['id']} unexpectedly has iscrowd != 0")
        if ann.get("iscrowd", 0):
            LOGGER.warning("Crowd annotation %s is retained as a single instance", ann["id"])
        grouped[ann["image_id"]].append(InstanceRecord(
            image_id=ann["image_id"], annotation_id=ann["id"],
            bbox=tuple(float(x) for x in ann.get("bbox", (0, 0, 0, 0))),
            area=float(ann.get("area", 0)), segmentation=ann["segmentation"],
        ))
    index = CocoIndex(images, dict(grouped), raw.get("categories", []))
    physical = group_by_physical_observation(index)
    print(json.dumps({"coco_annotation_image_entries": len(index.images), "unique_physical_observations": len(physical),
                      "annotation_count": sum(map(len, index.annotations_by_image.values())),
                      "category_names": [category.get("name") for category in index.categories]}, indent=2))
    return index


def _decode_annotation(instance: InstanceRecord, height: int, width: int) -> np.ndarray:
    segmentation = instance.segmentation
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = coco_mask.frPyObjects(rle, height, width)
        decoded = coco_mask.decode(rle)
        return decoded.max(axis=2) if decoded.ndim == 3 else decoded
    rles = coco_mask.frPyObjects(segmentation, height, width)
    decoded = coco_mask.decode(rles)
    return decoded.max(axis=2) if decoded.ndim == 3 else decoded


def build_instance_masks(index: CocoIndex, image_id: str) -> list[np.ndarray]:
    image = index.images[image_id]
    return [_decode_annotation(ann, int(image["height"]), int(image["width"])).astype(bool)
            for ann in index.annotations_by_image.get(image_id, [])]


def build_semantic_mask(index: CocoIndex, image_id: str) -> np.ndarray:
    image = index.images[image_id]
    mask = np.zeros((int(image["height"]), int(image["width"])), dtype=bool)
    for instance in build_instance_masks(index, image_id):
        mask |= instance
    return mask


def build_instance_id_mask(index: CocoIndex, image_id: str) -> np.ndarray:
    image = index.images[image_id]
    output = np.zeros((int(image["height"]), int(image["width"])), dtype=np.int32)
    for label, instance in enumerate(build_instance_masks(index, image_id), start=1):
        # COCO masks should not overlap.  Keep the first label if they do and log it.
        if np.any((output > 0) & instance):
            LOGGER.warning("Overlapping COCO instances in image %s", image_id)
        output[(output == 0) & instance] = label
    return output


def group_train_validation_split(index: CocoIndex, fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Split by UTC-like observation date, keeping each JPEG and every annotator view together.

    Filaments can persist across neighboring observations. Grouping by date gives
    a less optimistic validation estimate than merely separating annotation IDs
    or filenames while still preserving every annotator-specific target.
    """
    ids = np.array(sorted(index.images), dtype=object)
    physical = np.array([
        get_physical_observation_key(index.images[image_id]) for image_id in ids
    ], dtype=object)
    # MAGFiLO filenames begin with YYYYMMDDhhmmss; use date rather than image ID.
    groups = np.array([
        Path(get_physical_observation_key(index.images[image_id])).stem[:8]
        for image_id in ids
    ], dtype=object)
    if len(set(groups)) < 2:
        raise ValueError("At least two distinct observation dates are needed for a grouped split.")
    train_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        .split(ids, groups=groups)
    )
    train_ids, val_ids = ids[train_idx].tolist(), ids[val_idx].tolist()
    train_physical, val_physical = set(physical[train_idx]), set(physical[val_idx])
    train_dates, val_dates = set(groups[train_idx]), set(groups[val_idx])
    assert train_physical.isdisjoint(val_physical), "Physical-observation leakage detected"
    assert train_dates.isdisjoint(val_dates), "Observation-date leakage detected"
    print(f"COCO annotation-image entries={len(ids)}, physical observations={len(set(physical))}, "
          f"unique dates={len(set(groups))}, training annotation entries={len(train_ids)}, "
          f"validation annotation entries={len(val_ids)}, training physical observations={len(train_physical)}, "
          f"validation physical observations={len(val_physical)}, training dates={len(train_dates)}, "
          f"validation dates={len(val_dates)}, "
          f"training filament instances={sum(len(index.annotations_by_image.get(i, [])) for i in train_ids)}, "
          f"validation filament instances={sum(len(index.annotations_by_image.get(i, [])) for i in val_ids)}")
    return train_ids, val_ids


def measure_foreground_prevalence(index: CocoIndex, image_ids: Sequence[str]) -> dict[str, float]:
    """Measure imbalance from decoded training masks; never use a guessed BCE weight."""
    foreground = 0
    pixels = 0
    for image_id in image_ids:
        mask = build_semantic_mask(index, image_id)
        foreground += int(mask.sum())
        pixels += int(mask.size)
    ratio = foreground / max(pixels, 1)
    return {"foreground_pixels": float(foreground), "total_pixels": float(pixels), "foreground_ratio": ratio,
            "background_to_foreground": (pixels - foreground) / max(foreground, 1)}


def robust_normalize(image: np.ndarray) -> np.ndarray:
    """Normalize grayscale image data and guarantee OpenCV-compatible float32 output."""
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, (1, 99))
    scale = np.float32(max(float(high) - float(low), 1e-6))
    normalized = (image - np.float32(low)) / scale
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)

def preprocess_flattened_uint8(image: np.ndarray) -> tuple[np.ndarray, Disk]:
    """Return cached-style limb-corrected pixels and their fitted solar geometry."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a single-channel H-alpha image, got shape {image.shape}")
    disk = detect_disk(image)
    flat = flatten(image, disk)
    packed = np.clip(flat, 0.0, 2.0) * (255.0 / 2.0)
    return np.rint(packed).astype(np.uint8), disk


def detect_solar_disk(image: np.ndarray) -> np.ndarray:
    """Mask the fitted solar disk, trimming the noisy outer limb conservatively."""
    disk = detect_disk(np.asarray(image))
    return disk.mask(image.shape, shrink=0.985)


def boundary_target(instance_ids: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(instance_ids, dtype=np.uint8)
    for label in np.unique(instance_ids):
        if label:
            binary = (instance_ids == label).astype(np.uint8)
            boundary |= binary - cv2.erode(binary, np.ones((3, 3), np.uint8))
    return boundary


class SolarFilamentDataset(Dataset[dict[str, Any]]):
    """Crop sampler with full-image normalization and disk-cached instance masks.

    COCO annotation entries stay independent. Repeated annotations for one JPEG
    share the same physical pixels but never share their label maps.
    """

    def __init__(self, index: CocoIndex, image_dir: Path, image_ids: Sequence[str],
                 config: PipelineConfig, training: bool) -> None:
        self.index = index
        self.image_dir = Path(image_dir)
        self.image_ids = list(image_ids)
        self.config = config
        self.training = training
        total = config.positive_probability + config.hard_negative_probability + config.random_probability
        if training and not np.isclose(total, 1.0):
            raise ValueError("Crop sampling probabilities must sum to one")
        self._sampling_counts: dict[str, int] = defaultdict(int)
        self.positive_centers: dict[str, np.ndarray] = {}
        self.hard_negative_centers: dict[str, np.ndarray] = {}
        self.mask_cache_dir = Path(config.mask_cache_dir)
        self.mask_cache_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir = self.mask_cache_dir / "preprocessed_limb_v1"
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        if config.in_channels not in (1, 2):
            raise ValueError("in_channels must be 1 or 2")
        if training:
            self._build_crop_candidates()

    def __len__(self) -> int:
        if not self.training:
            return len(self.image_ids)
        if self.config.samples_per_epoch is not None:
            return int(self.config.samples_per_epoch)
        return max(1, len(self.image_ids) * 4)

    @lru_cache(maxsize=4)
    def _load_image(self, file_name: str) -> np.ndarray:
        path = self.image_dir / file_name
        with Image.open(path) as handle:
            image = np.asarray(handle.convert("L"))
        if image.shape != (self.config.image_size, self.config.image_size):
            raise ValueError(f"Unexpected image size for {path}: {image.shape}")
        return image

    @lru_cache(maxsize=16)
    def _load_preprocessed(self, image_id: str) -> tuple[np.ndarray, Disk]:
        """Load or create limb-darkening-flattened pixels and solar disk geometry.

        Cache by physical filename rather than annotation ID, because multiple
        annotators label the same JPEG. The cache stores only image-derived
        information; it never stores or reads test ground-truth labels.
        """
        meta = self.index.images[image_id]
        stem = Path(str(meta["file_name"])).stem
        flat_path = self.preprocessed_dir / f"{stem}.png"
        geometry_path = self.preprocessed_dir / f"{stem}.json"

        if flat_path.is_file() and geometry_path.is_file():
            flat_u8 = cv2.imread(str(flat_path), cv2.IMREAD_GRAYSCALE)
            try:
                geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
                if flat_u8 is not None and flat_u8.shape == (self.config.image_size, self.config.image_size):
                    return flat_u8, Disk(
                        cx=float(geometry["cx"]),
                        cy=float(geometry["cy"]),
                        radius=float(geometry["radius"]),
                    )
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass

        raw = self._load_image(str(meta["file_name"]))
        flat_u8, disk = preprocess_flattened_uint8(raw)
        if not cv2.imwrite(str(flat_path), flat_u8):
            raise OSError(f"Could not write preprocessed image cache: {flat_path}")
        geometry_path.write_text(
            json.dumps({"cx": disk.cx, "cy": disk.cy, "radius": disk.radius}),
            encoding="utf-8",
        )
        return flat_u8, disk

    def _mask_cache_path(self, image_id: str) -> Path:
        meta = self.index.images[image_id]
        records = self.index.annotations_by_image.get(image_id, [])
        # Include annotation IDs and file identity so changed label sets don't
        # silently reuse a stale mask. Cache version is also encoded in dirname.
        token = "|".join([
            image_id,
            Path(str(meta["file_name"])).name,
            *[record.annotation_id for record in records],
        ])
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        return self.mask_cache_dir / f"{digest}.npz"

    @lru_cache(maxsize=4)
    def _load_instance_ids(self, image_id: str) -> np.ndarray:
        path = self._mask_cache_path(image_id)
        if path.exists():
            with np.load(path, allow_pickle=False) as stored:
                return stored["instance_ids"].astype(np.int32, copy=False)

        labels = build_instance_id_mask(self.index, image_id)
        # Filament count per image is far below uint16 limits. Compressing the
        # sparse label map dramatically reduces the on-disk cache footprint.
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, instance_ids=labels.astype(np.uint16))
        temporary.replace(path)
        return labels

    @lru_cache(maxsize=4)
    def _load_boundary(self, image_id: str) -> np.ndarray:
        return boundary_target(self._load_instance_ids(image_id)).astype(np.uint8)

    def _build_crop_candidates(self) -> None:
        """Build positive and hard-negative crop centers once per annotation view."""
        gradient_by_stem: dict[str, np.ndarray] = {}
        for image_id in self.image_ids:
            meta = self.index.images[image_id]
            stem = Path(str(meta["file_name"])).stem
            flat_u8, _ = self._load_preprocessed(image_id)
            labels = self._load_instance_ids(image_id)
            positive = np.argwhere(labels > 0)
            if len(positive) > 4096:
                positive = positive[::max(1, len(positive) // 4096)]
            self.positive_centers[image_id] = positive.astype(np.int32, copy=False)

            # Use flattened intensity for the hard-negative texture map. Explicit
            # float32 input avoids OpenCV CV_32F depth/channel assertion failures.
            gradient = gradient_by_stem.get(stem)
            if gradient is None:
                normalized = np.ascontiguousarray(flat_u8.astype(np.float32) / 255.0)
                gradient = np.abs(cv2.Laplacian(normalized, cv2.CV_32F))
                gradient_by_stem[stem] = gradient
            stride = max(32, self.config.train_tile_size // 4)
            half = self.config.train_tile_size // 2
            candidates: list[tuple[int, int]] = []
            gradient_threshold = float(np.percentile(gradient, 75))
            for y in range(half, max(half + 1, flat_u8.shape[0] - half), stride):
                for x in range(half, max(half + 1, flat_u8.shape[1] - half), stride):
                    crop = labels[y-half:y+half, x-half:x+half]
                    if crop.size and not crop.any() and gradient[y, x] >= gradient_threshold:
                        candidates.append((y, x))
            self.hard_negative_centers[image_id] = np.asarray(
                candidates, dtype=np.int32
            ).reshape(-1, 2)

    def sampling_distribution(self) -> dict[str, float]:
        total = sum(self._sampling_counts.values())
        return {name: count / total for name, count in self._sampling_counts.items()} if total else {}

    def __getitem__(self, item: int) -> dict[str, Any]:
        if not self.image_ids:
            raise IndexError("Dataset has no image IDs")
        image_id = (
            self.image_ids[random.randrange(len(self.image_ids))]
            if self.training else self.image_ids[item]
        )
        flat_u8, disk = self._load_preprocessed(image_id)
        # Channel 0 is limb-darkening-corrected intensity; channel 1 carries
        # normalized distance from the fitted solar limb.
        image = np.ascontiguousarray(flat_u8.astype(np.float32) / 255.0)
        radial = np.ascontiguousarray(disk.radius_map(image.shape).astype(np.float32))
        instance_ids = self._load_instance_ids(image_id)
        boundary = self._load_boundary(image_id)

        if self.training:
            image, instance_ids, boundary, radial = self._sample_crop(
                image, instance_ids, boundary, radial, image_id
            )
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                instance_ids = np.fliplr(instance_ids).copy()
                boundary = np.fliplr(boundary).copy()
                radial = np.fliplr(radial).copy()
            if random.random() < 0.5:
                image = np.flipud(image).copy()
                instance_ids = np.flipud(instance_ids).copy()
                boundary = np.flipud(boundary).copy()
                radial = np.flipud(radial).copy()
            k = random.randrange(4)
            image = np.rot90(image, k).copy()
            instance_ids = np.rot90(instance_ids, k).copy()
            boundary = np.rot90(boundary, k).copy()
            radial = np.rot90(radial, k).copy()
            # Mild intensity-only jitter helps across observing sites. The
            # radial coordinate and labels remain geometrically exact.
            if random.random() < 0.5:
                gain = random.uniform(0.90, 1.10)
                offset = random.uniform(-0.04, 0.04)
                image = np.clip(image * gain + offset, 0.0, 1.0).astype(np.float32)

        channels = (
            np.stack([image, radial], axis=0)
            if self.config.in_channels == 2 else image[None]
        )
        image_tensor = torch.from_numpy(np.ascontiguousarray(channels.astype(np.float32)))
        semantic = (instance_ids > 0).astype(np.float32)
        return {
            "image": image_tensor,
            "semantic": torch.from_numpy(np.ascontiguousarray(semantic[None])),
            "boundary": torch.from_numpy(np.ascontiguousarray(boundary[None].astype(np.float32))),
            "instance_ids": torch.from_numpy(np.ascontiguousarray(instance_ids.astype(np.int32))),
            "image_id": image_id,
            "sample_index": torch.tensor(item, dtype=torch.int64),
        }

    def _sample_crop(
        self, image: np.ndarray, labels: np.ndarray, boundary: np.ndarray,
        radial: np.ndarray, image_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        size = self.config.train_tile_size
        pad_h = max(0, size - image.shape[0])
        pad_w = max(0, size - image.shape[1])
        padding = ((0, pad_h), (0, pad_w))
        if pad_h or pad_w:
            image = np.pad(image, padding, constant_values=0)
            labels = np.pad(labels, padding, constant_values=0)
            boundary = np.pad(boundary, padding, constant_values=0)
            radial = np.pad(radial, padding, constant_values=2.0)

        draw = random.random()
        positive = self.positive_centers.get(image_id, np.argwhere(labels > 0))
        hard_negative = self.hard_negative_centers.get(
            image_id, np.empty((0, 2), dtype=np.int32)
        )

        if len(positive) and draw < self.config.positive_probability:
            self._sampling_counts["positive"] += 1
            y, x = positive[random.randrange(len(positive))]
            top, left = int(y) - random.randrange(size), int(x) - random.randrange(size)
        elif len(hard_negative) and draw < (
            self.config.positive_probability + self.config.hard_negative_probability
        ):
            self._sampling_counts["hard_negative"] += 1
            y, x = hard_negative[random.randrange(len(hard_negative))]
            top, left = int(y) - size // 2, int(x) - size // 2
        else:
            self._sampling_counts["random"] += 1
            top = random.randrange(image.shape[0] - size + 1)
            left = random.randrange(image.shape[1] - size + 1)

        top = int(np.clip(top, 0, image.shape[0] - size))
        left = int(np.clip(left, 0, image.shape[1] - size))
        crop = np.s_[top:top + size, left:left + size]
        return (
            image[crop].copy(),
            labels[crop].copy(),
            boundary[crop].copy(),
            radial[crop].copy(),
        )



class _BasicResidualBlock(nn.Module):
    """Small torchvision-independent ResNet basic block used only as a fallback."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class _FallbackResNet34(nn.Module):
    """ResNet-34 shape-compatible fallback; no pretrained weights are claimed."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self._in_channels = 64
        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)

    def _make_layer(self, channels: int, count: int, stride: int) -> nn.Sequential:
        blocks = [_BasicResidualBlock(self._in_channels, channels, stride)]
        self._in_channels = channels
        blocks.extend(_BasicResidualBlock(channels, channels) for _ in range(count - 1))
        return nn.Sequential(*blocks)


def build_encoder(pretrained: bool = True) -> tuple[nn.Module, bool]:
    """Use cached ImageNet ResNet-34 weights when possible.

    Torchvision is imported lazily: if the Kaggle image has a binary
    torch/torchvision mismatch, the rest of this module remains importable and
    a shape-compatible pure-PyTorch encoder is used with an explicit warning.
    """
    if not pretrained:
        print("Pretrained encoder loaded: NO (disabled by configuration)")
        return _FallbackResNet34(), False
    try:
        from torchvision.models import ResNet34_Weights, resnet34
        encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        print("Pretrained encoder loaded: YES")
        return encoder, True
    except Exception as error:
        LOGGER.warning(
            "Could not load torchvision/ImageNet ResNet-34 weights; using a local "
            "randomly initialized ResNet-34 fallback. Torch remains installed as provided. Error: %s",
            error,
        )
        print("Pretrained encoder loaded: NO (torchvision unavailable; random fallback)")
        return _FallbackResNet34(), False


class FilamentSegmentationModel(nn.Module):
    """ResNet-34 U-Net-style decoder with semantic and optional boundary heads."""

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__()
        encoder, self.pretrained_loaded = build_encoder(config.pretrained_encoder)
        if config.in_channels != 3:
            old = encoder.conv1
            encoder.conv1 = nn.Conv2d(config.in_channels, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                # Preserve the RGB-pretrained filters for intensity. Start the
                # geometry/radius channel at zero so it cannot swamp that signal.
                mean_weight = old.weight.mean(1, keepdim=True)
                encoder.conv1.weight[:, :1].copy_(mean_weight)
                if config.in_channels > 1:
                    encoder.conv1.weight[:, 1:].zero_()
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool, self.layer1, self.layer2 = encoder.maxpool, encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.decode3 = self._block(512 + 256, 256)
        self.decode2 = self._block(256 + 128, 128)
        self.decode1 = self._block(128 + 64, 64)
        self.decode0 = self._block(64 + 64, 64)
        self.semantic_head, self.boundary_head = nn.Conv2d(64, 1, 1), nn.Conv2d(64, 1, 1)
        self.use_boundary = config.boundary_head

    @staticmethod
    def _block(inputs: int, outputs: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
                             nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))

    @staticmethod
    def _up(x: Tensor, skip: Tensor) -> Tensor:
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        s = self.stem(x); x1 = self.layer1(self.pool(s)); x2 = self.layer2(x1); x3 = self.layer3(x2); x4 = self.layer4(x3)
        d3 = self.decode3(self._up(x4, x3)); d2 = self.decode2(self._up(d3, x2)); d1 = self.decode1(self._up(d2, x1)); d0 = self.decode0(self._up(d1, s))
        d0 = F.interpolate(d0, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"semantic": self.semantic_head(d0), "boundary": self.boundary_head(d0)}


def soft_dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    prob = logits.sigmoid(); numerator = 2 * (prob * target).sum((1, 2, 3)) + 1
    return 1 - (numerator / (prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


def tversky_loss(logits: Tensor, target: Tensor, alpha: float = .3, beta: float = .7) -> Tensor:
    prob = logits.sigmoid(); tp = (prob * target).sum(); fp = (prob * (1 - target)).sum(); fn = ((1 - prob) * target).sum()
    return 1 - (tp + 1) / (tp + alpha * fp + beta * fn + 1)


def compound_loss(output: dict[str, Tensor], batch: dict[str, Tensor], config: PipelineConfig, pos_weight: float) -> Tensor:
    w = config.loss_weights
    semantic = output["semantic"]
    loss = w["bce"] * F.binary_cross_entropy_with_logits(semantic, batch["semantic"], pos_weight=torch.tensor(pos_weight, device=semantic.device))
    loss += w["dice"] * soft_dice_loss(semantic, batch["semantic"]) + w["tversky"] * tversky_loss(semantic, batch["semantic"])
    if config.boundary_head:
        loss += w["boundary"] * F.binary_cross_entropy_with_logits(output["boundary"], batch["boundary"])
    return loss


def compute_segmentation_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp, fp, fn = np.logical_and(pred, truth).sum(), np.logical_and(pred, ~truth).sum(), np.logical_and(~pred, truth).sum()
    return {"dice": float((2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)), "iou": float((tp + 1e-7) / (tp + fp + fn + 1e-7)),
            "precision": float((tp + 1e-7) / (tp + fp + 1e-7)), "recall": float((tp + 1e-7) / (tp + fn + 1e-7))}


def compute_instance_iou_matrix(pred_ids: np.ndarray, truth_ids: np.ndarray) -> np.ndarray:
    """Vectorized pair histogram; scans the image once rather than per object pair."""
    if pred_ids.shape != truth_ids.shape:
        raise ValueError("Prediction and ground-truth label maps must have identical shapes")
    # Flatten explicitly: NumPy 2.x may preserve the input shape in return_inverse,
    # but np.bincount requires a one-dimensional index array.
    gt_labels, gt_inverse = np.unique(np.asarray(truth_ids).ravel(), return_inverse=True)
    pred_labels, pred_inverse = np.unique(np.asarray(pred_ids).ravel(), return_inverse=True)
    # Labels are compacted locally, so arbitrary COCO/source labels are safe.
    overlap = np.bincount(
        gt_inverse * len(pred_labels) + pred_inverse,
        minlength=len(gt_labels) * len(pred_labels),
    ).reshape(len(gt_labels), len(pred_labels))
    gt_foreground = gt_labels > 0
    pred_foreground = pred_labels > 0
    intersections = overlap[np.ix_(gt_foreground, pred_foreground)].astype(np.float64)
    gt_area = overlap[gt_foreground, :].sum(axis=1, dtype=np.float64)[:, None]
    pred_area = overlap[:, pred_foreground].sum(axis=0, dtype=np.float64)[None, :]
    union = gt_area + pred_area - intersections
    return np.divide(intersections, union, out=np.zeros_like(intersections), where=union > 0)


def benchmark_instance_iou(shape: tuple[int, int] = (2048, 2048), instances: int = 24) -> dict[str, float]:
    """Compare the vectorized pair histogram against the former pairwise reference."""
    rng = np.random.default_rng(123)
    truth = rng.integers(0, instances + 1, shape, dtype=np.int16)
    prediction = rng.integers(0, instances + 1, shape, dtype=np.int16)
    started = time.perf_counter(); fast = compute_instance_iou_matrix(prediction, truth); fast_seconds = time.perf_counter() - started
    started = time.perf_counter()
    slow = np.zeros_like(fast)
    for gt in range(1, instances + 1):
        for pred in range(1, instances + 1):
            intersection = np.count_nonzero((truth == gt) & (prediction == pred))
            union = np.count_nonzero((truth == gt) | (prediction == pred))
            slow[gt - 1, pred - 1] = intersection / union if union else 0
    slow_seconds = time.perf_counter() - started
    assert np.allclose(fast, slow)
    return {"vectorized_seconds": fast_seconds, "pairwise_seconds": slow_seconds, "speedup": slow_seconds / max(fast_seconds, 1e-12)}


def match_instances(iou: np.ndarray, threshold: float = .5) -> list[tuple[int, int, float]]:
    """Greedy maximum-IoU matching is exact at IoU > .5 for non-overlapping masks."""
    matches: list[tuple[int, int, float]] = []
    for flat in np.argsort(iou.ravel())[::-1]:
        g, p = np.unravel_index(flat, iou.shape)
        if iou[g, p] <= threshold: break
        if g not in [x[0] for x in matches] and p not in [x[1] for x in matches]: matches.append((g, p, float(iou[g, p])))
    return matches


def compute_pq(pred_ids: np.ndarray, truth_ids: np.ndarray, threshold: float = .5) -> dict[str, float]:
    iou = compute_instance_iou_matrix(pred_ids, truth_ids); matches = match_instances(iou, threshold)
    gt_count, pred_count, tp = iou.shape[0], iou.shape[1], len(matches)
    fp, fn = pred_count - tp, gt_count - tp
    sq = sum(m[2] for m in matches) / tp if tp else 0.0
    rq = tp / (tp + .5 * fp + .5 * fn) if tp + fp + fn else 1.0
    # Split/merge diagnostics count a relation only when it clears a modest overlap.
    one_to_many = int(sum((row >= .1).sum() > 1 for row in iou))
    many_to_one = int(sum((iou[:, col] >= .1).sum() > 1 for col in range(iou.shape[1])))
    return {"pq": sq * rq, "sq": sq, "rq": rq, "matched_iou_sum": sum(m[2] for m in matches), "tp": tp, "fp": fp, "fn": fn,
            "one_to_many": one_to_many, "many_to_one": many_to_one}


def positive_instance_labels(labels: np.ndarray) -> np.ndarray:
    """Return all positive instance IDs, even if label 0 is absent."""
    values = np.unique(labels)
    return values[values > 0]


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Compact surviving instance IDs while preserving watershed separations."""
    result = np.zeros_like(labels, dtype=np.int32)
    unique = np.unique(labels)
    for new, old in enumerate(unique[unique > 0], start=1):
        result[labels == old] = new
    return result


def reconstruct_instances(probability: np.ndarray, config: PipelineConfig, boundary_probability: np.ndarray | None = None,
                          disk_mask: np.ndarray | None = None) -> np.ndarray:
    binary = probability >= config.threshold
    if disk_mask is not None and config.use_solar_disk_mask:
        binary &= disk_mask
    # Boundary prediction guides marker placement or optional separator bands; it never erases all boundary pixels.
    separator = None
    if boundary_probability is not None and config.boundary_mode == "separator":
        separator = boundary_probability >= config.boundary_threshold
    if config.morphology_kernel > 1:
        kernel = np.ones((config.morphology_kernel, config.morphology_kernel), np.uint8)
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    if config.watershed and binary.any():
        distance = ndi.distance_transform_edt(binary)
        if boundary_probability is not None and config.boundary_mode == "guidance":
            distance = distance * (1.0 - 0.5 * boundary_probability)
        watershed_mask = binary & ~separator if separator is not None else binary
        # Seeds must lie inside the actual watershed domain. In separator mode,
        # finding peaks against the uncut binary mask can place seeds on removed
        # boundary pixels and produce unstable marker assignments.
        peaks = peak_local_max(
            distance,
            min_distance=config.watershed_peak_distance,
            labels=watershed_mask,
        )
        markers = np.zeros(binary.shape, dtype=np.int32)
        for label, (y, x) in enumerate(peaks, 1): markers[y, x] = label
        markers, _ = ndi.label(markers)
        labels = watershed(-distance, markers, mask=watershed_mask) if markers.max() else ndi.label(binary)[0]
    else:
        labels, _ = ndi.label(binary)
    for label in positive_instance_labels(labels):
        if (labels == label).sum() < config.min_component_area: labels[labels == label] = 0
    return relabel_sequential(labels)


def tile_starts(length: int, tile: int, overlap: float) -> list[int]:
    stride = max(1, int(tile * (1 - overlap))); starts = list(range(0, max(length - tile, 0) + 1, stride))
    return starts if starts and starts[-1] == max(length - tile, 0) else starts + [max(length - tile, 0)]


@torch.inference_mode()
def predict_tiled(
    model: nn.Module, image: np.ndarray, config: PipelineConfig, device: torch.device
) -> tuple[np.ndarray, np.ndarray | None]:
    """Limb-corrected, radial-aware tiled inference with optional dihedral TTA."""
    image = np.asarray(image)
    h, w = image.shape
    tile = config.train_tile_size
    flat_u8, disk = preprocess_flattened_uint8(image)
    intensity = flat_u8.astype(np.float32) / 255.0
    if config.in_channels == 2:
        image_channels = np.stack(
            [intensity, disk.radius_map(image.shape).astype(np.float32)], axis=0
        )
    else:
        image_channels = intensity[None]
    image_channels = np.ascontiguousarray(image_channels, dtype=np.float32)

    sums = np.zeros((h, w), np.float32)
    counts = np.zeros((h, w), np.float32)
    boundary_sums = np.zeros((h, w), np.float32)
    window_1d = np.hanning(tile).astype(np.float32)
    blend_window = (
        np.outer(window_1d, window_1d) + 0.05
        if config.weighted_tile_blending else np.ones((tile, tile), np.float32)
    )

    if not config.use_tta:
        transforms = [(0, False)]
    elif config.tta_mode == "light":
        transforms = [(0, False), (0, True), (2, False), (2, True)]
    elif config.tta_mode == "dihedral":
        transforms = [(k, flip) for k in range(4) for flip in (False, True)]
    else:
        raise ValueError(f"Unknown TTA mode: {config.tta_mode!r}")

    pending: list[tuple[np.ndarray, int, int, int, int, int, bool]] = []

    def flush() -> None:
        if not pending:
            return
        batch_array = np.stack([entry[0] for entry in pending]).astype(np.float32, copy=False)
        batch = torch.from_numpy(np.ascontiguousarray(batch_array)).to(device)
        output = model(batch)
        semantic = output["semantic"].sigmoid().cpu().numpy()[:, 0]
        boundary = output["boundary"].sigmoid().cpu().numpy()[:, 0]
        for prediction, boundary_prediction, (_, y, x, oh, ow, k, flip_lr) in zip(
            semantic, boundary, pending
        ):
            # Input was flip(rot90(tile)); invert that transform before blending.
            if flip_lr:
                prediction = np.flip(prediction, axis=-1)
                boundary_prediction = np.flip(boundary_prediction, axis=-1)
            if k:
                prediction = np.rot90(prediction, -k)
                boundary_prediction = np.rot90(boundary_prediction, -k)
            weight = blend_window[:oh, :ow]
            sums[y:y + oh, x:x + ow] += prediction[:oh, :ow] * weight
            boundary_sums[y:y + oh, x:x + ow] += boundary_prediction[:oh, :ow] * weight
            counts[y:y + oh, x:x + ow] += weight
        pending.clear()

    model.eval()
    for y in tile_starts(h, tile, config.tile_overlap):
        for x in tile_starts(w, tile, config.tile_overlap):
            crop = image_channels[:, y:y + tile, x:x + tile]
            original_h, original_w = crop.shape[-2:]
            crop = np.pad(
                crop,
                ((0, 0), (0, tile - original_h), (0, tile - original_w)),
                mode="constant",
            )
            for k, flip_lr in transforms:
                transformed = np.rot90(crop, k, axes=(-2, -1))
                if flip_lr:
                    transformed = np.flip(transformed, axis=-1)
                transformed = np.ascontiguousarray(transformed, dtype=np.float32)
                pending.append((transformed, y, x, original_h, original_w, k, flip_lr))
                if len(pending) >= config.inference_batch_size:
                    flush()
    flush()
    return sums / np.maximum(counts, 1e-8), (
        boundary_sums / np.maximum(counts, 1e-8) if config.boundary_head else None
    )


def mask_to_rle(mask: np.ndarray) -> str:
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def validate_rle(mask: np.ndarray) -> None:
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    assert np.array_equal(coco_mask.decode(encoded).astype(bool), mask.astype(bool)), "RLE round trip failed"


def create_submission(model: nn.Module, data: CompetitionData, config: PipelineConfig, output_path: str | Path,
                      device: torch.device) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    per_image: list[int] = []
    empty_observations: list[str] = []
    for path in list_image_files(data.test_images):
        image = np.asarray(Image.open(path).convert("L")); probability, boundary = predict_tiled(model, image, config, device)
        labels = reconstruct_instances(probability, config, boundary, detect_solar_disk(image))
        count = 0
        for label in positive_instance_labels(labels):
            mask = labels == label
            assert mask.shape == image.shape, "Predicted mask dimensions changed during inference"
            validate_rle(mask)
            rows.append({"filament_id": f"{path.stem}_{label}", "segmentation_rle": mask_to_rle(mask)})
            count += 1
        per_image.append(count)
        if count == 0:
            empty_observations.append(path.stem)
    submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    assert submission.filament_id.is_unique and not submission.isna().any().any()
    submission.to_csv(output_path, index=False)
    print({"test_observations": len(per_image), "submission_rows": len(submission), "predicted_instances": sum(per_image),
           "average_instances_image": float(np.mean(per_image)) if per_image else 0, "min_instances_image": min(per_image, default=0), "max_instances_image": max(per_image, default=0),
           "empty_detection_observations": empty_observations})
    return submission


def run_synthetic_metric_tests() -> None:
    truth = np.zeros((10, 10), np.int32); truth[1:4, 1:4] = 1; truth[6:9, 6:9] = 2
    perfect = truth.copy(); result = compute_pq(perfect, truth)
    assert result["pq"] == 1 and result["tp"] == 2 and result["fp"] == result["fn"] == 0
    fp = truth.copy(); fp[1:3, 6:8] = 3
    assert compute_pq(fp, truth)["fp"] == 1
    fn = truth.copy(); fn[6:9, 6:9] = 0
    assert compute_pq(fn, truth)["fn"] == 1
    merged = np.where(truth > 0, 1, 0); result = compute_pq(merged, truth)
    assert result["many_to_one"] == 1 and result["pq"] < 1
    split = truth.copy(); split[1:4, 2] = 0; labels, _ = ndi.label(split > 0); result = compute_pq(labels, truth)
    assert result["one_to_many"] >= 1 and result["pq"] < 1
    assert not match_instances(np.array([[.5]]))
    assert match_instances(np.array([[.500001]]))


def environment_report(config: PipelineConfig) -> dict[str, Any]:
    return {"config": asdict(config), "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}


def build_grad_scaler(device: torch.device, enabled: bool) -> Any:
    """Support both current torch.amp and older torch.cuda.amp Kaggle images."""
    active = enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=active)
    return torch.cuda.amp.GradScaler(enabled=active)


def amp_autocast(device: torch.device, enabled: bool) -> Any:
    active = enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=active)
    return torch.cuda.amp.autocast(enabled=active)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: Any,
                    config: PipelineConfig, device: torch.device, pos_weight: float) -> float:
    """One AMP-enabled epoch with accumulation and clipping for large tiles."""
    model.train(); optimizer.zero_grad(set_to_none=True); losses: list[float] = []
    for step, batch in enumerate(loader, 1):
        batch = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}
        with amp_autocast(device, config.use_amp):
            loss = compound_loss(model(batch["image"]), batch, config, pos_weight) / config.gradient_accumulation
        scaler.scale(loss).backward()
        if step % config.gradient_accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()) * config.gradient_accumulation)
    return float(np.mean(losses))


@torch.inference_mode()
def validate(model: nn.Module, dataset: SolarFilamentDataset, config: PipelineConfig, device: torch.device) -> dict[str, float]:
    """Score every annotator view while running the model once per physical JPEG."""
    aggregate: dict[str, list[float]] = defaultdict(list)
    totals: dict[str, float] = defaultdict(float)
    predictions_by_stem: dict[str, np.ndarray] = {}
    for image_id in dataset.image_ids:
        meta = dataset.index.images[image_id]
        stem = Path(str(meta["file_name"])).stem
        predicted = predictions_by_stem.get(stem)
        if predicted is None:
            image = np.asarray(Image.open(dataset.image_dir / meta["file_name"]).convert("L"))
            probability, boundary = predict_tiled(model, image, config, device)
            disk_mask = detect_solar_disk(image)
            predicted = reconstruct_instances(probability, config, boundary, disk_mask)
            predictions_by_stem[stem] = predicted
        truth = build_instance_id_mask(dataset.index, image_id)
        semantic = compute_segmentation_metrics(predicted > 0, truth > 0)
        panoptic = compute_pq(predicted, truth)
        for key, value in {**semantic, **panoptic}.items():
            aggregate[key].append(float(value)); totals[key] += float(value)
    result = {key: float(np.mean(values)) for key, values in aggregate.items()}
    # Dataset-global PQ is recomputed from summed matching components, unlike a mean per-image PQ.
    denom = totals["tp"] + .5 * totals["fp"] + .5 * totals["fn"]
    result["global_sq"] = totals["matched_iou_sum"] / max(totals["tp"], 1)
    result["global_rq"] = totals["tp"] / denom if denom else 1.0
    result["global_pq"] = result["global_sq"] * result["global_rq"]
    # Explicit aliases prevent accidentally comparing mean image PQ with global dataset PQ.
    result["mean_image_pq"] = result["pq"]
    result["mean_dice"] = result["dice"]
    result["mean_iou"] = result["iou"]
    return result


@torch.inference_mode()
def cache_validation_predictions(model: nn.Module, dataset: SolarFilamentDataset, config: PipelineConfig,
                                 device: torch.device, cache_dir: str | Path) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]]:
    """Persist full-resolution validation outputs and disk masks so post-processing matches final inference."""
    destination = Path(cache_dir); destination.mkdir(parents=True, exist_ok=True)
    cached: list[tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]] = []
    predictions_by_stem: dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray]] = {}
    for image_id in dataset.image_ids:
        path = destination / f"{image_id}.npz"
        meta = dataset.index.images[image_id]
        stem = Path(str(meta["file_name"])).stem
        if path.exists():
            with np.load(path, allow_pickle=False) as loaded:
                semantic = loaded["semantic"]
                boundary = loaded["boundary"] if "boundary" in loaded.files else None
                truth = loaded["truth"]
                disk = loaded["disk"] if "disk" in loaded.files else np.ones(truth.shape, dtype=bool)
            predictions_by_stem.setdefault(stem, (semantic, boundary, disk))
            cached.append((semantic, boundary, truth, disk))
            continue

        if stem in predictions_by_stem:
            # Multiple annotator entries can point to the exact same JPEG. Reuse
            # the prediction arrays; each view still receives its own GT mask.
            semantic, boundary, disk = predictions_by_stem[stem]
        else:
            image = np.asarray(Image.open(dataset.image_dir / meta["file_name"]).convert("L"))
            semantic, boundary = predict_tiled(model, image, config, device)
            disk = detect_solar_disk(image)
            predictions_by_stem[stem] = (semantic, boundary, disk)

        truth = build_instance_id_mask(dataset.index, image_id)
        payload: dict[str, np.ndarray] = {
            "semantic": semantic.astype(np.float32),
            "truth": truth.astype(np.int32),
            "disk": disk.astype(np.uint8),
        }
        if boundary is not None:
            payload["boundary"] = boundary.astype(np.float32)
        np.savez_compressed(path, **payload)
        cached.append((semantic, boundary, truth, disk))
    return cached


def optimize_postprocessing(probabilities: Iterable[np.ndarray], boundaries: Iterable[np.ndarray | None],
                            truths: Iterable[np.ndarray], base: PipelineConfig,
                            disk_masks: Iterable[np.ndarray] | None = None) -> tuple[PipelineConfig, pd.DataFrame]:
    """Staged PQ-first search over cached maps; every evaluated setting is retained."""
    probabilities = list(probabilities)
    boundaries = list(boundaries)
    truths = list(truths)
    if not (len(probabilities) == len(boundaries) == len(truths)):
        raise ValueError("Probability, boundary, and truth caches must have equal lengths")
    if disk_masks is None:
        disk_masks = [None] * len(truths)
    else:
        disk_masks = list(disk_masks)
    if len(disk_masks) != len(truths):
        raise ValueError("Disk-mask cache length does not match truth cache")
    cached = list(zip(probabilities, boundaries, truths, disk_masks))
    if not cached:
        raise ValueError("Cannot optimize post-processing with an empty validation cache")
    rows: list[dict[str, float]] = []
    def evaluate(stage: str, update: dict[str, Any]) -> PipelineConfig:
        candidate = PipelineConfig(**{**asdict(base), **update})
        panoptic = [compute_pq(reconstruct_instances(p, candidate, b, d), truth)
                    for p, b, truth, d in cached]
        semantic = [
            compute_segmentation_metrics(reconstruct_instances(p, candidate, b, d) > 0, truth > 0)
            for p, b, truth, d in cached
        ]
        # The organizer's PQ aggregates TP/FP/FN and matched IoU across annotation
        # views. Optimize that pooled objective; retain mean-image PQ as a diagnostic.
        total_tp = sum(float(x["tp"]) for x in panoptic)
        total_fp = sum(float(x["fp"]) for x in panoptic)
        total_fn = sum(float(x["fn"]) for x in panoptic)
        total_iou = sum(float(x["matched_iou_sum"]) for x in panoptic)
        denominator = total_tp + 0.5 * total_fp + 0.5 * total_fn
        pq_pooled = total_iou / denominator if denominator else 1.0
        rows.append({
            "stage": stage,
            **update,
            "pq_pooled": float(pq_pooled),
            "mean_pq": float(np.mean([x["pq"] for x in panoptic])),
            "mean_dice": float(np.mean([x["dice"] for x in semantic])),
            "mean_iou": float(np.mean([x["iou"] for x in semantic])),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "matched_iou_sum": total_iou,
            "one_to_many": int(sum(x["one_to_many"] for x in panoptic)),
            "many_to_one": int(sum(x["many_to_one"] for x in panoptic)),
        })
        return candidate
    # Stage 1 selects semantic threshold and whether to split. Later stages refine the winner only.
    stage1 = [evaluate("threshold_watershed", {"threshold": threshold, "watershed": watershed_enabled})
              for threshold in (.20, .25, .30, .35, .40, .45, .50, .55, .60, .65) for watershed_enabled in (False, True)]
    best = max(zip(stage1, rows), key=lambda item: (item[1]["pq_pooled"], item[1]["mean_dice"]))[0]
    stage2 = [evaluate("min_area", {**asdict(best), "min_component_area": area}) for area in (10, 20, 50, 100, 200, 500)]
    best = max(zip(stage2, rows[-len(stage2):]), key=lambda item: (item[1]["pq_pooled"], item[1]["mean_dice"]))[0]
    if best.watershed:
        stage3 = [evaluate("peak_distance", {**asdict(best), "watershed_peak_distance": distance})
                  for distance in (8, 12, 16, 20, 24, 32)]
        best = max(zip(stage3, rows[-len(stage3):]),
                   key=lambda item: (item[1]["pq_pooled"], item[1]["mean_dice"]))[0]
    stage4 = [evaluate("boundary", {**asdict(best), "boundary_mode": mode, "boundary_threshold": threshold})
              for mode in ("disabled", "guidance", "separator") for threshold in (.55, .65, .75)]
    best = max(zip(stage4, rows[-len(stage4):]), key=lambda item: (item[1]["pq_pooled"], item[1]["mean_dice"]))[0]
    stage5 = [evaluate("morphology", {**asdict(best), "morphology_kernel": kernel}) for kernel in (0, 1, 2, 3, 5)]
    best = max(zip(stage5, rows[-len(stage5):]), key=lambda item: (item[1]["pq_pooled"], item[1]["mean_dice"]))[0]
    stage6 = [evaluate("solar_disk_mask", {**asdict(best), "use_solar_disk_mask": enabled})
              for enabled in (False, True)]
    best = max(zip(stage6, rows[-len(stage6):]), key=lambda item: (item[1]["pq_pooled"], item[1]["mean_dice"]))[0]
    table = pd.DataFrame(rows).sort_values(["pq_pooled", "mean_dice", "mean_pq"], ascending=False).reset_index(drop=True)
    return best, table


def append_experiment(path: str | Path, name: str, config: PipelineConfig, metrics: dict[str, float], runtime_seconds: float,
                      pos_weight: float, split_summary: dict[str, int] | None = None) -> None:
    """Append measured results only; callers must never insert claimed scores."""
    row = {"experiment": name, "architecture": "resnet34_unet", "encoder": config.encoder,
           "pretrained": config.pretrained_encoder, "tile_size": config.train_tile_size, "epochs": config.epochs,
           "batch_size": config.batch_size, "learning_rate": config.learning_rate, "loss": json.dumps(config.loss_weights, sort_keys=True),
           "pos_weight": pos_weight, "threshold": config.threshold, "min_area": config.min_component_area,
           "watershed": config.watershed, "peak_distance": config.watershed_peak_distance, "boundary_mode": config.boundary_mode,
           "tta": config.use_tta, "runtime_seconds": runtime_seconds, **(split_summary or {}), **metrics}
    destination = Path(path); previous = pd.read_csv(destination) if destination.exists() else pd.DataFrame()
    pd.concat([previous, pd.DataFrame([row])], ignore_index=True).to_csv(destination, index=False)
