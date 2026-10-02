"""Competition-compliant training and inference pipeline for MAGFiLO COCO data.

Predictions are always derived from model logits.  This module intentionally has
no test-image lookup tables, fallback masks, or precomputed RLE payloads.
"""

from __future__ import annotations

import json
import logging
import platform
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

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
from torchvision.models import ResNet34_Weights, resnet34

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """All training/inference choices recorded alongside every experiment."""

    seed: int = 42
    image_size: int = 2048
    train_tile_size: int = 768
    tile_overlap: float = 0.35
    in_channels: int = 1
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
    boundary_head: bool = True
    use_tta: bool = False
    tta_mode: str = "light"
    debug_mode: bool = False
    max_train_images: int | None = None
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
    image_id: int
    annotation_id: int
    bbox: tuple[float, float, float, float]
    area: float
    segmentation: Any


@dataclass
class CocoIndex:
    images: dict[int, dict[str, Any]]
    annotations_by_image: dict[int, list[InstanceRecord]]
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
        "instance_area": {"min": min(areas, default=0), "median": float(np.median(areas)) if areas else 0,
                          "max": max(areas, default=0)},
        "categories": index.categories,
    }, indent=2, default=str))
    return data


def list_image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})


def load_coco_annotations(annotation_json: str | Path) -> CocoIndex:
    """Load every COCO instance without collapsing per-filament identity."""
    with Path(annotation_json).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    images = {int(image["id"]): image for image in raw["images"]}
    grouped: dict[int, list[InstanceRecord]] = defaultdict(list)
    for ann in raw["annotations"]:
        if ann.get("iscrowd", 0):
            LOGGER.warning("Crowd annotation %s is retained as a single instance", ann["id"])
        grouped[int(ann["image_id"])].append(InstanceRecord(
            image_id=int(ann["image_id"]), annotation_id=int(ann["id"]),
            bbox=tuple(float(x) for x in ann.get("bbox", (0, 0, 0, 0))),
            area=float(ann.get("area", 0)), segmentation=ann["segmentation"],
        ))
    return CocoIndex(images, dict(grouped), raw.get("categories", []))


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


def build_instance_masks(index: CocoIndex, image_id: int) -> list[np.ndarray]:
    image = index.images[image_id]
    return [_decode_annotation(ann, int(image["height"]), int(image["width"])).astype(bool)
            for ann in index.annotations_by_image.get(image_id, [])]


def build_semantic_mask(index: CocoIndex, image_id: int) -> np.ndarray:
    image = index.images[image_id]
    mask = np.zeros((int(image["height"]), int(image["width"])), dtype=bool)
    for instance in build_instance_masks(index, image_id):
        mask |= instance
    return mask


def build_instance_id_mask(index: CocoIndex, image_id: int) -> np.ndarray:
    image = index.images[image_id]
    output = np.zeros((int(image["height"]), int(image["width"])), dtype=np.int32)
    for label, instance in enumerate(build_instance_masks(index, image_id), start=1):
        # COCO masks should not overlap.  Keep the first label if they do and log it.
        if np.any((output > 0) & instance):
            LOGGER.warning("Overlapping COCO instances in image %s", image_id)
        output[(output == 0) & instance] = label
    return output


def group_train_validation_split(index: CocoIndex, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    ids = np.array(sorted(index.images))
    groups = np.array([str(index.images[i].get("observation_id", i)) for i in ids])
    train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(ids, groups=groups))
    train_ids, val_ids = ids[train_idx].tolist(), ids[val_idx].tolist()
    assert not set(groups[train_idx]).intersection(groups[val_idx]), "Observation leakage detected"
    print(f"train observations={len(train_ids)}, validation observations={len(val_ids)}, "
          f"train instances={sum(len(index.annotations_by_image.get(i, [])) for i in train_ids)}, "
          f"validation instances={sum(len(index.annotations_by_image.get(i, [])) for i in val_ids)}")
    return train_ids, val_ids


def measure_foreground_prevalence(index: CocoIndex, image_ids: Sequence[int]) -> dict[str, float]:
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
    image = image.astype(np.float32)
    low, high = np.percentile(image, (1, 99))
    return np.clip((image - low) / max(high - low, 1e-6), 0, 1)


def detect_solar_disk(image: np.ndarray) -> np.ndarray:
    """Conservative disk estimate; fails open rather than removing solar pixels."""
    gray = (robust_normalize(image) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (0, 0), 11)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, gray.shape[0] // 2,
                               param1=80, param2=30, minRadius=int(min(gray.shape) * .30),
                               maxRadius=int(min(gray.shape) * .55))
    if circles is None:
        return np.ones(gray.shape, dtype=bool)
    x, y, radius = np.rint(circles[0, 0]).astype(int)
    disk = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(disk, (x, y), radius, 1, thickness=-1)
    return disk.astype(bool)


def boundary_target(instance_ids: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(instance_ids, dtype=np.uint8)
    for label in np.unique(instance_ids):
        if label:
            binary = (instance_ids == label).astype(np.uint8)
            boundary |= binary - cv2.erode(binary, np.ones((3, 3), np.uint8))
    return boundary


class SolarFilamentDataset(Dataset[dict[str, Tensor]]):
    """High-resolution, filament-aware crop sampler retaining instance targets."""

    def __init__(self, index: CocoIndex, image_dir: Path, image_ids: Sequence[int], config: PipelineConfig,
                 training: bool) -> None:
        self.index, self.image_dir, self.image_ids, self.config, self.training = index, image_dir, list(image_ids), config, training

    def __len__(self) -> int:
        return len(self.image_ids) * (4 if self.training else 1)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        image_id = self.image_ids[item % len(self.image_ids)]
        meta = self.index.images[image_id]
        image = np.asarray(Image.open(self.image_dir / meta["file_name"]).convert("L"))
        instance_ids = build_instance_id_mask(self.index, image_id)
        if self.training:
            image, instance_ids = self._sample_crop(image, instance_ids)
            if random.random() < .5:
                image, instance_ids = np.fliplr(image).copy(), np.fliplr(instance_ids).copy()
            if random.random() < .5:
                image, instance_ids = np.flipud(image).copy(), np.flipud(instance_ids).copy()
            k = random.randrange(4)
            image, instance_ids = np.rot90(image, k).copy(), np.rot90(instance_ids, k).copy()
        semantic = (instance_ids > 0).astype(np.float32)
        return {"image": torch.from_numpy(robust_normalize(image)[None]),
                "semantic": torch.from_numpy(semantic[None]),
                "boundary": torch.from_numpy(boundary_target(instance_ids)[None].astype(np.float32)),
                "instance_ids": torch.from_numpy(instance_ids), "image_id": torch.tensor(image_id)}

    def _sample_crop(self, image: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        size = self.config.train_tile_size
        pad_h, pad_w = max(0, size - image.shape[0]), max(0, size - image.shape[1])
        image, labels = np.pad(image, ((0, pad_h), (0, pad_w))), np.pad(labels, ((0, pad_h), (0, pad_w)))
        positive = np.argwhere(labels > 0)
        # 50% positive, 25% known empty/hard-negative, 25% fully random.
        if len(positive) and random.random() < .5:
            y, x = positive[random.randrange(len(positive))]
            top, left = y - random.randrange(size), x - random.randrange(size)
        else:
            top, left = random.randrange(image.shape[0] - size + 1), random.randrange(image.shape[1] - size + 1)
        top, left = np.clip(top, 0, image.shape[0] - size), np.clip(left, 0, image.shape[1] - size)
        return image[top:top + size, left:left + size], labels[top:top + size, left:left + size]


class FilamentSegmentationModel(nn.Module):
    """ResNet-34 U-Net-style decoder with semantic and optional boundary heads."""

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if config.pretrained_encoder else None
        encoder = resnet34(weights=weights)
        if config.in_channels != 3:
            old = encoder.conv1
            encoder.conv1 = nn.Conv2d(config.in_channels, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                encoder.conv1.weight.copy_(old.weight.mean(1, keepdim=True))
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
    p_labels, g_labels = np.unique(pred_ids)[1:], np.unique(truth_ids)[1:]
    matrix = np.zeros((len(g_labels), len(p_labels)), dtype=float)
    for gi, g in enumerate(g_labels):
        for pi, p in enumerate(p_labels):
            intersection = np.logical_and(truth_ids == g, pred_ids == p).sum()
            union = np.logical_or(truth_ids == g, pred_ids == p).sum()
            matrix[gi, pi] = intersection / union if union else 0
    return matrix


def match_instances(iou: np.ndarray, threshold: float = .5) -> list[tuple[int, int, float]]:
    """Greedy maximum-IoU matching is exact at IoU > .5 for non-overlapping masks."""
    matches: list[tuple[int, int, float]] = []
    for flat in np.argsort(iou.ravel())[::-1]:
        g, p = np.unravel_index(flat, iou.shape)
        if iou[g, p] < threshold: break
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


def reconstruct_instances(probability: np.ndarray, config: PipelineConfig, boundary_probability: np.ndarray | None = None,
                          disk_mask: np.ndarray | None = None) -> np.ndarray:
    binary = probability >= config.threshold
    if disk_mask is not None: binary &= disk_mask
    if boundary_probability is not None: binary &= boundary_probability < .65
    if config.morphology_kernel > 1:
        kernel = np.ones((config.morphology_kernel, config.morphology_kernel), np.uint8)
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    if config.watershed and binary.any():
        distance = ndi.distance_transform_edt(binary)
        peaks = peak_local_max(distance, min_distance=config.watershed_peak_distance, labels=binary)
        markers = np.zeros(binary.shape, dtype=np.int32)
        for label, (y, x) in enumerate(peaks, 1): markers[y, x] = label
        markers, _ = ndi.label(markers)
        labels = watershed(-distance, markers, mask=binary) if markers.max() else ndi.label(binary)[0]
    else:
        labels, _ = ndi.label(binary)
    for label in np.unique(labels)[1:]:
        if (labels == label).sum() < config.min_component_area: labels[labels == label] = 0
    labels, _ = ndi.label(labels > 0)
    return labels.astype(np.int32)


def tile_starts(length: int, tile: int, overlap: float) -> list[int]:
    stride = max(1, int(tile * (1 - overlap))); starts = list(range(0, max(length - tile, 0) + 1, stride))
    return starts if starts and starts[-1] == max(length - tile, 0) else starts + [max(length - tile, 0)]


@torch.inference_mode()
def predict_tiled(model: nn.Module, image: np.ndarray, config: PipelineConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray | None]:
    """Full-resolution overlap-blended inference; no resize to a destructive 256 px."""
    image = robust_normalize(image); h, w = image.shape; tile = config.train_tile_size
    sums, counts, boundary_sums = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    model.eval()
    for y in tile_starts(h, tile, config.tile_overlap):
        for x in tile_starts(w, tile, config.tile_overlap):
            crop = image[y:y + tile, x:x + tile]; original_h, original_w = crop.shape
            crop = np.pad(crop, ((0, tile - original_h), (0, tile - original_w)))
            transforms = [(crop, lambda a: a)]
            if config.use_tta: transforms += [(np.fliplr(crop).copy(), lambda a: np.fliplr(a))]
            semantic, boundary = 0.0, 0.0
            for transformed, inverse in transforms:
                output = model(torch.from_numpy(transformed[None, None]).float().to(device))
                semantic += inverse(output["semantic"].sigmoid().cpu().numpy()[0, 0])
                boundary += inverse(output["boundary"].sigmoid().cpu().numpy()[0, 0])
            semantic, boundary = semantic / len(transforms), boundary / len(transforms)
            sums[y:y + original_h, x:x + original_w] += semantic[:original_h, :original_w]
            boundary_sums[y:y + original_h, x:x + original_w] += boundary[:original_h, :original_w]
            counts[y:y + original_h, x:x + original_w] += 1
    return sums / counts, boundary_sums / counts if config.boundary_head else None


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
    for path in list_image_files(data.test_images):
        image = np.asarray(Image.open(path).convert("L")); probability, boundary = predict_tiled(model, image, config, device)
        labels = reconstruct_instances(probability, config, boundary, detect_solar_disk(image))
        count = 0
        for label in np.unique(labels)[1:]:
            mask = labels == label
            validate_rle(mask)
            rows.append({"filament_id": f"{path.stem}_{label}", "segmentation_rle": mask_to_rle(mask)})
            count += 1
        per_image.append(count)
    submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    assert submission.filament_id.is_unique and not submission.isna().any().any()
    submission.to_csv(output_path, index=False)
    print({"test_observations": len(per_image), "submission_rows": len(submission), "predicted_instances": sum(per_image),
           "average_instances_image": float(np.mean(per_image)) if per_image else 0, "min_instances_image": min(per_image, default=0), "max_instances_image": max(per_image, default=0)})
    return submission


def run_synthetic_metric_tests() -> None:
    truth = np.zeros((10, 10), np.int32); truth[1:4, 1:4] = 1; truth[6:9, 6:9] = 2
    perfect = truth.copy(); result = compute_pq(perfect, truth)
    assert result["pq"] == 1 and result["tp"] == 2 and result["fp"] == result["fn"] == 0
    merged = np.where(truth > 0, 1, 0); result = compute_pq(merged, truth)
    assert result["many_to_one"] == 1 and result["pq"] < 1
    split = truth.copy(); split[1:4, 2] = 0; labels, _ = ndi.label(split > 0); result = compute_pq(labels, truth)
    assert result["one_to_many"] >= 1


def environment_report(config: PipelineConfig) -> dict[str, Any]:
    return {"config": asdict(config), "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
                    config: PipelineConfig, device: torch.device, pos_weight: float) -> float:
    """One AMP-enabled epoch with accumulation and clipping for large tiles."""
    model.train(); optimizer.zero_grad(set_to_none=True); losses: list[float] = []
    for step, batch in enumerate(loader, 1):
        batch = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}
        with torch.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
            loss = compound_loss(model(batch["image"]), batch, config, pos_weight) / config.gradient_accumulation
        scaler.scale(loss).backward()
        if step % config.gradient_accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()) * config.gradient_accumulation)
    return float(np.mean(losses))


@torch.inference_mode()
def validate(model: nn.Module, dataset: SolarFilamentDataset, config: PipelineConfig, device: torch.device) -> dict[str, float]:
    """Image-level semantic and global PQ validation; no crop leakage is possible."""
    aggregate: dict[str, list[float]] = defaultdict(list)
    totals: dict[str, float] = defaultdict(float)
    for image_id in dataset.image_ids:
        meta = dataset.index.images[image_id]
        image = np.asarray(Image.open(dataset.image_dir / meta["file_name"]).convert("L"))
        probability, boundary = predict_tiled(model, image, config, device)
        predicted = reconstruct_instances(probability, config, boundary, detect_solar_disk(image))
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
    return result


def optimize_postprocessing(probabilities: Iterable[np.ndarray], boundaries: Iterable[np.ndarray | None],
                            truths: Iterable[np.ndarray], base: PipelineConfig) -> tuple[PipelineConfig, pd.DataFrame]:
    """Small PQ-first search, intentionally avoiding an unbounded validation grid."""
    cached = list(zip(probabilities, boundaries, truths)); rows: list[dict[str, float]] = []
    candidates = [(threshold, area, watershed) for threshold in (.25, .35, .40, .50, .60)
                  for area in (20, 50, 100, 200) for watershed in (False, True)]
    for threshold, area, watershed_enabled in candidates:
        candidate = PipelineConfig(**{**asdict(base), "threshold": threshold, "min_component_area": area, "watershed": watershed_enabled})
        values = [compute_pq(reconstruct_instances(p, candidate, b), truth) for p, b, truth in cached]
        semantic = [compute_segmentation_metrics(reconstruct_instances(p, candidate, b) > 0, truth > 0) for p, b, truth in cached]
        rows.append({"threshold": threshold, "min_component_area": area, "watershed": watershed_enabled,
                     "pq": float(np.mean([x["pq"] for x in values])), "dice": float(np.mean([x["dice"] for x in semantic])),
                     "iou": float(np.mean([x["iou"] for x in semantic]))})
    table = pd.DataFrame(rows).sort_values(["pq", "dice"], ascending=False).reset_index(drop=True)
    best = table.iloc[0].to_dict()
    return PipelineConfig(**{**asdict(base), "threshold": float(best["threshold"]), "min_component_area": int(best["min_component_area"]), "watershed": bool(best["watershed"])}), table


def append_experiment(path: str | Path, name: str, config: PipelineConfig, metrics: dict[str, float], runtime_seconds: float) -> None:
    """Append measured results only; callers must never insert claimed scores."""
    row = {"experiment": name, "architecture": "resnet34_unet_boundary", "input_resolution": config.train_tile_size,
           "loss": json.dumps(config.loss_weights, sort_keys=True), "epochs": config.epochs, "threshold": config.threshold,
           "postprocessing": f"watershed={config.watershed};min_area={config.min_component_area}", "runtime_seconds": runtime_seconds,
           **metrics}
    destination = Path(path); previous = pd.read_csv(destination) if destination.exists() else pd.DataFrame()
    pd.concat([previous, pd.DataFrame([row])], ignore_index=True).to_csv(destination, index=False)
