"""Synthetic regression tests for strict, competition-aligned PQ behaviour."""

import json

import numpy as np
from pycocotools import mask as coco_mask

from solar_filament.pipeline import (build_instance_masks, compute_pq, load_coco_annotations,
                                     match_instances, mask_to_rle, relabel_sequential)


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


def test_string_ids_and_polygon_rasterization(tmp_path) -> None:
    payload = {
        "images": [{"id": "annotator-A_20260101000000Bh", "width": 2048, "height": 2048, "file_name": "20260101000000Bh.jpeg"}],
        "annotations": [{"id": "0e4dc87e-4f5d-4ebb-a1f6-123456789abc", "image_id": "annotator-A_20260101000000Bh", "category_id": 1,
                         "segmentation": [[10.5, 10.5, 30.5, 10.5, 30.5, 30.5, 10.5, 30.5]], "area": 400.0,
                         "bbox": [10.5, 10.5, 20.0, 20.0], "iscrowd": 0}], "categories": [{"id": 1, "name": "Left"}],
    }
    annotation_path = tmp_path / "annotations.json"; annotation_path.write_text(json.dumps(payload))
    index = load_coco_annotations(annotation_path)
    image_id = "annotator-A_20260101000000Bh"
    assert isinstance(next(iter(index.images)), str)
    assert index.annotations_by_image[image_id][0].annotation_id == "0e4dc87e-4f5d-4ebb-a1f6-123456789abc"
    mask = build_instance_masks(index, image_id)[0]
    assert mask.shape == (2048, 2048) and mask.any()


def test_rle_round_trip() -> None:
    mask = np.zeros((8, 9), dtype=bool); mask[2:6, 3:7] = True
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    assert isinstance(mask_to_rle(mask), str)
    assert np.array_equal(coco_mask.decode(encoded).astype(bool), mask)
