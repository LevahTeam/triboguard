"""COCO decoding: exactness against pycocotools plus malformed-input handling."""

from __future__ import annotations

import numpy as np
import pytest

from tribovision import coco
from tribovision.coco import CocoSegmentationError, segmentation_mask

pycocotools = pytest.importorskip("pycocotools")
from pycocotools import mask as mask_utils  # noqa: E402

POLYGONS = [
    [[2.0, 2.0, 11.0, 2.0, 11.0, 11.0, 2.0, 11.0]],
    [[1.5, 3.25, 18.75, 2.0, 20.0, 15.5, 4.0, 19.25]],
    [[0.0, 0.0, 31.0, 0.0, 31.0, 1.0, 0.0, 1.0]],
    [[5.0, 5.0, 25.0, 6.0, 24.0, 20.0, 12.0, 24.0, 6.0, 14.0]],
]


@pytest.mark.parametrize("polygon", POLYGONS)
def test_polygon_rasterisation_is_pixel_identical_to_pycocotools(polygon: list) -> None:
    width, height = 32, 28
    reference = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(polygon, height, width)))
    produced = segmentation_mask({"id": 1, "segmentation": polygon}, width, height)
    assert np.array_equal(produced, reference)


def test_backend_is_the_exact_one_when_pycocotools_is_installed() -> None:
    assert coco.backend() == "pycocotools"


def test_flat_polygon_is_treated_as_a_single_ring() -> None:
    flat = [2, 2, 11, 2, 11, 11, 2, 11]
    nested = segmentation_mask({"id": 1, "segmentation": [flat]}, 16, 16)
    assert np.array_equal(segmentation_mask({"id": 1, "segmentation": flat}, 16, 16), nested)


def test_multiple_polygons_are_unioned() -> None:
    mask = segmentation_mask(
        {"id": 1, "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4], [7, 7, 9, 7, 9, 9, 7, 9]]}, 12, 12
    )
    assert mask[2, 2] == 1 and mask[8, 8] == 1 and mask[0, 0] == 0


@pytest.mark.parametrize(
    "segmentation",
    [
        [[1, 1, 3, 1]],
        [[2, 2, 4, 2, 4]],
        [[float("inf"), 1, 2, 2, 3, 3]],
        [[1, 1, 2, 2, float("nan")]],
    ],
)
def test_malformed_polygons_are_rejected(segmentation: list) -> None:
    with pytest.raises(CocoSegmentationError):
        segmentation_mask({"id": 2, "segmentation": segmentation}, 8, 8)


def test_unknown_encoding_is_rejected() -> None:
    with pytest.raises(CocoSegmentationError, match="Unsupported segmentation"):
        segmentation_mask({"id": 3, "segmentation": "invalid"}, 8, 8)
    with pytest.raises(CocoSegmentationError, match="Malformed polygon"):
        segmentation_mask({"id": 4, "segmentation": [None, 3]}, 8, 8)


@pytest.mark.parametrize("width,height", [(0, 8), (8, 0), (-1, 4)])
def test_invalid_dimensions_are_rejected(width: int, height: int) -> None:
    with pytest.raises(CocoSegmentationError, match="Invalid image dimensions"):
        segmentation_mask({"id": 5, "segmentation": [[1, 1, 2, 2, 3, 3]]}, width, height)


def test_uncompressed_rle_round_trips() -> None:
    mask = segmentation_mask(
        {"id": 10, "segmentation": {"size": [4, 4], "counts": [5, 2, 9]}}, 4, 4
    )
    assert mask.shape == (4, 4) and mask.sum() == 2


def test_compressed_rle_round_trips() -> None:
    original = np.zeros((6, 6), dtype=np.uint8)
    original[1:4, 2:5] = 1
    encoded = mask_utils.encode(np.asfortranarray(original))
    encoded["counts"] = encoded["counts"].decode("ascii")
    assert np.array_equal(segmentation_mask({"id": 11, "segmentation": encoded}, 6, 6), original)


def test_rle_size_mismatch_is_rejected() -> None:
    with pytest.raises(CocoSegmentationError, match="does not match image size"):
        segmentation_mask({"id": 9, "segmentation": {"size": [7, 8], "counts": [56]}}, 8, 8)


def test_rle_with_broken_counts_is_rejected() -> None:
    with pytest.raises(CocoSegmentationError):
        segmentation_mask({"id": 12, "segmentation": {"size": [4, 4], "counts": "!!!!"}}, 4, 4)


def test_empty_annotation_list_gives_an_empty_stack_and_mask() -> None:
    assert coco.instance_masks([], 5, 4).shape == (0, 4, 5)
    assert coco.semantic_mask([], 5, 4).sum() == 0
    assert coco.label_image([], 5, 4).max() == 0


def test_label_image_numbers_instances_in_order() -> None:
    annotations = [
        {"id": 1, "segmentation": [[0, 0, 3, 0, 3, 3, 0, 3]]},
        {"id": 2, "segmentation": [[6, 6, 9, 6, 9, 9, 6, 9]]},
    ]
    labels = coco.label_image(annotations, 12, 12)
    assert labels.max() == 2
    assert labels[1, 1] == 1 and labels[7, 7] == 2
