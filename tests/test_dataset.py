"""Dataset loading: geometry, normalisation, caching, and error handling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import rewrite_manifest
from PIL import Image

from tribovision.data import DatasetError, LiveCellDataset, dataset_fingerprint


def load(root: Path, split: str = "train", **kwargs) -> LiveCellDataset:
    return LiveCellDataset(root / "manifests" / f"{split}.jsonl", image_size=32, **kwargs)


def test_items_are_image_mask_and_validity(tiny_training_data: Path) -> None:
    dataset = load(tiny_training_data)
    image, mask, valid = dataset[0]
    assert image.shape == mask.shape == valid.shape == (1, 32, 32)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert set(valid.unique().tolist()) <= {0.0, 1.0}
    assert mask.sum().item() > 0


def test_padding_is_marked_invalid_and_carries_no_signal(tiny_training_data: Path) -> None:
    """The fixture is 40x24, so a square canvas must contain real padding."""
    dataset = load(tiny_training_data)
    image, mask, valid = dataset[0]
    assert 0.0 < valid.mean().item() < 1.0
    assert image[valid == 0].abs().max().item() == 0.0
    assert mask[valid == 0].sum().item() == 0.0


def test_intensity_is_standardised_over_real_pixels_only(tiny_training_data: Path) -> None:
    image, _, valid = load(tiny_training_data)[0]
    interior = image[valid > 0]
    assert abs(float(interior.mean())) < 1e-4
    assert abs(float(interior.std(unbiased=False)) - 1.0) < 1e-3


def test_masks_are_cached_and_the_cache_is_reused(tiny_training_data: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    dataset = load(tiny_training_data, cache_dir=cache)
    first = dataset.native_mask(0)
    assert list(cache.glob("*.npy"))

    fresh = load(tiny_training_data, cache_dir=cache)
    assert np.array_equal(fresh.native_mask(0), first)


def test_a_read_only_cache_location_does_not_break_loading(
    tiny_training_data: Path, tmp_path: Path
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    dataset = load(tiny_training_data, cache_dir=blocked)
    assert dataset.native_mask(0).sum() > 0


@pytest.mark.parametrize("size", [0, -8, 17, 33, 2.5, True])
def test_invalid_image_size_is_rejected(tiny_training_data: Path, size: object) -> None:
    with pytest.raises(ValueError):
        LiveCellDataset(tiny_training_data / "manifests" / "train.jsonl", image_size=size)


def test_the_dataset_defers_the_depth_dependent_size_rule_to_the_model() -> None:
    """The dataset cannot know the network depth, so it only checks what it can."""
    from tribovision.model import TriboUNet

    # 30 is even and loadable, but a depth-3 U-Net still rejects it.
    with pytest.raises(ValueError, match="nearest valid size"):
        TriboUNet(base_channels=2, depth=3).check_input_size(30, 30)


def test_manifest_problems_surface_as_dataset_errors(tiny_training_data: Path) -> None:
    rewrite_manifest(
        tiny_training_data, "train", lambda rows: [{**row, "annotation_ids": [42]} for row in rows]
    )
    with pytest.raises(DatasetError, match="missing from"):
        load(tiny_training_data)


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="does not exist"):
        LiveCellDataset(tmp_path / "nope.jsonl", image_size=32)


def test_image_that_disagrees_with_the_manifest_is_rejected(tiny_training_data: Path) -> None:
    record = json.loads(
        (tiny_training_data / "manifests" / "train.jsonl").read_text().splitlines()[0]
    )
    path = tiny_training_data / record["image_path"]
    Image.new("L", (60, 60), 10).save(path)
    dataset = load(tiny_training_data, verify_hashes=False)
    with pytest.raises(DatasetError, match="does not match manifest size"):
        dataset[0]


def test_unreadable_image_is_reported(tiny_training_data: Path) -> None:
    record = json.loads(
        (tiny_training_data / "manifests" / "train.jsonl").read_text().splitlines()[0]
    )
    (tiny_training_data / record["image_path"]).write_bytes(b"definitely not a tiff")
    dataset = load(tiny_training_data, verify_hashes=False)
    with pytest.raises(DatasetError, match="Cannot read image"):
        dataset[0]


def test_an_image_with_no_annotations_yields_an_empty_mask(tiny_training_data: Path) -> None:
    rewrite_manifest(
        tiny_training_data, "train", lambda rows: [{**row, "annotation_ids": []} for row in rows]
    )
    _, mask, _ = load(tiny_training_data)[0]
    assert mask.sum().item() == 0.0


def test_fingerprint_identifies_the_exact_data_used(tiny_training_data: Path) -> None:
    fingerprint = dataset_fingerprint(load(tiny_training_data))
    assert fingerprint["images"] == 2
    assert fingerprint["wells"] == ["A1"]
    assert fingerprint["coco_backend"] in {"pycocotools", "pillow-approximate"}
    assert len(fingerprint["content_sha256"]) == 64


def test_dataset_collates_into_batches(tiny_training_data: Path) -> None:
    loader = torch.utils.data.DataLoader(load(tiny_training_data), batch_size=2)
    images, masks, valid = next(iter(loader))
    assert images.shape == masks.shape == valid.shape == (2, 1, 32, 32)
