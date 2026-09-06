"""The three-class representation, its decoder, and the ceiling it lifts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tribovision import evaluation
from tribovision.instance_model import (
    BACKGROUND,
    BOUNDARY,
    INTERIOR,
    InstanceConfig,
    ThreeClassDataset,
    decode_instances,
    three_class_target,
    train_instance_model,
)
from tribovision.model import TriboUNet
from tribovision.morphology import label_objects


def _touching_discs(gap: int = -4) -> np.ndarray:
    """Two discs whose separation is controlled; negative gap means overlapping."""
    ys, xs = np.mgrid[0:80, 0:80]
    labels = np.zeros((80, 80), dtype=np.int32)
    offset = 22 + gap // 2
    labels[((ys - 40) ** 2 + (xs - (40 - offset)) ** 2) <= 400] = 1
    labels[((ys - 40) ** 2 + (xs - (40 + offset)) ** 2) <= 400] = 2
    return labels


def test_the_target_separates_touching_cells_that_a_binary_mask_merges() -> None:
    labels = _touching_discs()
    target = three_class_target(labels)
    assert set(np.unique(target)) == {BACKGROUND, INTERIOR, BOUNDARY}
    # The binary view has one object; the interiors have two.
    assert label_objects(labels > 0).max() == 1
    from scipy import ndimage

    interiors, count = ndimage.label(target == INTERIOR, structure=np.ones((3, 3)))
    assert count == 2


def test_a_perfect_three_class_map_decodes_back_to_the_true_instances() -> None:
    labels = _touching_discs()
    target = three_class_target(labels)
    decoded = decode_instances(
        (target == INTERIOR).astype(float), (target > BACKGROUND).astype(float), min_area=10
    )
    assert decoded.max() == 2
    score = evaluation.matching_score(decoded, labels)["mean"]
    assert score > 0.6, f"perfect target should decode nearly perfectly, got {score}"


def _ruffled_cluster() -> np.ndarray:
    """Five touching cells with irregular outlines — the real failure mode.

    Two clean discs are separable by a distance-transform watershed, so they prove
    nothing. Adherent cells at confluence are ruffled and concave, and that is
    where splitting a binary mask breaks down.
    """
    ys, xs = np.mgrid[0:120, 0:120]
    labels = np.zeros((120, 120), dtype=np.int32)
    for index, (cy, cx) in enumerate([(35, 35), (35, 80), (80, 35), (80, 80), (58, 58)], 1):
        radius = 26 + 6 * np.sin(4 * np.arctan2(ys - cy, xs - cx))
        labels[((ys - cy) ** 2 + (xs - cx) ** 2) <= radius**2] = index
    return labels


def test_the_representation_lifts_the_measured_ceiling() -> None:
    """The claim the whole experiment rests on, asserted as a test.

    A perfect *binary* mask, split by watershed, cannot recover ruffled touching
    cells at all. A perfect three-class map recovers every one. If this inverts,
    the premise of the instance model is gone.
    """
    labels = _ruffled_cluster()
    assert labels.max() == 5
    binary_ceiling = evaluation.matching_score(
        label_objects(labels > 0, method="watershed_split", min_area=10), labels
    )["mean"]
    target = three_class_target(labels)
    decoded = decode_instances(
        (target == INTERIOR).astype(float), (target > BACKGROUND).astype(float), min_area=10
    )
    three_class_ceiling = evaluation.matching_score(decoded, labels)["mean"]
    assert binary_ceiling < 0.1, "a binary mask should fail here"
    assert three_class_ceiling > 0.9, "the three-class target should not"
    assert decoded.max() == 5


def test_an_empty_prediction_decodes_to_no_instances() -> None:
    empty = np.zeros((40, 40), dtype=float)
    assert decode_instances(empty, empty).max() == 0


def test_tiny_fragments_are_dropped_by_min_area() -> None:
    interior = np.zeros((40, 40), dtype=float)
    interior[5:15, 5:15] = 1.0
    interior[30, 30] = 1.0
    decoded = decode_instances(interior, interior, min_area=20)
    assert decoded.max() == 1


def test_the_model_head_widens_without_changing_the_body() -> None:
    binary = TriboUNet(base_channels=4, depth=2)
    three = TriboUNet(base_channels=4, depth=2, out_channels=3)
    assert binary(torch.zeros(1, 1, 32, 32)).shape[1] == 1
    assert three(torch.zeros(1, 1, 32, 32)).shape[1] == 3
    # Only the final 1x1 convolution differs, so the comparison is representation
    # against representation and not capacity against capacity.
    binary_body = sum(p.numel() for n, p in binary.named_parameters() if not n.startswith("output"))
    three_body = sum(p.numel() for n, p in three.named_parameters() if not n.startswith("output"))
    assert binary_body == three_body


@pytest.mark.parametrize("out_channels", [0, -1])
def test_an_invalid_head_width_is_rejected(out_channels: int) -> None:
    with pytest.raises(ValueError, match="out_channels"):
        TriboUNet(out_channels=out_channels)


def test_the_dataset_produces_a_letterboxed_class_map(tiny_training_data: Path) -> None:
    dataset = ThreeClassDataset(tiny_training_data / "manifests" / "train.jsonl", image_size=32)
    image, target, valid = dataset[0]
    assert image.shape == (1, 32, 32)
    assert target.shape == (32, 32) and target.dtype == torch.int64
    assert set(target.unique().tolist()) <= {BACKGROUND, INTERIOR, BOUNDARY}
    # Padding must stay background, or the model learns to read the border.
    assert target[valid[0] == 0].sum().item() == 0


def test_the_pipeline_can_learn_a_tiny_three_class_dataset(
    tmp_path: Path, tiny_training_data: Path
) -> None:
    result = train_instance_model(
        InstanceConfig(
            data_dir=tiny_training_data,
            output_dir=tmp_path / "run",
            epochs=25,
            batch_size=2,
            image_size=32,
            base_channels=4,
            depth=2,
            learning_rate=5e-3,
            device="cpu",
        ),
        progress=False,
    )
    assert (tmp_path / "run" / "best_model.pt").is_file()
    assert result["history"][-1]["train"]["loss"] < result["history"][0]["train"]["loss"]
    assert result["test"]["interior_recall"] > 0.5
    # No absolute paths in a published artifact.
    import json

    assert "/Users/" not in json.dumps(result["config"])
