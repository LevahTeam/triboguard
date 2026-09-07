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


def test_the_training_set_can_be_capped_without_touching_the_held_out_splits(
    tmp_path: Path, tiny_training_data: Path
) -> None:
    """Every point on a scaling curve must be scored on the same held-out images."""
    result = train_instance_model(
        InstanceConfig(
            data_dir=tiny_training_data,
            output_dir=tmp_path / "run",
            epochs=2,
            batch_size=1,
            image_size=32,
            base_channels=4,
            depth=2,
            device="cpu",
            train_limit=1,
        ),
        progress=False,
    )
    assert result["training_images"] == 1
    assert result["training_images_available"] == 2
    # Validation and test are untouched by the cap.
    assert result["test"]["samples"] == 2
    assert result["best_validation"]["samples"] == 2


def test_a_cap_above_the_available_images_is_a_no_op(
    tmp_path: Path, tiny_training_data: Path
) -> None:
    result = train_instance_model(
        InstanceConfig(
            data_dir=tiny_training_data,
            output_dir=tmp_path / "run",
            epochs=1,
            batch_size=1,
            image_size=32,
            base_channels=4,
            depth=2,
            device="cpu",
            train_limit=999,
        ),
        progress=False,
    )
    assert result["training_images"] == result["training_images_available"] == 2


def test_an_invalid_cap_is_rejected(tmp_path: Path, tiny_training_data: Path) -> None:
    with pytest.raises(ValueError, match="train_limit"):
        train_instance_model(
            InstanceConfig(
                data_dir=tiny_training_data,
                output_dir=tmp_path / "run",
                epochs=1,
                image_size=32,
                base_channels=4,
                depth=2,
                device="cpu",
                train_limit=0,
            ),
            progress=False,
        )


def test_the_instance_benchmark_reports_intervals_and_paired_differences(
    tmp_path: Path, tiny_training_data: Path
) -> None:
    """Every method gets an interval; every comparison gets a paired difference."""
    from tribovision.instance_benchmark import run as run_benchmark

    report = run_benchmark(
        tiny_training_data / "manifests" / "test.jsonl",
        None,
        include_cellpose=False,
        min_area=2,
    )
    for method, values in report["summary"].items():
        interval = values.get("matching_50_95_ci")
        assert interval is not None, f"{method} has no interval"
        if interval.get("evaluated"):
            assert interval["ci_low"] <= values["matching_50_95"] <= interval["ci_high"]
    # No checkpoint was given, so only the ground-truth ceiling and the classical
    # rule are scored; the ceiling comparison is always available.
    assert "ceiling" in report["paired_differences"]
    assert "classical" in report["paired_differences"]["ceiling"]
    assert "resampled by acquisition group" in report["paired_difference_note"]


def test_scoring_a_checkpoint_on_a_manifest_matches_the_full_benchmark(
    tmp_path: Path, tiny_training_data: Path
) -> None:
    """The lean scorer must agree with the benchmark it is a shortcut for."""
    from tribovision.instance_benchmark import run as run_benchmark
    from tribovision.instance_model import score_on_manifest

    train_instance_model(
        InstanceConfig(
            data_dir=tiny_training_data,
            output_dir=tmp_path / "run",
            epochs=6,
            batch_size=1,
            image_size=32,
            base_channels=4,
            depth=2,
            device="cpu",
        ),
        progress=False,
    )
    checkpoint = tmp_path / "run" / "best_model.pt"
    manifest = tiny_training_data / "manifests" / "test.jsonl"

    lean = score_on_manifest(checkpoint, manifest, device="cpu", min_area=2)
    full = run_benchmark(
        manifest,
        None,
        three_class_checkpoint=checkpoint,
        include_cellpose=False,
        min_area=2,
        device="cpu",
    )
    assert lean["matching_50_95"] == pytest.approx(
        full["summary"]["tribovision_three_class"]["matching_50_95"], abs=1e-9
    )
    assert lean["images"] == full["images"]
    assert len(lean["per_image"]) == len(lean["groups"])


def test_the_lean_scorer_reports_repository_relative_paths(
    tmp_path: Path, tiny_training_data: Path
) -> None:
    from tribovision.instance_model import score_on_manifest

    train_instance_model(
        InstanceConfig(
            data_dir=tiny_training_data,
            output_dir=tmp_path / "run",
            epochs=2,
            batch_size=1,
            image_size=32,
            base_channels=4,
            depth=2,
            device="cpu",
        ),
        progress=False,
    )
    result = score_on_manifest(
        tmp_path / "run" / "best_model.pt",
        tiny_training_data / "manifests" / "test.jsonl",
        device="cpu",
        min_area=2,
    )
    assert "/Users/" not in result["checkpoint"] + result["manifest"]
