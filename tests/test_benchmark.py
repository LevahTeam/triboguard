"""The acceptance gate: a learned model must beat the transparent rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tribovision.benchmark import _sign_test_p_value, compare
from tribovision.training import TrainConfig, train

SMALL = {"image_size": 32, "base_channels": 4, "depth": 2, "batch_size": 2, "device": "cpu"}


@pytest.fixture
def trained(tmp_path: Path, tiny_training_data: Path) -> Path:
    train(
        TrainConfig(
            data_dir=tiny_training_data,
            output_dir=tmp_path / "run",
            epochs=30,
            augment=False,
            learning_rate=5e-3,
            **SMALL,
        ),
        progress=False,
    )
    return tmp_path / "run" / "best_model.pt"


def test_comparison_scores_both_methods_on_the_same_held_out_images(
    trained: Path, tiny_training_data: Path, tmp_path: Path
) -> None:
    report = compare(
        trained,
        tiny_training_data / "manifests" / "test.jsonl",
        tmp_path / "comparison",
        device="cpu",
    )
    assert report["images"] == 2
    assert set(report["neural"]) >= {"macro_dice", "micro_dice", "matching_score_50_95"}
    assert set(report["classical"]) >= {"macro_dice", "micro_dice"}
    assert isinstance(report["passed"], bool)
    assert report["verdict"].startswith("neural")
    assert set(report["all_foreground"]) >= {"macro_dice", "macro_iou"}
    assert report["all_background"]["macro_dice"] == 0.0
    assert (tmp_path / "comparison" / "comparison.json").is_file()


def test_the_gate_is_a_real_comparison_not_a_rubber_stamp(
    trained: Path, tiny_training_data: Path, tmp_path: Path
) -> None:
    """A model trained to convergence on this fixture should win; the flag must track it."""
    report = compare(
        trained, tiny_training_data / "manifests" / "test.jsonl", tmp_path / "c", device="cpu"
    )
    floor = max(report["classical"]["macro_dice"], report["all_foreground"]["macro_dice"])
    assert report["passed"] == (report["neural"]["macro_dice"] > floor)


def test_the_bar_is_the_stronger_reference_not_the_weaker(
    trained: Path, tiny_training_data: Path, tmp_path: Path
) -> None:
    """Crowded frames make an all-foreground predictor beat the classical rule.

    Measuring only against the classical rule would let a mediocre model pass on
    dense images, so the floor is whichever reference scores higher.
    """
    report = compare(
        trained, tiny_training_data / "manifests" / "test.jsonl", tmp_path / "c", device="cpu"
    )
    assert report["reference_floor_macro_dice"] == max(
        report["classical"]["macro_dice"], report["all_foreground"]["macro_dice"]
    )
    assert report["margin_over_reference_floor"] == pytest.approx(
        report["neural"]["macro_dice"] - report["reference_floor_macro_dice"]
    )


def test_a_model_that_only_beats_the_weak_reference_does_not_pass(
    trained: Path, tiny_training_data: Path, tmp_path: Path
) -> None:
    """Threshold 0 makes the model predict everything - exactly the trivial predictor.

    It ties the all-foreground reference and so must not be reported as a win.
    """
    report = compare(
        trained,
        tiny_training_data / "manifests" / "test.jsonl",
        tmp_path / "c",
        device="cpu",
        threshold=1e-9,
    )
    assert report["neural"]["macro_dice"] == pytest.approx(
        report["all_foreground"]["macro_dice"], abs=1e-9
    )
    assert not report["passed"]


def test_a_deliberately_useless_threshold_fails_the_gate(
    trained: Path, tiny_training_data: Path, tmp_path: Path
) -> None:
    """Pushing the threshold to the extreme empties the mask, which must not pass."""
    report = compare(
        trained,
        tiny_training_data / "manifests" / "test.jsonl",
        tmp_path / "c",
        device="cpu",
        threshold=0.999999,
    )
    assert not report["passed"]
    assert report["verdict"] == "neural does NOT clear the reference floor"


def test_report_paths_are_repository_relative(
    trained: Path, tiny_training_data: Path, tmp_path: Path
) -> None:
    report = compare(
        trained, tiny_training_data / "manifests" / "test.jsonl", tmp_path / "c", device="cpu"
    )
    text = json.dumps({"a": report["checkpoint"], "b": report["manifest"]})
    assert "/Users/" not in text and "/home/" not in text


def test_max_images_limits_the_comparison(
    trained: Path, tiny_training_data: Path, tmp_path: Path
) -> None:
    report = compare(
        trained,
        tiny_training_data / "manifests" / "test.jsonl",
        None,
        device="cpu",
        max_images=1,
    )
    assert report["images"] == 1


@pytest.mark.parametrize(
    "wins,trials,expected",
    [(10, 10, pytest.approx(2 / 1024)), (5, 10, 1.0), (0, 0, 1.0)],
)
def test_sign_test_matches_the_exact_binomial(wins: int, trials: int, expected) -> None:
    assert _sign_test_p_value(wins, trials) == expected
