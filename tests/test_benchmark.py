"""The acceptance gate: a learned model must beat every non-learned reference.

The fixture matters as much as the assertions here. An earlier version of these
tests used flat-background squares, on which the classical Otsu rule scores a
perfect 1.0 — so the gate could never pass, every "must fail" test passed for the
wrong reason, and a mutation that deleted the trivial-predictor floor entirely
survived the whole suite. The crowded fixture reproduces the property that makes
the floor necessary on real data: labelling every pixel a cell beats the fixed
rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import build_crowded_dataset, build_dataset

from tribovision.benchmark import _group_differences, _sign_test_p_value, compare
from tribovision.training import TrainConfig, train

SMALL = {"image_size": 32, "base_channels": 4, "depth": 2, "batch_size": 2, "device": "cpu"}


@pytest.fixture(scope="module")
def crowded(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Train once on frames a fixed threshold cannot solve; reuse across tests."""
    root = tmp_path_factory.mktemp("crowded")
    data = build_crowded_dataset(root / "data")
    train(
        TrainConfig(
            data_dir=data,
            output_dir=root / "run",
            epochs=60,
            batch_size=2,
            image_size=64,
            base_channels=8,
            depth=2,
            device="cpu",
            learning_rate=3e-3,
        ),
        progress=False,
    )
    return root / "run" / "best_model.pt", data


def test_the_fixture_reproduces_the_property_that_makes_the_floor_necessary(
    crowded: tuple[Path, Path], tmp_path: Path
) -> None:
    """All-foreground must beat the classical rule, as it does on real LIVECell."""
    checkpoint, data = crowded
    report = compare(checkpoint, data / "manifests" / "test.jsonl", None, device="cpu")
    assert report["all_foreground"]["macro_dice"] > report["classical"]["macro_dice"]
    assert report["reference_floor_macro_dice"] == report["all_foreground"]["macro_dice"]


def test_a_good_model_passes_the_gate(crowded: tuple[Path, Path], tmp_path: Path) -> None:
    """The success branch — previously never executed by any test."""
    checkpoint, data = crowded
    report = compare(checkpoint, data / "manifests" / "test.jsonl", tmp_path / "cmp", device="cpu")
    assert report["passed"] is True
    assert report["verdict"] == "neural beats every reference predictor"
    assert report["neural"]["macro_dice"] > report["reference_floor_macro_dice"]
    assert report["margin_over_reference_floor"] > 0
    assert (tmp_path / "cmp" / "comparison.json").is_file()


def test_a_model_that_only_ties_the_trivial_predictor_fails(crowded: tuple[Path, Path]) -> None:
    """Threshold near zero makes the model predict everything — the trivial predictor."""
    checkpoint, data = crowded
    report = compare(
        checkpoint, data / "manifests" / "test.jsonl", None, device="cpu", threshold=1e-9
    )
    assert report["neural"]["macro_dice"] == pytest.approx(
        report["all_foreground"]["macro_dice"], abs=1e-9
    )
    assert report["passed"] is False
    assert report["verdict"] == "neural does NOT clear the reference floor"


def test_an_empty_prediction_fails_the_gate(crowded: tuple[Path, Path]) -> None:
    checkpoint, data = crowded
    report = compare(
        checkpoint,
        data / "manifests" / "test.jsonl",
        None,
        device="cpu",
        threshold=0.999999,
    )
    assert report["passed"] is False


def test_the_floor_is_the_stronger_reference_not_whichever_comes_first(
    crowded: tuple[Path, Path], tmp_path: Path
) -> None:
    """Guards the specific mutation that deleted the trivial predictor from the floor.

    On the crowded fixture the trivial predictor is the stronger reference; on the
    flat fixture the classical rule is. Asserting `max` on both means neither
    reference can be dropped without a failure.
    """
    checkpoint, data = crowded
    crowded_report = compare(checkpoint, data / "manifests" / "test.jsonl", None, device="cpu")
    assert (
        crowded_report["reference_floor_macro_dice"]
        == crowded_report["all_foreground"]["macro_dice"]
    )

    flat = build_dataset(tmp_path / "flat")
    train(
        TrainConfig(
            data_dir=flat, output_dir=tmp_path / "flatrun", epochs=10, augment=False, **SMALL
        ),
        progress=False,
    )
    flat_report = compare(
        tmp_path / "flatrun" / "best_model.pt",
        flat / "manifests" / "test.jsonl",
        None,
        device="cpu",
    )
    # Flat squares are perfectly separable by a threshold, so here the fixed rule wins.
    assert flat_report["classical"]["macro_dice"] > flat_report["all_foreground"]["macro_dice"]
    assert flat_report["reference_floor_macro_dice"] == flat_report["classical"]["macro_dice"]


def test_all_reference_predictors_are_reported(crowded: tuple[Path, Path]) -> None:
    checkpoint, data = crowded
    report = compare(checkpoint, data / "manifests" / "test.jsonl", None, device="cpu")
    assert report["all_background"]["macro_dice"] == 0.0
    assert 0.0 < report["mean_foreground_fraction"] < 1.0
    for key in ("neural", "classical", "all_foreground"):
        assert set(report[key]) >= {"macro_dice", "micro_dice", "macro_iou"}


def test_report_paths_are_repository_relative(crowded: tuple[Path, Path]) -> None:
    checkpoint, data = crowded
    report = compare(checkpoint, data / "manifests" / "test.jsonl", None, device="cpu")
    text = json.dumps({"a": report["checkpoint"], "b": report["manifest"]})
    assert "/Users/" not in text and "/home/" not in text


def test_max_images_limits_the_comparison(crowded: tuple[Path, Path]) -> None:
    checkpoint, data = crowded
    report = compare(
        checkpoint, data / "manifests" / "test.jsonl", None, device="cpu", max_images=1
    )
    assert report["images"] == 1


# ------------------------------------------------------- independence of trials


def test_the_headline_p_value_counts_acquisition_groups_not_crops(
    crowded: tuple[Path, Path],
) -> None:
    """LIVECell tiles one captured frame into crops, so images are not independent."""
    checkpoint, data = crowded
    report = compare(checkpoint, data / "manifests" / "test.jsonl", None, device="cpu")
    assert report["independent_units"] <= report["images"]
    assert report["sign_test_unit"].startswith("acquisition group")
    assert "must not be quoted" in report["independence_note"]
    assert "sign_test_p_value_by_image_pseudoreplicated" in report


def test_grouping_collapses_crops_of_one_field_into_one_trial() -> None:
    rows = [
        {"acquisition_group": "A172|C7|1|00d", "neural_dice": 0.9, "classical_dice": 0.4},
        {"acquisition_group": "A172|C7|1|00d", "neural_dice": 0.7, "classical_dice": 0.4},
        {"acquisition_group": "A172|C7|2|00d", "neural_dice": 0.2, "classical_dice": 0.4},
    ]
    grouped = _group_differences(rows, "acquisition_group")
    assert len(grouped) == 2
    assert grouped["A172|C7|1|00d"] == pytest.approx(0.4)
    assert grouped["A172|C7|2|00d"] == pytest.approx(-0.2)


@pytest.mark.parametrize(
    "wins,trials,expected",
    [
        (10, 10, pytest.approx(2 / 1024)),
        # Losing every trial is exactly as surprising as winning every trial.
        (0, 10, pytest.approx(2 / 1024)),
        (8, 10, pytest.approx(112 / 1024)),
        (2, 10, pytest.approx(112 / 1024)),
        (5, 10, 1.0),
        (0, 0, 1.0),
    ],
)
def test_sign_test_matches_the_exact_two_sided_binomial(wins: int, trials: int, expected) -> None:
    assert _sign_test_p_value(wins, trials) == expected


def test_sign_test_is_symmetric_about_half() -> None:
    """Without the two-sided correction, a total loss would look unremarkable."""
    for wins in range(11):
        assert _sign_test_p_value(wins, 10) == pytest.approx(_sign_test_p_value(10 - wins, 10))


def test_the_instance_ceiling_is_reported_alongside_the_instance_score(
    crowded: tuple[Path, Path],
) -> None:
    """0.05 against a ceiling of 0.12 is a different statement from 0.05 out of 1.0."""
    checkpoint, data = crowded
    report = compare(checkpoint, data / "manifests" / "test.jsonl", None, device="cpu")
    ceiling = report["instance_ceiling"]["matching_score_50_95"]
    assert 0.0 <= ceiling <= 1.0
    # The ground truth cannot do worse than the model at separating its own cells.
    assert ceiling >= report["neural"]["matching_score_50_95"] - 1e-9
    assert "predicts instances directly" in report["instance_ceiling"]["explanation"]


def test_the_comparison_reports_intervals_and_a_paired_margin(
    crowded: tuple[Path, Path], tmp_path: Path
) -> None:
    """A headline number without uncertainty invites a question it cannot answer."""
    checkpoint, data = crowded
    report = compare(checkpoint, data / "manifests" / "test.jsonl", None, device="cpu")
    for key in ("neural", "classical", "all_foreground"):
        assert key in report["dice_intervals"]
    margin = report["margin_over_best_reference"]
    if margin.get("evaluated"):
        # The margin is measured against the stronger reference, per image.
        assert margin["ci_low"] <= margin["difference"] <= margin["ci_high"]
        assert "excludes_zero" in margin


def test_intervals_resample_acquisition_groups_not_crops(crowded: tuple[Path, Path]) -> None:
    checkpoint, data = crowded
    report = compare(checkpoint, data / "manifests" / "test.jsonl", None, device="cpu")
    neural = report["dice_intervals"]["neural"]
    if neural.get("evaluated"):
        assert neural["resampling_unit"] == "acquisition group"
        assert neural["clusters"] <= report["images"]
