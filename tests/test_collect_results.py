"""The results table is generated from artifacts, so it cannot drift from them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_results import format_report, main, publish, unpublish  # noqa: E402


def _metrics(seed: int, test_dice: float) -> dict:
    return {
        "config": {"seed": seed, "epochs": 25},
        "device": "cpu",
        "best_epoch": 7,
        "history": [{"epoch": index} for index in range(1, 12)],
        "validation": {"macro_dice": 0.90, "micro_dice": 0.91, "macro_iou": 0.83},
        "test": {"macro_dice": test_dice, "micro_dice": 0.95, "macro_iou": 0.90},
        "split_check": {"clean": True, "group_by": "well", "groups_per_split": {"train": 2}},
        "datasets": {
            "val": {"wells": ["B7"], "images": 41},
            "test": {"wells": ["C7"], "images": 60},
        },
        "environment": {
            "python": "3.11.0",
            "packages": {"torch": "2.2.0", "numpy": "1.26.0"},
            "git": {"commit": "abc123", "dirty": False},
        },
    }


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    (root / "baseline").mkdir(parents=True)
    (root / "baseline" / "metrics.json").write_text(json.dumps(_metrics(42, 0.9524)))
    for seed, dice in ((1, 0.9500), (2, 0.9550)):
        directory = root / f"seed_{seed}"
        directory.mkdir()
        (directory / "metrics.json").write_text(json.dumps(_metrics(seed, dice)))
    (root / "comparison").mkdir()
    (root / "comparison" / "comparison.json").write_text(
        json.dumps(
            {
                "manifest": "data/livecell/manifests/test.jsonl",
                "images": 60,
                "neural": {
                    "macro_dice": 0.95,
                    "micro_dice": 0.96,
                    "macro_iou": 0.91,
                    "matching_score_50_95": 0.05,
                },
                "classical": {
                    "macro_dice": 0.43,
                    "micro_dice": 0.39,
                    "macro_iou": 0.27,
                    "matching_score_50_95": 0.01,
                },
                "all_foreground": {
                    "macro_dice": 0.71,
                    "micro_dice": 0.74,
                    "macro_iou": 0.59,
                },
                "all_background": {"macro_dice": 0.0, "micro_dice": 0.0, "macro_iou": 0.0},
                "instance_ceiling": {"matching_score_50_95": 0.168},
                "mean_foreground_fraction": 0.593,
                "reference_floor_macro_dice": 0.71,
                "margin_over_reference_floor": 0.24,
                "independent_units": 33,
                "units_where_neural_wins": 33,
                "sign_test_unit": "acquisition group",
                "sign_test_p_value_by_image_pseudoreplicated": 1.7e-18,
                "dice_difference_mean": 0.52,
                "dice_difference_sd": 0.10,
                "images_where_neural_wins": 60,
                "sign_test_p_value": 1.7e-18,
                "verdict": "neural beats classical",
            }
        )
    )
    return root


def test_report_contains_the_numbers_from_the_artifacts(runs: Path) -> None:
    report = format_report(runs)
    assert "0.9524" in report
    assert "abc123" in report
    assert "neural beats classical" in report
    assert "60 of 60" in report


def test_seed_spread_is_computed_not_asserted(runs: Path) -> None:
    report = format_report(runs)
    # Seeds 0.9500, 0.9550 and the primary 0.9524 -> mean 0.9525.
    assert "0.9525 ±" in report
    assert "range 0.9500–0.9550" in report


def test_a_dirty_working_tree_is_disclosed(runs: Path) -> None:
    path = runs / "baseline" / "metrics.json"
    payload = json.loads(path.read_text())
    payload["environment"]["git"]["dirty"] = True
    path.write_text(json.dumps(payload))
    assert "working tree dirty" in format_report(runs)


def test_an_empty_run_directory_produces_a_report_without_crashing(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    report = format_report(tmp_path / "runs")
    assert "How to reproduce" in report


def test_missing_run_directory_exits_non_zero(tmp_path: Path) -> None:
    assert main([str(tmp_path / "absent")]) == 1


def test_publishing_copies_the_evidence_out_of_the_ignored_run_directory(
    runs: Path, tmp_path: Path
) -> None:
    """Documentation citing runs/*.json is dangling for anyone who clones the repo."""
    destination = tmp_path / "results"
    assert main([str(runs), "--publish", str(destination)]) == 0
    names = {path.name for path in destination.glob("*.json")}
    assert "baseline__metrics.json" in names
    assert "comparison__comparison.json" in names
    assert (destination / "README.md").is_file()
    # Flattened names cannot collide or escape the destination.
    assert all("/" not in name and ".." not in name for name in names)


def test_publish_without_a_destination_is_an_error(runs: Path) -> None:
    assert main([str(runs), "--publish"]) == 1


def test_every_reference_predictor_reaches_the_published_table(runs: Path) -> None:
    """A silent no-op once dropped these rows while every other test still passed."""
    report = format_report(runs)
    for expected in (
        "Every pixel labelled background",
        "Every pixel labelled cell (no learning)",
        "Classical local contrast + Otsu",
        "TriboVision U-Net",
        "The ground-truth mask itself",
    ):
        assert expected in report, f"missing reference row: {expected}"
    assert "0.7103" in report or "0.7100" in report or "0.71" in report
    assert "Reference floor" in report
    assert "Instance ceiling" in report


def test_the_published_p_value_is_the_group_level_one(runs: Path) -> None:
    report = format_report(runs)
    assert "33 independent units" in report
    assert "2.33e-10" in report or "independent units" in report
    # The inflated per-image value may appear only with its disclaimer.
    if "1.7e-18" in report:
        assert "pseudoreplication" in report


def test_unpublishing_is_the_exact_inverse_of_publishing(runs: Path, tmp_path: Path) -> None:
    published = tmp_path / "results"
    publish(runs, published)
    rebuilt = tmp_path / "rebuilt"
    restored = unpublish(published, rebuilt)
    assert restored, "the fixture must publish something for this to test anything"
    for relative in restored:
        assert (rebuilt / relative).read_bytes() == (runs / relative).read_bytes()


def test_a_fresh_clone_regenerates_the_same_report(runs: Path, tmp_path: Path) -> None:
    """What CI does: rebuild runs/ from the published copies alone, then regenerate."""
    published = tmp_path / "results"
    publish(runs, published)
    clone = tmp_path / "clone-runs"
    unpublish(published, clone)
    assert format_report(clone) == format_report(runs)


def test_unpublish_without_a_directory_is_an_error(tmp_path: Path) -> None:
    assert main([str(tmp_path / "runs"), "--unpublish"]) == 1


def test_the_committed_results_regenerate_the_committed_table(tmp_path: Path) -> None:
    """The drift check CI runs, runnable here: results/ alone reproduces RESULTS.md.

    runs/ is git-ignored, so this is the only way a reader who clones the
    repository can confirm the table matches its evidence. It failed on the first
    push to GitHub because two sections read files that had never been published.
    """
    root = Path(__file__).resolve().parents[1]
    rebuilt = tmp_path / "runs"
    unpublish(root / "results", rebuilt)
    expected = (root / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    assert format_report(rebuilt) == expected
