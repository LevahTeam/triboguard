"""The results table is generated from artifacts, so it cannot drift from them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_results import format_report, main  # noqa: E402


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
