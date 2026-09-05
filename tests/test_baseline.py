"""The classical baseline: outputs, safety, and honest metric naming."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import rewrite_manifest
from PIL import Image

from tribovision.baseline import otsu_threshold, run_baseline, segment_classical
from tribovision.manifest import ManifestError


def test_otsu_separates_two_intensity_groups() -> None:
    values = np.array([0] * 20 + [200] * 20, dtype=np.uint8)
    assert 0 <= otsu_threshold(values) < 200
    assert otsu_threshold(np.array([], dtype=np.uint8)) == 0


def test_segmentation_finds_a_bright_block_against_a_flat_background() -> None:
    pixels = np.full((40, 40), 30, dtype=np.uint8)
    pixels[10:30, 10:30] = 220
    mask = segment_classical(Image.fromarray(pixels), background_radius=3.0, min_area=4)
    # The block must be found, and the far corner must stay background.
    assert mask[10:30, 10:30].mean() > 0.5
    assert mask[0:5, 0:5].sum() == 0


def test_baseline_writes_auditable_outputs(tiny_training_data: Path, tmp_path: Path) -> None:
    output = tmp_path / "classical"
    report = run_baseline(
        tiny_training_data / "manifests" / "test.jsonl",
        output,
        max_images=1,
        background_radius=2.0,
        min_area=2,
    )

    assert report["overall"]["images"] == 1
    # A real value, not `0 <= x <= 1`. With this small a blur radius the rule
    # dilates each square (tp 100, fp 80, fn 0), giving exactly 0.714 Dice, and
    # it finds the right number of objects.
    assert report["overall"]["macro_dice"] == pytest.approx(0.7143, abs=0.01)
    assert report["overall"]["total_fn"] == 0
    assert report["overall"]["count_mae"] == 0
    assert report["overall"]["matching_score_50"] == 1.0
    assert (output / "per_image_metrics.csv").is_file()
    assert (output / "morphology_features.csv").is_file()
    assert list(output.glob("*_overlay.png")) and list(output.glob("*_mask.png"))

    persisted = json.loads((output / "baseline_report.json").read_text())
    assert "no Tribonema treatment" in " ".join(persisted["limitations"])
    assert "/Users/" not in persisted["manifest"]


def test_the_baseline_is_exact_on_perfectly_separable_frames(
    tiny_training_data: Path, tmp_path: Path
) -> None:
    """At its default blur radius the rule segments flat squares perfectly.

    That is what makes this fixture unsuitable for the acceptance gate, which is
    why tests/test_benchmark.py uses a crowded one instead.
    """
    report = run_baseline(
        tiny_training_data / "manifests" / "test.jsonl", tmp_path / "out", max_images=1
    )
    assert report["overall"]["macro_dice"] == pytest.approx(1.0)
    assert report["overall"]["matching_score_50_95"] == pytest.approx(1.0)


def test_the_instance_metric_is_not_advertised_as_average_precision(
    tiny_training_data: Path, tmp_path: Path
) -> None:
    report = run_baseline(
        tiny_training_data / "manifests" / "test.jsonl", tmp_path / "out", max_images=1
    )
    assert "instance_ap_50_95" not in report["overall"]
    assert "matching_score_50_95" in report["overall"]
    assert "NOT average precision" in report["metric_definitions"]["matching_score_50_95"]


def test_connected_component_limitation_is_stated_in_the_report(
    tiny_training_data: Path, tmp_path: Path
) -> None:
    report = run_baseline(
        tiny_training_data / "manifests" / "test.jsonl", tmp_path / "out", max_images=1
    )
    assert any("touching cells" in line for line in report["limitations"])


def test_baseline_refuses_a_manifest_that_points_outside_the_dataset(
    tiny_training_data: Path, tmp_path: Path
) -> None:
    """The baseline used to resolve manifest paths itself, with no containment."""
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps({"annotations": []}), encoding="utf-8")
    path = rewrite_manifest(
        tiny_training_data,
        "test",
        lambda rows: [{**row, "annotation_path": f"../../{secret.name}"} for row in rows],
    )
    with pytest.raises(ManifestError, match="escapes the data directory"):
        run_baseline(path, tmp_path / "out", max_images=1)


def test_baseline_refuses_an_image_id_that_would_escape_the_output_directory(
    tiny_training_data: Path, tmp_path: Path
) -> None:
    path = rewrite_manifest(
        tiny_training_data, "test", lambda rows: [{**row, "image_id": "../escaped"} for row in rows]
    )
    with pytest.raises(ManifestError):
        run_baseline(path, tmp_path / "out", max_images=1)
    assert not (tmp_path / "escaped_mask.png").exists()


def test_rerunning_into_the_same_directory_removes_stale_artifacts(
    tiny_training_data: Path, tmp_path: Path
) -> None:
    output = tmp_path / "reused"
    output.mkdir()
    stale = output / "999999_mask.png"
    Image.new("L", (4, 4)).save(stale)
    run_baseline(tiny_training_data / "manifests" / "test.jsonl", output, max_images=1)
    assert not stale.exists()


def test_max_images_must_be_at_least_one(tiny_training_data: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_baseline(
            tiny_training_data / "manifests" / "test.jsonl", tmp_path / "out", max_images=0
        )


def test_morphology_csv_records_the_instance_method(
    tiny_training_data: Path, tmp_path: Path
) -> None:
    output = tmp_path / "out"
    run_baseline(
        tiny_training_data / "manifests" / "test.jsonl",
        output,
        max_images=1,
        instance_method="watershed_split",
        background_radius=2.0,
        min_area=2,
    )
    with (output / "morphology_features.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row["instance_method"] == "watershed_split" for row in rows)


def test_mixed_calibrations_are_refused_rather_than_silently_picking_one(
    tiny_training_data: Path, tmp_path: Path
) -> None:
    """Using the first row's scale for every image would misreport the rest."""

    def mutate(rows: list[dict]) -> list[dict]:
        rows[0]["micrometers_per_pixel"] = 0.5
        rows[1]["micrometers_per_pixel"] = 1.25
        return rows

    path = rewrite_manifest(tiny_training_data, "test", mutate)
    with pytest.raises(ValueError, match="mixes different micrometers_per_pixel"):
        run_baseline(path, tmp_path / "out", max_images=2)

    # An explicit override resolves the ambiguity.
    report = run_baseline(path, tmp_path / "out", max_images=2, micrometers_per_pixel=0.75)
    assert report["parameters"]["micrometers_per_pixel"] == 0.75
