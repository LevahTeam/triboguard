"""Checkpoint round trip: train, save, load, and analyse a brand-new image."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tribovision.predict import (
    PredictionError,
    artifact_stem,
    collect_images,
    load_checkpoint,
    overlay_image,
    predict_mask,
    run_prediction,
)
from tribovision.training import TrainConfig, train

SMALL = {"image_size": 32, "base_channels": 4, "depth": 2, "batch_size": 2, "device": "cpu"}


@pytest.fixture
def trained_checkpoint(tmp_path: Path, tiny_training_data: Path) -> Path:
    train(
        TrainConfig(
            data_dir=tiny_training_data,
            output_dir=tmp_path / "run",
            epochs=25,
            augment=False,
            learning_rate=5e-3,
            **SMALL,
        ),
        progress=False,
    )
    return tmp_path / "run" / "best_model.pt"


@pytest.fixture
def new_image(tmp_path: Path) -> Path:
    """An image the model has never seen, in the same style as the training data."""
    pixels = np.full((24, 40), 30, dtype=np.uint8)
    pixels[6:16, 8:20] = 220
    path = tmp_path / "unseen" / "A172_Phase_Z9_1_01d00h00m_1.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    return path


def test_checkpoint_loads_and_segments_a_new_image(
    trained_checkpoint: Path, new_image: Path
) -> None:
    model, payload = load_checkpoint(trained_checkpoint, torch.device("cpu"))
    assert payload["architecture"]["normalization"] == "GroupNorm"
    with Image.open(new_image) as handle:
        image = handle.convert("L")
    mask, probabilities = predict_mask(model, image, image_size=32, device=torch.device("cpu"))
    # The mask comes back on the native grid, not the padded square.
    assert mask.shape == (24, 40)
    assert probabilities.shape == (24, 40)
    assert mask.max() == 1
    # The bright block is where the cells are; the model should mostly find it.
    assert mask[6:16, 8:20].mean() > mask[0:4, 0:4].mean()


def test_full_prediction_run_writes_masks_overlays_and_morphology(
    trained_checkpoint: Path, new_image: Path, tmp_path: Path
) -> None:
    output = tmp_path / "predictions"
    report = run_prediction(trained_checkpoint, [new_image], output, device="cpu")

    # The artifact name carries a digest of the source path, so two inputs with
    # the same file name in different directories cannot overwrite each other.
    stem = artifact_stem(new_image)
    assert stem.startswith(new_image.stem)
    assert (output / f"{stem}_mask.png").is_file()
    assert (output / f"{stem}_overlay.png").is_file()
    assert (output / "prediction_report.json").is_file()
    with Image.open(output / f"{stem}_overlay.png") as overlay:
        assert overlay.mode == "RGB"
        assert overlay.size == (40, 24)

    persisted = json.loads((output / "prediction_report.json").read_text())
    assert persisted["images"][0]["source_image"] == new_image.name
    assert "/Users/" not in json.dumps(persisted)
    assert not Path(persisted["images"][0]["source_path"]).is_absolute()
    assert "does not by itself" not in persisted["units_note"]
    assert "not a measurement of viability" in persisted["interpretation_note"]
    assert report["objects"] >= 1

    with (output / "morphology_features.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and "area_pixels" in rows[0]
    # No calibration was supplied, so no micrometre columns may appear.
    assert not any(key.endswith("_um") or key.endswith("_um2") for key in rows[0])


def test_supplying_a_calibration_adds_physical_units(
    trained_checkpoint: Path, new_image: Path, tmp_path: Path
) -> None:
    output = tmp_path / "calibrated"
    run_prediction(
        trained_checkpoint, [new_image], output, device="cpu", micrometers_per_pixel=0.62
    )
    with (output / "morphology_features.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and "area_um2" in rows[0]
    assert float(rows[0]["area_um2"]) == pytest.approx(
        float(rows[0]["area_pixels"]) * 0.62**2, rel=1e-6
    )


def test_a_directory_of_images_is_processed(
    trained_checkpoint: Path, new_image: Path, tmp_path: Path
) -> None:
    report = run_prediction(trained_checkpoint, [new_image.parent], tmp_path / "dir", device="cpu")
    assert len(report["images"]) == 1


def test_overlapping_inputs_are_processed_once(new_image: Path) -> None:
    found = collect_images([new_image.parent, new_image, new_image.parent])
    assert found == [new_image.resolve()]


def test_missing_and_empty_inputs_are_reported(tmp_path: Path) -> None:
    with pytest.raises(PredictionError, match="does not exist"):
        collect_images([tmp_path / "absent.tif"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PredictionError, match="No readable images"):
        collect_images([empty])


def test_a_missing_or_bogus_checkpoint_is_reported(tmp_path: Path) -> None:
    with pytest.raises(PredictionError, match="does not exist"):
        load_checkpoint(tmp_path / "nope.pt", torch.device("cpu"))
    bogus = tmp_path / "bogus.pt"
    torch.save({"something": 1}, bogus)
    with pytest.raises(PredictionError, match="not a TriboVision checkpoint"):
        load_checkpoint(bogus, torch.device("cpu"))


def test_a_checkpoint_from_a_different_architecture_is_reported(
    trained_checkpoint: Path, tmp_path: Path
) -> None:
    payload = torch.load(trained_checkpoint, map_location="cpu", weights_only=True)
    payload["architecture"] = {**payload["architecture"], "base_channels": 16}
    mismatched = tmp_path / "mismatch.pt"
    torch.save(payload, mismatched)
    with pytest.raises(PredictionError, match="does not match this model version"):
        load_checkpoint(mismatched, torch.device("cpu"))


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.2, 1.5])
def test_invalid_thresholds_are_rejected(
    trained_checkpoint: Path, new_image: Path, threshold: float
) -> None:
    model, _ = load_checkpoint(trained_checkpoint, torch.device("cpu"))
    with Image.open(new_image) as handle:
        image = handle.convert("L")
    with pytest.raises(PredictionError, match="threshold"):
        predict_mask(model, image, image_size=32, device=torch.device("cpu"), threshold=threshold)


def test_overlay_marks_prediction_and_truth_in_different_colours() -> None:
    image = Image.new("L", (10, 10), 100)
    prediction = np.zeros((10, 10), dtype=bool)
    prediction[0:3, 0:3] = True
    truth = np.zeros((10, 10), dtype=bool)
    truth[6:9, 6:9] = True
    pixels = np.asarray(overlay_image(image, prediction, truth))
    assert pixels[1, 1][0] > pixels[1, 1][1]  # prediction is red-dominant
    assert pixels[7, 7][1] > pixels[7, 7][0]  # truth is green-dominant


class _EdgeSpikeModel(torch.nn.Module):
    """Confident only in the final valid column of the letterboxed canvas.

    A crop that is one pixel short drops that column entirely, so the signal
    disappears rather than shifting slightly — which a shape check, and even a
    smooth ramp, would both miss.
    """

    def __init__(self, spike_column: int) -> None:
        super().__init__()
        self.spike_column = spike_column

    def check_input_size(self, height: int, width: int) -> None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full_like(x, -20.0)
        logits[..., self.spike_column] = 20.0
        return logits


def test_the_probability_map_crop_keeps_the_last_valid_column(new_image: Path) -> None:
    """Guards an off-by-one in the probability crop that no shape check can see."""
    from tribovision.geometry import plan_letterbox

    with Image.open(new_image) as handle:
        image = handle.convert("L")
    size = 64
    transform = plan_letterbox(image.width, image.height, size)
    assert transform.pad_top > 0  # the fixture really is letterboxed
    last_valid = transform.pad_left + transform.resized_width - 1

    _, probabilities = predict_mask(
        _EdgeSpikeModel(last_valid), image, image_size=size, device=torch.device("cpu")
    )
    assert probabilities.shape == (image.height, image.width)
    # The spike must survive into the native map, and land at its right edge.
    column_means = probabilities.mean(axis=0)
    assert int(np.argmax(column_means)) == image.width - 1
    assert column_means[-1] > 0.5
    # Everything left of the resampled spike stays background.
    assert column_means[: image.width - 4].max() < 0.1


def test_the_probability_map_crop_keeps_the_first_valid_column(new_image: Path) -> None:
    from tribovision.geometry import plan_letterbox

    with Image.open(new_image) as handle:
        image = handle.convert("L")
    size = 64
    transform = plan_letterbox(image.width, image.height, size)
    _, probabilities = predict_mask(
        _EdgeSpikeModel(transform.pad_left), image, image_size=size, device=torch.device("cpu")
    )
    column_means = probabilities.mean(axis=0)
    assert int(np.argmax(column_means)) == 0
    assert column_means[0] > 0.5
    assert column_means[4:].max() < 0.1


class _RampModel(torch.nn.Module):
    """Emits a horizontal ramp in logit space: strongly negative left, positive right.

    Deterministic and asymmetric, so any crop or resize error in the probability
    path shows up as a shifted ramp rather than being hidden by a shape check.
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def check_input_size(self, height: int, width: int) -> None:  # pragma: no cover
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        columns = torch.linspace(-8.0, 8.0, x.shape[-1])
        return columns.view(1, 1, 1, -1).expand_as(x).clone()


def test_the_probability_map_is_aligned_with_the_mask_not_merely_the_right_shape(
    new_image: Path,
) -> None:
    """A one-pixel crop error would misalign every overlay and no shape check sees it."""
    from tribovision.geometry import plan_letterbox

    with Image.open(new_image) as handle:
        image = handle.convert("L")
    size = 64
    mask, probabilities = predict_mask(
        _RampModel(size), image, image_size=size, device=torch.device("cpu")
    )
    assert probabilities.shape == (image.height, image.width)

    # The ramp is monotonically increasing left to right, and un-letterboxing
    # must preserve that: column means strictly increase across the native width.
    column_means = probabilities.mean(axis=0)
    assert np.all(np.diff(column_means) > -1e-6)
    # The two ends must reach the extremes of the ramp, which pins the crop
    # boundaries: a crop that is one pixel short leaves the last column short of 1.
    assert column_means[0] < 0.02
    assert column_means[-1] > 0.98

    transform = plan_letterbox(image.width, image.height, size)
    assert transform.pad_top > 0  # the fixture really is letterboxed
    # And the mask must agree with the probability map at the 0.5 threshold.
    assert (
        np.array_equal(mask.astype(bool), probabilities >= 0.5)
        or np.mean(mask.astype(bool) == (probabilities >= 0.5)) > 0.98
    )


def test_probabilities_and_mask_agree_on_a_uniform_prediction(new_image: Path) -> None:
    class _Constant(torch.nn.Module):
        def check_input_size(self, height: int, width: int) -> None:
            return None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x, 10.0)

    with Image.open(new_image) as handle:
        image = handle.convert("L")
    mask, probabilities = predict_mask(
        _Constant(), image, image_size=64, device=torch.device("cpu")
    )
    assert mask.all()
    assert probabilities.min() > 0.9


def test_two_images_with_the_same_name_do_not_overwrite_each_other(
    trained_checkpoint: Path, new_image: Path, tmp_path: Path
) -> None:
    """The collision that silently lost a result.

    Previously both inputs wrote "<stem>_mask.png", so the second overwrote the
    first and the morphology rows -- keyed on the bare file name -- could not say
    which image they came from. A batch spanning two plates with the same field
    naming would have lost half its outputs without an error.
    """
    first_dir, second_dir = tmp_path / "plate1", tmp_path / "plate2"
    for directory in (first_dir, second_dir):
        directory.mkdir()
        shutil.copy(new_image, directory / "field1.tif")

    output = tmp_path / "predictions"
    report = run_prediction(
        trained_checkpoint,
        [first_dir / "field1.tif", second_dir / "field1.tif"],
        output,
        device="cpu",
    )

    masks = sorted(path.name for path in output.glob("*_mask.png"))
    assert len(masks) == 2, f"one image overwrote the other: {masks}"
    assert len({row["artifact_stem"] for row in report["images"]}) == 2
    # Both keep the readable stem, and both record where they came from.
    assert all(name.startswith("field1_") for name in masks)
    assert len({row["source_id"] for row in report["images"]}) == 2
    assert all(not Path(row["source_path"]).is_absolute() for row in report["images"])


def test_rerun_removes_artifacts_from_images_no_longer_requested(
    trained_checkpoint: Path, new_image: Path, tmp_path: Path
) -> None:
    second = tmp_path / "second.tif"
    shutil.copy(new_image, second)
    output = tmp_path / "predictions"
    run_prediction(trained_checkpoint, [new_image, second], output, device="cpu")
    assert len(list(output.glob("*_mask.png"))) == 2

    run_prediction(trained_checkpoint, [new_image], output, device="cpu")
    assert len(list(output.glob("*_mask.png"))) == 1
    assert len(list(output.glob("*_overlay.png"))) == 1


def test_the_artifact_name_is_stable_across_runs(tmp_path: Path) -> None:
    """A name that depended on the batch would break reproducibility."""
    path = tmp_path / "a" / "field1.tif"
    path.parent.mkdir()
    path.write_bytes(b"")
    assert artifact_stem(path) == artifact_stem(path)


def test_the_artifact_name_differs_for_the_same_name_elsewhere(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert artifact_stem(tmp_path / "a" / "field1.tif") != artifact_stem(
        tmp_path / "b" / "field1.tif"
    )
