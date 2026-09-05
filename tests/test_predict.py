"""Checkpoint round trip: train, save, load, and analyse a brand-new image."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tribovision.predict import (
    PredictionError,
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

    stem = new_image.stem
    assert (output / f"{stem}_mask.png").is_file()
    assert (output / f"{stem}_overlay.png").is_file()
    assert (output / "prediction_report.json").is_file()
    with Image.open(output / f"{stem}_overlay.png") as overlay:
        assert overlay.mode == "RGB"
        assert overlay.size == (40, 24)

    persisted = json.loads((output / "prediction_report.json").read_text())
    assert persisted["images"][0]["source_image"] == new_image.name
    assert "/Users/" not in json.dumps(persisted["checkpoint"])
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
