"""Load a trained checkpoint and analyse new microscope images.

This is the step that was missing: the CLI could prepare data and train, but
there was no way to point the trained model at an image, so "we have a trained
segmenter" was an unverifiable claim. ``predict`` closes the loop — checkpoint
in, mask, overlay, per-object morphology and a summary out — and it deliberately
maps predictions back to the native image grid before measuring anything.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tribovision import morphology, provenance
from tribovision.geometry import letterbox_image, normalize_intensity, unletterbox_mask
from tribovision.model import TriboUNet

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


class PredictionError(RuntimeError):
    """Raised when a checkpoint or an input image cannot be used."""


def load_checkpoint(path: Path, device: torch.device) -> tuple[TriboUNet, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise PredictionError(f"Checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:  # torch raises several unrelated types here
        raise PredictionError(f"Could not read checkpoint {path.name}: {exc}") from exc
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise PredictionError(f"{path.name} is not a TriboVision checkpoint.")
    architecture = payload.get("architecture") or {}
    model = TriboUNet(
        base_channels=int(architecture.get("base_channels", 32)),
        depth=int(architecture.get("depth", 3)),
    )
    try:
        model.load_state_dict(payload["model_state"])
    except RuntimeError as exc:
        raise PredictionError(
            f"Checkpoint architecture does not match this model version: {exc}"
        ) from exc
    model.to(device).eval()
    return model, payload


def _image_size(payload: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return override
    preprocessing = payload.get("preprocessing") or {}
    config = payload.get("config") or {}
    return int(preprocessing.get("image_size") or config.get("image_size") or 512)


@torch.no_grad()
def predict_mask(
    model: TriboUNet,
    image: Image.Image,
    *,
    image_size: int,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (native-resolution binary mask, native-resolution probability map)."""
    if not 0.0 < threshold < 1.0:
        raise PredictionError(f"threshold must be in (0, 1), got {threshold}.")
    grey = image.convert("L")
    boxed, transform = letterbox_image(grey, image_size)
    valid = transform.valid_mask()
    pixels = np.asarray(boxed, dtype=np.float32)
    normalized = np.zeros_like(pixels)
    interior = pixels[valid > 0]
    if interior.size:
        normalized[valid > 0] = normalize_intensity(interior)
    tensor = torch.from_numpy(normalized[None, None, ...]).to(device)
    probabilities = torch.sigmoid(model(tensor))[0, 0].cpu().numpy() * valid
    mask = unletterbox_mask((probabilities >= threshold).astype(np.uint8), transform)
    # Probabilities are resampled the same way so that overlays and masks agree.
    probability_native = (
        np.asarray(
            Image.fromarray((probabilities * 255).astype(np.uint8))
            .crop(
                (
                    transform.pad_left,
                    transform.pad_top,
                    transform.pad_left + transform.resized_width,
                    transform.pad_top + transform.resized_height,
                )
            )
            .resize(
                (transform.original_width, transform.original_height), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    return mask, probability_native


def overlay_image(
    image: Image.Image, prediction: np.ndarray, truth: np.ndarray | None = None
) -> Image.Image:
    """Red = prediction, green = ground truth when available."""
    grey = np.asarray(image.convert("L"))
    canvas = np.repeat(grey[..., None], 3, axis=2).astype(np.float32)
    if truth is not None:
        selection = np.asarray(truth).astype(bool)
        canvas[selection] = 0.45 * canvas[selection] + 0.55 * np.array([50, 220, 80])
    selection = np.asarray(prediction).astype(bool)
    canvas[selection] = 0.45 * canvas[selection] + 0.55 * np.array([240, 60, 60])
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def collect_images(inputs: Sequence[Path]) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        item = Path(item)
        if item.is_dir():
            found.extend(
                sorted(p for p in item.rglob("*") if p.suffix.casefold() in IMAGE_SUFFIXES)
            )
        elif item.is_file():
            found.append(item)
        else:
            raise PredictionError(f"Input does not exist: {item}")
    if not found:
        raise PredictionError("No readable images were found in the given inputs.")
    return found


def run_prediction(
    checkpoint: Path,
    inputs: Iterable[Path],
    output_dir: Path,
    *,
    device: str = "auto",
    threshold: float = 0.5,
    image_size: int | None = None,
    instance_method: str = "watershed_split",
    min_area: int = 20,
    micrometers_per_pixel: float | None = None,
    save_overlays: bool = True,
) -> dict[str, Any]:
    """Segment every input image and write masks, overlays, and morphology."""
    from tribovision.training import resolve_device

    torch_device = resolve_device(device)
    model, payload = load_checkpoint(Path(checkpoint), torch_device)
    size = _image_size(payload, image_size)
    model.check_input_size(size, size)
    calibration = morphology.Calibration(micrometers_per_pixel)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_images(list(inputs))

    per_image: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            with Image.open(path) as handle:
                handle.load()
                image = handle.convert("L")
        except (OSError, ValueError) as exc:
            raise PredictionError(f"Cannot read image {path}: {exc}") from exc
        mask, probabilities = predict_mask(
            model, image, image_size=size, device=torch_device, threshold=threshold
        )
        labels = morphology.label_objects(mask, method=instance_method, min_area=min_area)
        rows = morphology.measure(
            labels,
            np.asarray(image, dtype=np.float32),
            calibration=calibration,
            method=instance_method,
            extra={"source_image": path.name},
        )
        feature_rows.extend(rows)
        stem = path.stem
        Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / f"{stem}_mask.png")
        if save_overlays:
            overlay_image(image, mask).save(output_dir / f"{stem}_overlay.png")
        per_image.append(
            {
                "source_image": path.name,
                "width": image.width,
                "height": image.height,
                "foreground_fraction": float(mask.mean()),
                "mean_probability": float(probabilities.mean()),
                **morphology.summarise_image(rows),
            }
        )

    if feature_rows:
        fieldnames = sorted({key for row in feature_rows for key in row})
        with (output_dir / "morphology_features.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(feature_rows)

    report = {
        "checkpoint": provenance.relative_to_repo(Path(checkpoint)),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_validation": payload.get("val_metrics"),
        "device": str(torch_device),
        "threshold": threshold,
        "image_size": size,
        "instance_method": instance_method,
        "min_area": min_area,
        "micrometers_per_pixel": micrometers_per_pixel,
        "images": per_image,
        "objects": len(feature_rows),
        "units_note": (
            "Lengths and areas are in pixels unless micrometers_per_pixel was supplied. "
            f"A perfectly round digitised object scores about "
            f"{morphology.DIGITISED_DISC_CIRCULARITY} on the circularity estimator used here."
        ),
        "interpretation_note": (
            "Objects are predicted regions. They are evidence about morphology, not a "
            "measurement of viability; pair them with an independent assay."
        ),
        "environment": provenance.environment(),
    }
    (output_dir / "prediction_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
