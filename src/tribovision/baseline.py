"""Transparent classical baseline, evaluation, and visual artifacts.

This module intentionally uses simple image processing rather than a learned
model. It gives the project an understandable reference point: a trained model is
worth reporting only if it beats this fixed local-contrast rule on held-out
images that it never saw.

It now shares one validated manifest loader with the neural dataset, so it gets
the same containment and provenance guarantees instead of resolving paths itself,
and it writes output files under names derived through the same filename guard.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from tribovision import coco, evaluation, morphology, provenance
from tribovision.manifest import load_manifest, safe_output_name


def otsu_threshold(values: np.ndarray) -> int:
    """Return Otsu's between-class-variance threshold for a uint8 array."""
    histogram = np.bincount(values.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total == 0:
        return 0
    probabilities = histogram / total
    cumulative_probability = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * np.arange(256))
    global_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1 - cumulative_probability)
    score = np.zeros_like(denominator)
    valid = denominator > 0
    score[valid] = (
        global_mean * cumulative_probability[valid] - cumulative_mean[valid]
    ) ** 2 / denominator[valid]
    return int(np.argmax(score))


def segment_classical(
    image: Image.Image,
    *,
    background_radius: float = 7.0,
    min_area: int = 20,
) -> np.ndarray:
    """Segment high local-contrast regions with Otsu thresholding and cleanup.

    Phase-contrast cells differ from their slowly varying background in both dark
    and bright directions, so absolute deviation from a blurred background is a
    more transparent baseline than choosing one global intensity direction.
    """
    grey = image.convert("L")
    pixels = np.asarray(grey, dtype=np.int16)
    background = np.asarray(
        grey.filter(ImageFilter.GaussianBlur(background_radius)), dtype=np.int16
    )
    deviation = np.clip(np.abs(pixels - background), 0, 255).astype(np.uint8)
    threshold = otsu_threshold(deviation)
    raw = Image.fromarray((deviation > threshold).astype(np.uint8) * 255)
    closed = raw.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    labels = morphology.label_objects(np.asarray(closed) > 0, min_area=min_area)
    return (labels > 0).astype(np.uint8)


# Kept as the module's public labelling helper so callers do not import scipy directly.
connected_components = morphology.connected_components
semantic_metrics = evaluation.semantic_metrics


def _clear_previous_artifacts(output_dir: Path) -> None:
    """Remove artifacts from an earlier run so results cannot be mixed together."""
    if not output_dir.is_dir():
        return
    for pattern in ("*_mask.png", "*_overlay.png", "*.csv", "baseline_report.json"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()
            elif path.is_dir():  # pragma: no cover - defensive
                shutil.rmtree(path)


def run_baseline(
    manifest_path: Path,
    output_dir: Path,
    *,
    max_images: int = 10,
    background_radius: float = 7.0,
    min_area: int = 20,
    instance_method: str = "connected_components",
    micrometers_per_pixel: float | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    if max_images < 1:
        raise ValueError("max_images must be at least 1.")
    manifest_path = Path(manifest_path).resolve()
    records = load_manifest(manifest_path, verify_hashes=verify_hashes)[:max_images]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_artifacts(output_dir)

    calibration = morphology.Calibration(
        micrometers_per_pixel
        if micrometers_per_pixel is not None
        else records[0].micrometers_per_pixel
    )

    annotation_cache: dict[Path, dict[int, dict[str, Any]]] = {}
    result_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for record in records:
        if record.annotation_path not in annotation_cache:
            payload = json.loads(record.annotation_path.read_text(encoding="utf-8"))
            annotation_cache[record.annotation_path] = {
                int(annotation["id"]): annotation for annotation in payload.get("annotations", [])
            }
        index = annotation_cache[record.annotation_path]
        annotations = [index[annotation_id] for annotation_id in record.annotation_ids]

        with Image.open(record.image_path) as source:
            image = source.convert("L")
        width, height = image.size
        if (width, height) != (record.width, record.height):
            raise ValueError(
                f"Image {record.image_path.name} is {width}x{height} but the manifest "
                f"records {record.width}x{record.height}."
            )

        true_labels = coco.label_image(annotations, width, height)
        truth = (true_labels > 0).astype(np.uint8)
        prediction = segment_classical(
            image, background_radius=background_radius, min_area=min_area
        )
        predicted_labels = morphology.label_objects(
            prediction, method=instance_method, min_area=min_area
        )

        pixel_metrics = evaluation.semantic_metrics(prediction, truth)
        instance_metrics = evaluation.matching_score(predicted_labels, true_labels)
        counts = evaluation.instance_counts(predicted_labels, true_labels)
        result_rows.append(
            {
                "image_id": record.image_id,
                "cell_type": record.cell_type,
                "well": record.well,
                "split": record.split,
                "true_count": int(true_labels.max()),
                "predicted_count": int(predicted_labels.max()),
                "count_absolute_error": abs(int(predicted_labels.max()) - int(true_labels.max())),
                "matching_score_50_95": instance_metrics["mean"],
                "matching_score_50": instance_metrics["at_0.50"],
                "instance_tp": counts["tp"],
                "instance_fp": counts["fp"],
                "instance_fn": counts["fn"],
                **pixel_metrics,
            }
        )
        feature_rows.extend(
            morphology.measure(
                predicted_labels,
                np.asarray(image, dtype=np.float32),
                calibration=calibration,
                method=instance_method,
                extra={
                    "image_id": record.image_id,
                    "cell_type": record.cell_type,
                    "well": record.well,
                },
            )
        )

        stem = safe_output_name(record.image_id)
        from tribovision.predict import overlay_image

        overlay_image(image, prediction, truth).save(output_dir / f"{stem}_overlay.png")
        Image.fromarray(prediction * 255).save(output_dir / f"{stem}_mask.png")

    _write_csv(output_dir / "per_image_metrics.csv", result_rows)
    _write_csv(output_dir / "morphology_features.csv", feature_rows)

    report: dict[str, Any] = {
        "method": "absolute local contrast + Otsu threshold + morphological closing",
        "parameters": {
            "background_radius": background_radius,
            "min_area": min_area,
            "instance_method": instance_method,
            "micrometers_per_pixel": calibration.micrometers_per_pixel,
        },
        "manifest": provenance.relative_to_repo(manifest_path),
        "overall": {
            **evaluation.aggregate(result_rows),
            "count_mae": _mean(result_rows, "count_absolute_error"),
            "matching_score_50_95": _mean(result_rows, "matching_score_50_95"),
            "matching_score_50": _mean(result_rows, "matching_score_50"),
        },
        "by_group": _grouped(result_rows),
        "metric_definitions": {
            "dice": "exact 2TP/(2TP+FP+FN); no epsilon smoothing",
            "matching_score_50_95": (
                "mean over IoU thresholds 0.50:0.05:0.95 of TP/(TP+FP+FN) after greedy "
                "one-to-one matching (the 2018 Data Science Bowl metric). This is NOT "
                "average precision: a fixed rule assigns no confidence, so there is no "
                "ranking and no precision-recall curve to integrate."
            ),
        },
        "limitations": [
            "Objects are predicted regions; with connected components, touching cells "
            "merge into one region, so per-object morphology is a region measurement.",
            "LIVECell has no Tribonema treatment, viability, stress, necrosis, or "
            "clinical labels, so nothing here measures a drug effect.",
            "Lengths and areas are in pixels unless micrometers_per_pixel was supplied.",
            "Green pixels in overlays are ground truth; red pixels are baseline predictions.",
        ],
        "environment": provenance.environment(),
    }
    (output_dir / "baseline_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _grouped(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {"cell_type": {}, "well": {}}
    for row in rows:
        for key in grouped:
            grouped[key].setdefault(str(row[key]), []).append(row)
    return {
        key: {
            name: {
                **evaluation.aggregate(values),
                "count_mae": _mean(values, "count_absolute_error"),
                "matching_score_50_95": _mean(values, "matching_score_50_95"),
            }
            for name, values in sorted(members.items())
        }
        for key, members in grouped.items()
    }
