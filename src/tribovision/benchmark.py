"""Head-to-head evaluation of the trained model against reference predictors.

A learned segmenter is only worth reporting if it beats predictors that required
no learning at all, on images none of them was tuned on.

Two references are scored, not one, because the classical rule alone is a
flattering comparison. LIVECell A172 frames average 59% foreground, so a
predictor that simply labels **every pixel a cell** scores 0.710 macro Dice on
this test set — far above the 0.425 of the local-contrast rule. Reporting only
the classical number would make the model look better than it is. The trivial
predictor is therefore the floor the verdict is measured against, and the report
states the margin over it explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tribovision import coco, evaluation, provenance
from tribovision.baseline import segment_classical
from tribovision.manifest import acquisition_group, load_manifest, well_group
from tribovision.morphology import label_objects
from tribovision.predict import load_checkpoint, predict_mask


def _annotations_for(
    record: Any, cache: dict[Path, dict[int, dict[str, Any]]]
) -> list[dict[str, Any]]:
    if record.annotation_path not in cache:
        payload = json.loads(record.annotation_path.read_text(encoding="utf-8"))
        cache[record.annotation_path] = {
            int(annotation["id"]): annotation for annotation in payload.get("annotations", [])
        }
    index = cache[record.annotation_path]
    return [index[annotation_id] for annotation_id in record.annotation_ids]


def compare(
    checkpoint: Path,
    manifest_path: Path,
    output_dir: Path | None = None,
    *,
    device: str = "auto",
    threshold: float = 0.5,
    max_images: int | None = None,
    background_radius: float = 7.0,
    min_area: int = 20,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Score both segmenters on the same held-out images and decide the verdict."""
    from tribovision.training import resolve_device

    torch_device = resolve_device(device)
    model, payload = load_checkpoint(Path(checkpoint), torch_device)
    preprocessing = payload.get("preprocessing") or {}
    image_size = int(preprocessing.get("image_size") or 512)

    records = load_manifest(Path(manifest_path), verify_hashes=verify_hashes)
    if max_images is not None:
        records = records[:max_images]
    cache: dict[Path, dict[int, dict[str, Any]]] = {}

    neural_rows: list[dict[str, Any]] = []
    classical_rows: list[dict[str, Any]] = []
    all_foreground_rows: list[dict[str, Any]] = []
    all_background_rows: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    foreground_fractions: list[float] = []

    for record in records:
        annotations = _annotations_for(record, cache)
        true_labels = coco.label_image(annotations, record.width, record.height)
        truth = (true_labels > 0).astype(np.uint8)
        with Image.open(record.image_path) as handle:
            image = handle.convert("L")

        with torch.no_grad():
            neural_mask, _ = predict_mask(
                model, image, image_size=image_size, device=torch_device, threshold=threshold
            )
        classical_mask = segment_classical(
            image, background_radius=background_radius, min_area=min_area
        )

        neural = evaluation.semantic_metrics(neural_mask, truth)
        classical = evaluation.semantic_metrics(classical_mask, truth)
        all_foreground_rows.append(evaluation.semantic_metrics(np.ones_like(truth), truth))
        all_background_rows.append(evaluation.semantic_metrics(np.zeros_like(truth), truth))
        foreground_fractions.append(float(truth.mean()))
        neural_instances = evaluation.matching_score(
            label_objects(neural_mask, method="watershed_split", min_area=min_area), true_labels
        )
        # The same instance step applied to the *perfect* mask. Whatever it scores
        # is the ceiling any semantic segmenter can reach here, and reporting the
        # model's instance number without it invites reading a representation
        # limit as a model failure.
        ceiling_instances = evaluation.matching_score(
            label_objects(truth, method="watershed_split", min_area=min_area), true_labels
        )
        classical_instances = evaluation.matching_score(
            label_objects(classical_mask, method="watershed_split", min_area=min_area),
            true_labels,
        )
        neural_rows.append(neural)
        classical_rows.append(classical)
        per_image.append(
            {
                "image_id": record.image_id,
                "well": record.well,
                "acquisition_group": acquisition_group(record),
                "well_group": well_group(record),
                "neural_dice": neural["dice"],
                "classical_dice": classical["dice"],
                "neural_matching_50_95": neural_instances["mean"],
                "classical_matching_50_95": classical_instances["mean"],
                "ceiling_matching_50_95": ceiling_instances["mean"],
                "true_instances": int(true_labels.max()),
            }
        )

    neural_summary = evaluation.aggregate(neural_rows)
    classical_summary = evaluation.aggregate(classical_rows)
    trivial_summary = evaluation.aggregate(all_foreground_rows)
    empty_summary = evaluation.aggregate(all_background_rows)
    differences = np.array(
        [row["neural_dice"] - row["classical_dice"] for row in per_image], dtype=float
    )
    wins = int(np.count_nonzero(differences > 0))
    n = len(differences)
    # A sign test over images would be pseudoreplication. LIVECell tiles one
    # captured frame into several crops, and this manifest is a time-lapse of a
    # single well, so the images are not independent trials. The reported p-value
    # is therefore computed over acquisition groups: each field of view at each
    # timestamp contributes one trial, decided by its mean Dice difference.
    grouped = _group_differences(per_image, "acquisition_group")
    group_wins = int(sum(1 for value in grouped.values() if value > 0))
    p_value = _sign_test_p_value(group_wins, len(grouped))
    image_level_p = _sign_test_p_value(wins, n)
    # The bar is the *stronger* of the two reference predictors, not the weaker.
    floor = max(classical_summary.get("macro_dice", 0.0), trivial_summary.get("macro_dice", 0.0))
    passed = bool(n > 0 and neural_summary["macro_dice"] > floor)

    report: dict[str, Any] = {
        "manifest": provenance.relative_to_repo(Path(manifest_path)),
        "checkpoint": provenance.relative_to_repo(Path(checkpoint)),
        "images": n,
        "neural": {
            **neural_summary,
            "matching_score_50_95": float(
                np.mean([row["neural_matching_50_95"] for row in per_image])
            )
            if per_image
            else 0.0,
        },
        "classical": {
            **classical_summary,
            "matching_score_50_95": float(
                np.mean([row["classical_matching_50_95"] for row in per_image])
            )
            if per_image
            else 0.0,
        },
        "instance_ceiling": {
            "matching_score_50_95": (
                float(np.mean([row["ceiling_matching_50_95"] for row in per_image]))
                if per_image
                else 0.0
            ),
            "explanation": (
                "The ground-truth mask itself, put through the same watershed instance "
                "step. This is the highest instance score any semantic segmenter can "
                "reach in this pipeline, so the model's instance number should be read "
                "against it and not against 1.0. Raising it needs a model that predicts "
                "instances directly, not more training of this one."
            ),
        },
        "all_foreground": trivial_summary,
        "all_background": empty_summary,
        "mean_foreground_fraction": (
            float(np.mean(foreground_fractions)) if foreground_fractions else 0.0
        ),
        "reference_floor_macro_dice": floor,
        "margin_over_reference_floor": (float(neural_summary["macro_dice"] - floor) if n else 0.0),
        "reference_note": (
            "all_foreground labels every pixel a cell. It requires no learning and "
            "scores well whenever frames are crowded, so it — not the classical rule — "
            "is the floor a learned segmenter has to clear. IoU separates them far more "
            "sharply than Dice does."
        ),
        "dice_difference_mean": float(differences.mean()) if n else 0.0,
        "dice_difference_sd": float(differences.std(ddof=1)) if n > 1 else 0.0,
        "images_where_neural_wins": wins,
        "independent_units": len(grouped),
        "units_where_neural_wins": group_wins,
        "sign_test_p_value": p_value,
        "sign_test_unit": "acquisition group (field of view at one timestamp)",
        "sign_test_p_value_by_image_pseudoreplicated": image_level_p,
        "independence_note": (
            "LIVECell splits one captured frame into several crops and this manifest is "
            "a time-lapse of one well, so images are correlated. The headline p-value "
            "counts each acquisition group once. The per-image value is reported only to "
            "show how much pseudoreplication would have inflated it, and must not be "
            "quoted. Note also that all of these units come from a single well on a "
            "single plate, so at the level of biological replication n = 1."
        ),
        "verdict": (
            "neural beats every reference predictor"
            if passed
            else "neural does NOT clear the reference floor"
        ),
        "passed": passed,
        "per_image": per_image,
        "environment": provenance.environment(),
    }
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "comparison.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def _group_differences(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Average the per-image Dice difference within each independent unit."""
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(
            float(row["neural_dice"]) - float(row["classical_dice"])
        )
    return {name: float(np.mean(values)) for name, values in buckets.items()}


def _sign_test_p_value(wins: int, trials: int) -> float:
    """Two-sided exact binomial p-value for *wins* out of *trials* at p=0.5."""
    if trials == 0:
        return 1.0
    from math import comb

    def tail(k: int) -> float:
        return sum(comb(trials, i) for i in range(k, trials + 1)) / 2**trials

    extreme = max(wins, trials - wins)
    return float(min(1.0, 2 * tail(extreme)))
