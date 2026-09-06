"""How well can each approach separate individual cells, not just find them?

`tribovision compare` answers a pixel question. This answers the instance one,
which is what the treatment analysis actually needs: per-cell morphology is only
per-cell if the cells were separated.

Four things are scored on the same held-out images:

* the in-house semantic U-Net, split afterwards with a watershed;
* the classical rule, split the same way;
* Cellpose, which predicts instances directly (optional dependency);
* the **ground-truth mask itself**, split the same way — the ceiling that a
  semantic representation imposes no matter how good the model behind it is.

The ceiling row is the point. Without it, a low instance score reads as a weak
model rather than as a limit of predicting a binary foreground mask.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tribovision import coco, evaluation, external, provenance
from tribovision.baseline import segment_classical
from tribovision.manifest import load_manifest
from tribovision.morphology import label_objects
from tribovision.predict import load_checkpoint, predict_mask


def _score(predicted_labels: np.ndarray, true_labels: np.ndarray) -> dict[str, float]:
    matching = evaluation.matching_score(predicted_labels, true_labels)
    truth = (true_labels > 0).astype(np.uint8)
    semantic = evaluation.semantic_metrics((predicted_labels > 0).astype(np.uint8), truth)
    return {
        "matching_50_95": matching["mean"],
        "matching_50": matching["at_0.50"],
        "matching_75": matching["at_0.75"],
        "dice": semantic["dice"],
        "predicted_objects": int(predicted_labels.max()),
        "true_objects": int(true_labels.max()),
    }


def run(
    manifest_path: Path,
    output_dir: Path | None = None,
    *,
    checkpoint: Path | None = None,
    device: str = "auto",
    max_images: int | None = None,
    min_area: int = 20,
    cellpose_diameter: float | None = 15.0,
    cellpose_model: str = external.DEFAULT_MODEL,
    include_cellpose: bool = True,
) -> dict[str, Any]:
    """Score every available instance approach on one held-out manifest."""
    from tribovision.training import resolve_device

    records = load_manifest(Path(manifest_path))
    if max_images is not None:
        records = records[:max_images]

    model = None
    torch_device = None
    image_size = 512
    if checkpoint is not None:
        torch_device = resolve_device(device)
        model, payload = load_checkpoint(Path(checkpoint), torch_device)
        image_size = int((payload.get("preprocessing") or {}).get("image_size") or 512)

    use_cellpose = include_cellpose and external.cellpose_available()
    per_image: list[dict[str, Any]] = []
    cache: dict[Path, dict[int, dict[str, Any]]] = {}

    for record in records:
        if record.annotation_path not in cache:
            payload_json = json.loads(record.annotation_path.read_text(encoding="utf-8"))
            cache[record.annotation_path] = {
                int(a["id"]): a for a in payload_json.get("annotations", [])
            }
        index = cache[record.annotation_path]
        annotations = [index[i] for i in record.annotation_ids]
        true_labels = coco.label_image(annotations, record.width, record.height)
        truth = (true_labels > 0).astype(np.uint8)
        with Image.open(record.image_path) as handle:
            image = handle.convert("L")
        array = np.asarray(image)

        row: dict[str, Any] = {"image_id": record.image_id, "well": record.well}
        # The ceiling: a perfect mask through the same instance step.
        row["ceiling"] = _score(
            label_objects(truth, method="watershed_split", min_area=min_area), true_labels
        )
        row["classical"] = _score(
            label_objects(
                segment_classical(image, min_area=min_area),
                method="watershed_split",
                min_area=min_area,
            ),
            true_labels,
        )
        if model is not None and torch_device is not None:
            with torch.no_grad():
                mask, _ = predict_mask(model, image, image_size=image_size, device=torch_device)
            row["tribovision_unet"] = _score(
                label_objects(mask, method="watershed_split", min_area=min_area), true_labels
            )
        if use_cellpose:
            row["cellpose"] = _score(
                external.cellpose_instances(
                    array, diameter=cellpose_diameter, model_name=cellpose_model
                ),
                true_labels,
            )
        per_image.append(row)

    methods = sorted({key for row in per_image for key in row if isinstance(row[key], dict)})
    summary = {
        method: {
            metric: float(np.mean([row[method][metric] for row in per_image if method in row]))
            for metric in (
                "matching_50_95",
                "matching_50",
                "matching_75",
                "dice",
                "predicted_objects",
                "true_objects",
            )
        }
        for method in methods
    }
    for values in summary.values():
        values["count_error"] = values["predicted_objects"] - values["true_objects"]

    report: dict[str, Any] = {
        "manifest": provenance.relative_to_repo(Path(manifest_path)),
        "images": len(per_image),
        "min_area": min_area,
        "summary": summary,
        "cellpose": {
            "used": use_cellpose,
            "version": external.cellpose_version(),
            "model": cellpose_model if use_cellpose else None,
            "diameter": cellpose_diameter if use_cellpose else None,
            "trained_on_this_data": False,
            "note": (
                "Cellpose is run zero-shot: no fine-tuning on LIVECell, and its one "
                "tunable knob (diameter) was chosen on the training split, never on the "
                "manifest scored here."
            )
            if use_cellpose
            else "Cellpose is not installed; install the optional 'cellpose' extra.",
        },
        "interpretation": (
            "Read every row against 'ceiling', which is the ground-truth mask put "
            "through the same watershed step. A method that predicts instances directly "
            "is not bound by that ceiling; one that predicts a binary mask is."
        ),
        "per_image": per_image,
        "environment": provenance.environment(),
    }
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "instance_benchmark.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report
