"""Does a segmenter find dying cells as reliably as healthy ones?

Segmenters are trained on healthy cultures because that is what annotated
datasets contain. If they find damaged cells less often, every viability measure
built on segmentation carries a bias with a direction: it under-reports damage,
making treatments look weaker than they are.

The design is paired inside each image. Every annotated cell is assigned by coin
flip to damaged or untouched, both arms share the same frame, and detection is
read against the hand-drawn outlines that come with the dataset. Any difference
between the arms therefore cannot be illumination, focus or confluence, because
both arms have the same ones.

Two segmenters are measured, because the claim is about the class of methods and
not about one model: the three-class network trained here, and Cellpose, which
was trained by somebody else on far more data and is the strongest general
option available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tribovision import coco, damage, external, provenance
from tribovision.instance_model import load_instance_checkpoint, predict_instances
from tribovision.manifest import load_manifest

CHECKPOINT = Path("runs/scale_304/best_model.pt")
MANIFEST = Path("data/livecell/manifests/test.jsonl")
SEVERITIES = (0.0, 0.2, 0.4, 0.6, 0.8)
IMAGES = 20
CELLPOSE_DIAMETER = 15.0
INTERIOR_THRESHOLD = 0.7


def _labels_for(record: Any, cache: dict[Path, dict[int, Any]]) -> np.ndarray:
    if record.annotation_path not in cache:
        payload = json.loads(record.annotation_path.read_text(encoding="utf-8"))
        cache[record.annotation_path] = {int(a["id"]): a for a in payload["annotations"]}
    index = cache[record.annotation_path]
    return coco.label_image([index[i] for i in record.annotation_ids], record.width, record.height)


def main() -> None:
    import warnings

    warnings.filterwarnings("ignore")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_instance_checkpoint(CHECKPOINT, device)
    records = load_manifest(MANIFEST)[:IMAGES]
    cache: dict[Path, dict[int, Any]] = {}

    rows: list[dict[str, Any]] = []
    for severity in SEVERITIES:
        # One generator per severity, seeded identically, so the same cells land
        # in the damaged arm at every severity. Re-drawing the split would let
        # cell selection vary along the axis being measured.
        totals = {
            name: {"damaged": 0, "intact": 0, "found_damaged": 0, "found_intact": 0}
            for name in ("three_class", "cellpose")
        }
        for position, record in enumerate(records):
            truth = _labels_for(record, cache)
            rng = np.random.default_rng(1000 + position)
            damaged, intact = damage.split_cells(truth, fraction=0.5, rng=rng)
            if not damaged or not intact:
                continue
            with Image.open(record.image_path) as handle:
                grey = np.asarray(handle.convert("L"), dtype=np.float32)
            degraded = damage.apply_damage(
                grey, truth, damaged, damage.Damage.at(severity), rng=rng
            )
            picture = Image.fromarray(degraded.astype(np.uint8))

            predictions = {
                "three_class": predict_instances(
                    model,
                    picture,
                    image_size=512,
                    device=device,
                    interior_threshold=INTERIOR_THRESHOLD,
                    min_area=20,
                ),
                "cellpose": external.cellpose_instances(
                    np.asarray(picture), diameter=CELLPOSE_DIAMETER
                ),
            }
            for name, predicted in predictions.items():
                found = damage.detected_cells(predicted, truth)
                totals[name]["damaged"] += len(damaged)
                totals[name]["intact"] += len(intact)
                totals[name]["found_damaged"] += len(found & damaged)
                totals[name]["found_intact"] += len(found & intact)

        for name, total in totals.items():
            damaged_rate = total["found_damaged"] / total["damaged"] if total["damaged"] else None
            intact_rate = total["found_intact"] / total["intact"] if total["intact"] else None
            row = {
                "severity": severity,
                "segmenter": name,
                **total,
                "detection_damaged": damaged_rate,
                "detection_intact": intact_rate,
                "detection_gap": (
                    None
                    if damaged_rate is None or intact_rate is None
                    else intact_rate - damaged_rate
                ),
            }
            if damaged_rate is not None and intact_rate is not None:
                # What a researcher would write down if half the cells died.
                row["apparent_kill_when_half_die"] = damage.under_reporting(
                    intact_rate, damaged_rate, 0.5
                )
            rows.append(row)
            print(
                f"  severity {severity:.1f}  {name:12s} "
                f"intact {intact_rate:.3f}  damaged {damaged_rate:.3f}  "
                f"gap {intact_rate - damaged_rate:+.3f}",
                flush=True,
            )

    report = {
        "design": {
            "images": len(records),
            "severities": list(SEVERITIES),
            "split": "each annotated cell assigned to damaged or intact by coin flip",
            "detection_rule": (
                f"a cell is found when one predicted object covers at least "
                f"{damage.DETECTION_COVERAGE:.0%} of it"
            ),
            "checkpoint": provenance.relative_to_repo(CHECKPOINT),
        },
        "rows": rows,
        "caveat": (
            "Synthetic damage is not real damage. What is measured is how a segmenter "
            "responds to a controlled loss of contrast, edge definition and interior "
            "texture. Whether dying cells in a real plate degrade along the same axes is "
            "what a real experiment would settle."
        ),
        "environment": provenance.environment(),
    }
    out = Path("runs/damage/blindness.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
