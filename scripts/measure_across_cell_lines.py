"""Does the shape-index recovery survive a change of cell line?

The original measurement was made on A172 alone. This repeats it on MCF7,
SHSY5Y and SkBr3 against polygon-derived truth, with the rasterised ground truth
included as a positive control: where the control fails, the limit is the
measurement chain rather than the segmenter, and the result must be read that
way.

The rho > 0.7 bar is not set here. It predates the three-class model and is
applied unchanged, which is the only reason clearing it means anything.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tribovision import coco, external, mechanics, provenance
from tribovision.instance_model import load_instance_checkpoint, predict_instances
from tribovision.manifest import load_manifest

CHECKPOINT = Path("runs/scale_304/best_model.pt")
#: Tuned on four A172 training images and deliberately not re-tuned per line;
#: scripts/measure_resolution_limit.py reports how well it fits each line.
CELLPOSE_DIAMETER = 15.0
INTERIOR_THRESHOLD = 0.7
MIN_AREA = 20
LINES = (("mcf7", "MCF7"), ("shsy5y", "SHSY5Y"), ("skbr3", "SkBr3"))


def _median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def measure_line(key: str, model: Any, device: torch.device) -> dict[str, Any]:
    records = load_manifest(Path(f"data/livecell/manifests_{key}/test.jsonl"))
    cache: dict[Path, dict[str, Any]] = {}
    truth: list[float | None] = []
    raster: list[float | None] = []
    cellpose: list[float | None] = []
    three_class: list[float | None] = []
    for record in records:
        if record.annotation_path not in cache:
            cache[record.annotation_path] = json.loads(
                record.annotation_path.read_text(encoding="utf-8")
            )
        index = {int(a["id"]): a for a in cache[record.annotation_path]["annotations"]}
        annotations = [index[i] for i in record.annotation_ids]

        polygon = [
            q
            for a in annotations
            if (q := mechanics.polygon_shape_index(a.get("segmentation"))) is not None
        ]
        truth.append(_median_or_none(polygon))

        labels = coco.label_image(annotations, record.width, record.height)
        raster.append(_median_or_none(mechanics.measure_labels(labels)))

        with Image.open(record.image_path) as handle:
            image = handle.convert("L")
        predicted = external.cellpose_instances(np.asarray(image), diameter=CELLPOSE_DIAMETER)
        cellpose.append(_median_or_none(mechanics.measure_labels(predicted)))

        decoded = predict_instances(
            model,
            image,
            image_size=512,
            device=device,
            interior_threshold=INTERIOR_THRESHOLD,
            min_area=MIN_AREA,
        )
        three_class.append(_median_or_none(mechanics.measure_labels(decoded)))

    measured = [t for t in truth if t is not None]
    return {
        "images": len(records),
        "median_true_q": float(np.median(measured)),
        "ground_truth_raster": mechanics.agreement(truth, raster),
        "cellpose": mechanics.agreement(truth, cellpose),
        "three_class": mechanics.agreement(truth, three_class),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_instance_checkpoint(CHECKPOINT, device)
    report: dict[str, Any] = {
        "checkpoint": provenance.relative_to_repo(CHECKPOINT),
        "cellpose_diameter": CELLPOSE_DIAMETER,
        "interior_threshold": INTERIOR_THRESHOLD,
    }
    for key, name in LINES:
        row = measure_line(key, model, device)
        row["cell_line"] = name
        report[key] = row
        print(
            f"{name}: true median q {row['median_true_q']:.3f} "
            f"| control rho {row['ground_truth_raster']['spearman_rho']:.3f} "
            f"| cellpose rho {row['cellpose']['spearman_rho']:.3f} "
            f"| three-class rho {row['three_class']['spearman_rho']:.3f}",
            flush=True,
        )
    report["environment"] = provenance.environment()
    out = Path("runs/mechanics/across_cell_lines.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
