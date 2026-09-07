"""How large must a cell be, in pixels, before its shape index is measurable?

Written after the MCF7 shape-index recovery came back far worse than A172's, to
test the first explanation that has to be ruled out: that the Cellpose diameter
was tuned on A172 and simply does not fit the other lines. It does not survive
contact with the data -- the tuned diameter is a poor match for A172 and a good
match for the lines that fail -- which leaves cell size in pixels, and the
perimeter discretisation error that follows from it, as the candidate.

This is exploratory. It was written after seeing a result, and it is labelled as
such in docs/PRE_SPECIFICATION.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, "src")

from tribovision import mechanics, provenance  # noqa: E402
from tribovision.manifest import load_manifest  # noqa: E402

#: The diameter passed to Cellpose everywhere, chosen on four A172 training
#: images and never re-tuned. Whether that was fair to the other lines is
#: exactly what the ``tuned_diameter_ratio`` column answers.
TUNED_DIAMETER = 15.0
LINES = (("", "A172"), ("_mcf7", "MCF7"), ("_shsy5y", "SHSY5Y"), ("_skbr3", "SkBr3"))


def _ring(segmentation: Any) -> list[float] | None:
    """The outer ring of a COCO polygon, or None for RLE and degenerate rings."""
    if not isinstance(segmentation, list) or not segmentation:
        return None
    ring = segmentation[0] if isinstance(segmentation[0], list) else segmentation
    if not isinstance(ring, list) or len(ring) < 6:
        return None
    return ring


def measure_line(manifest: Path) -> dict[str, Any]:
    records = load_manifest(manifest)
    cache: dict[Path, dict[int, dict[str, Any]]] = {}
    radii: list[float] = []
    residuals: list[float] = []
    resolvable: list[bool] = []
    for record in records:
        if record.annotation_path not in cache:
            source = json.loads(record.annotation_path.read_text(encoding="utf-8"))
            cache[record.annotation_path] = {int(a["id"]): a for a in source["annotations"]}
        index = cache[record.annotation_path]
        for identifier in record.annotation_ids:
            ring = _ring(index[identifier].get("segmentation"))
            if ring is None:
                continue
            try:
                area, perimeter = mechanics.polygon_area_perimeter(ring)
                q = mechanics.shape_index(area, perimeter)
            except mechanics.MechanicsError:
                continue
            radii.append(float(np.sqrt(area / np.pi)))
            residuals.append(mechanics.discretisation_residual(area))
            resolvable.append(mechanics.resolvable_near_threshold(area, q))
    median_radius = float(np.median(radii))
    return {
        "images": len(records),
        "cells": len(radii),
        "median_radius_px": median_radius,
        "median_diameter_px": 2.0 * median_radius,
        "median_discretisation_residual": float(np.median(residuals)),
        # >1 means the tuned diameter is larger than this line's cells.
        "tuned_diameter_ratio": TUNED_DIAMETER / (2.0 * median_radius),
        "resolvable_near_threshold_fraction": float(np.mean(resolvable)),
    }


def main() -> None:
    report: dict[str, Any] = {
        "tuned_diameter_px": TUNED_DIAMETER,
        "note": (
            "resolvable_near_threshold and discretisation_residual both predate this "
            "experiment; they were written for the A172 analysis and are applied here "
            "unchanged."
        ),
        "per_line": {
            name: measure_line(Path(f"data/livecell/manifests{key}/test.jsonl"))
            for key, name in LINES
        },
        "environment": provenance.environment(),
    }
    out = Path("runs/mechanics/resolution_limit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"{'line':8s} {'cells':>7s} {'diam':>8s} {'residual':>9s} "
        f"{'d=15 fit':>9s} {'resolvable':>11s}"
    )
    for name, row in report["per_line"].items():
        print(
            f"{name:8s} {row['cells']:7d} {row['median_diameter_px']:7.1f}px "
            f"{row['median_discretisation_residual'] * 100:8.1f}% "
            f"{row['tuned_diameter_ratio']:8.2f}x "
            f"{row['resolvable_near_threshold_fraction'] * 100:10.1f}%"
        )
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
