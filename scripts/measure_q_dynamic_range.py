"""Is there enough variation between images for a rank correlation to see?

A Spearman correlation between true and recovered shape index measures whether a
segmenter tracks variation across images. If the images barely differ, there is
nothing to track, and the correlation collapses no matter how good the segmenter
is. That is range restriction, and it is a property of the cell line rather than
of the method being tested.

This measures the biological spread directly from the polygon annotations, so it
does not depend on any segmenter.
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

LINES = (("", "A172"), ("_mcf7", "MCF7"), ("_shsy5y", "SHSY5Y"), ("_skbr3", "SkBr3"))


def _ring(segmentation: Any) -> list[float] | None:
    if not isinstance(segmentation, list) or not segmentation:
        return None
    ring = segmentation[0] if isinstance(segmentation[0], list) else segmentation
    if not isinstance(ring, list) or len(ring) < 6:
        return None
    return ring


def per_image_q(manifest: Path) -> list[float]:
    """Median true shape index of every image in a manifest, from polygons only."""
    records = load_manifest(manifest)
    cache: dict[Path, dict[int, dict[str, Any]]] = {}
    medians: list[float] = []
    for record in records:
        if record.annotation_path not in cache:
            source = json.loads(record.annotation_path.read_text(encoding="utf-8"))
            cache[record.annotation_path] = {int(a["id"]): a for a in source["annotations"]}
        index = cache[record.annotation_path]
        values: list[float] = []
        for identifier in record.annotation_ids:
            ring = _ring(index[identifier].get("segmentation"))
            if ring is None:
                continue
            try:
                area, perimeter = mechanics.polygon_area_perimeter(ring)
                values.append(mechanics.shape_index(area, perimeter))
            except mechanics.MechanicsError:
                continue
        if values:
            medians.append(float(np.median(values)))
    return medians


def main() -> None:
    report: dict[str, Any] = {"per_line": {}, "environment": provenance.environment()}
    for key, name in LINES:
        values = np.asarray(per_image_q(Path(f"data/livecell/manifests{key}/test.jsonl")))
        report["per_line"][name] = {
            "images": int(values.size),
            "median_q": float(np.median(values)),
            "sd_between_images": float(values.std(ddof=1)),
            "iqr_between_images": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "range_between_images": float(values.max() - values.min()),
        }
    out = Path("runs/mechanics/q_dynamic_range.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{'line':8s} {'images':>6s} {'median q':>9s} {'sd':>7s} {'IQR':>7s} {'range':>7s}")
    for name, row in report["per_line"].items():
        print(
            f"{name:8s} {row['images']:6d} {row['median_q']:9.3f} "
            f"{row['sd_between_images']:7.4f} {row['iqr_between_images']:7.4f} "
            f"{row['range_between_images']:7.3f}"
        )
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
