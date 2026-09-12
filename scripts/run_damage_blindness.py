"""Does a segmenter find dying cells as reliably as healthy ones?

Segmenters are trained on healthy cultures because that is what annotated
datasets contain. If they find damaged cells less often, every viability measure
built on segmentation carries a bias -- and the direction depends on the readout.
A morphology average is diluted, because the cells carrying the signal drop out
of it. A cell count is inflated, because a cell too faint to see is
indistinguishable from one that has gone.

The design is paired inside each image. Every annotated cell is assigned by coin
flip to damaged or untouched, both arms share the same frame, and detection is
read against the hand-drawn outlines that come with the dataset. Any difference
between the arms therefore cannot be illumination, focus or confluence, because
both arms have the same ones.

**The unit of analysis is the acquisition group, not the cell.** An earlier
version pooled every cell in every image into one rate and reported it as though
those cells were independent trials. They are not: LIVECell tiles one captured
frame into several crops and a manifest is a time-lapse of one well, so cells in
a frame share everything the frame has. The primary result here is therefore a
*per-image paired difference*, summarised with a bootstrap that resamples whole
acquisition groups -- the unit the segmentation benchmark in this project
already uses for exactly this reason.

Three segmenters are measured, because the claim is about the class of methods
and not about one model: the three-class network trained here; a binary U-Net
whose mask is split into cells by watershed, which is the pipeline most
published microscopy analysis still uses; and Cellpose, which was trained by
somebody else on far more data and is the strongest general option available.
The first two share a network body and differ in what they predict, so they are
not fully independent of each other -- Cellpose is the one that is. Several cell
lines are measured for the same reason.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tribovision import coco, damage, evaluation, external, preregistration, provenance
from tribovision.instance_model import load_instance_checkpoint, predict_instances
from tribovision.manifest import acquisition_group, load_manifest
from tribovision.morphology import label_objects
from tribovision.predict import load_checkpoint, predict_mask

CHECKPOINT = Path("runs/scale_304/best_model.pt")
#: The binary model scored as ``tribovision_unet`` in the instance benchmark,
#: through the same watershed step, so its row there and its row here describe
#: the same method.
UNET_CHECKPOINT = Path("runs/baseline/best_model.pt")
MIN_AREA = 20
SEGMENTERS = ("three_class", "unet_watershed", "cellpose")

#: The criteria this experiment is judged against, written before it ran. Its
#: hash is taken at the *start* of a run -- a plan edited mid-run must not end up
#: vouching for the run -- and recorded in the artifact, so the evaluation can
#: refuse any result scored under a plan that was changed afterwards.
PREREGISTRATION = Path("preregistration/damage_blindness_v1.json")
SEVERITIES = (0.0, 0.2, 0.4, 0.6, 0.8)
IMAGES = 20
CELLPOSE_DIAMETER = 15.0
INTERIOR_THRESHOLD = 0.7

#: Detection is a threshold on a continuous overlap, so the threshold is swept
#: rather than chosen. 0.5 stays the headline; the others say whether the
#: finding depends on it.
COVERAGE_THRESHOLDS = (0.3, 0.5, 0.7)

#: One manifest per cell line. A finding measured on a single line is a claim
#: about that line and nothing wider.
MANIFESTS = {
    "A172": Path("data/livecell/manifests/test.jsonl"),
    "MCF7": Path("data/livecell/manifests_mcf7/test.jsonl"),
    "SHSY5Y": Path("data/livecell/manifests_shsy5y/test.jsonl"),
    "SkBr3": Path("data/livecell/manifests_skbr3/test.jsonl"),
}

#: Damage axes. The combined sweep is the headline, because a dying cell fades,
#: softens and vacuolates at once; the single-axis runs say which of those a
#: segmenter is actually losing.
AXES = ("combined", "fade", "blur", "vacuoles")


def _damage_for(axis: str, severity: float) -> damage.Damage:
    if axis == "combined":
        return damage.Damage.at(severity)
    return damage.Damage(**{axis: severity})


def _labels_for(record: Any, cache: dict[Path, dict[int, Any]]) -> np.ndarray:
    if record.annotation_path not in cache:
        payload = json.loads(record.annotation_path.read_text(encoding="utf-8"))
        cache[record.annotation_path] = {int(a["id"]): a for a in payload["annotations"]}
    index = cache[record.annotation_path]
    return coco.label_image([index[i] for i in record.annotation_ids], record.width, record.height)


def _unet_instances(model: Any, picture: Image.Image, size: int, device: Any) -> np.ndarray:
    """Binary mask, split into cells by watershed -- the benchmark's own recipe."""
    with torch.no_grad():
        mask, _ = predict_mask(model, picture, image_size=size, device=device)
    return label_objects(mask, method="watershed_split", min_area=MIN_AREA)


def _rates(
    shares: dict[int, float], damaged: set[int], intact: set[int], threshold: float
) -> tuple[float | None, float | None]:
    """Detection rate in each arm of one image, at one coverage threshold."""
    found = {cell for cell, share in shares.items() if share >= threshold}
    damaged_rate = len(found & damaged) / len(damaged) if damaged else None
    intact_rate = len(found & intact) / len(intact) if intact else None
    return intact_rate, damaged_rate


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired per-image differences, with the interval over acquisition groups."""
    buckets: dict[tuple[str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for threshold in COVERAGE_THRESHOLDS:
            key = f"{threshold:.1f}"
            if row[f"intact@{key}"] is None or row[f"damaged@{key}"] is None:
                continue
            buckets[(row["cell_line"], row["severity"], row["segmenter"], key)].append(row)

    summary: dict[str, Any] = {}
    for (cell_line, severity, segmenter, key), group in buckets.items():
        label = f"{cell_line}/{severity:.1f}/{segmenter}/cov{key}"
        intact = [row[f"intact@{key}"] for row in group]
        damaged = [row[f"damaged@{key}"] for row in group]
        groups = [row["acquisition_group"] for row in group]
        summary[label] = {
            "evaluated": True,
            "images": len(group),
            "cells_damaged": sum(row["cells_damaged"] for row in group),
            "cells_intact": sum(row["cells_intact"] for row in group),
            # Means over images, not over pooled cells: an image with 400 cells
            # and one with 40 are one observation each, because the image is the
            # thing that was sampled.
            "mean_intact_rate": float(np.mean(intact)),
            "mean_damaged_rate": float(np.mean(damaged)),
            "paired_difference": evaluation.paired_bootstrap(intact, damaged, groups=groups),
            # The two readout consequences get the same treatment as the gap:
            # computed per image from that image's own two rates, then given an
            # interval over groups. Deriving them from pooled rates instead
            # would inherit exactly the independence assumption this rewrite
            # exists to remove.
            "morphology_effect_recovered": evaluation.bootstrap_interval(
                [damage.recovered_effect(h, d, 0.5) for h, d in zip(intact, damaged, strict=True)],
                groups=groups,
            ),
            "count_drop_when_none_died": evaluation.bootstrap_interval(
                [
                    damage.apparent_count_drop(h, d, 0.5)
                    for h, d in zip(intact, damaged, strict=True)
                ],
                groups=groups,
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Damage-blindness sweep.")
    parser.add_argument("--axis", choices=AXES, default="combined")
    parser.add_argument("--cell-lines", nargs="*", default=list(MANIFESTS))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--resummarise",
        type=Path,
        default=None,
        help=(
            "Recompute the summary from an existing artifact's per-image rows "
            "instead of running the segmenters again. The rows are the "
            "measurement; everything else is arithmetic on them."
        ),
    )
    args = parser.parse_args()

    if args.resummarise is not None:
        payload = json.loads(args.resummarise.read_text(encoding="utf-8"))
        payload["summary"] = _summarise(payload["per_image"])
        args.resummarise.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _print_summary(payload["summary"])
        print(f"\nREWROTE {args.resummarise}")
        return

    import warnings

    warnings.filterwarnings("ignore")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    fingerprint = (
        preregistration.plan_fingerprint(PREREGISTRATION) if PREREGISTRATION.exists() else None
    )
    model = load_instance_checkpoint(CHECKPOINT, device)
    unet, unet_payload = load_checkpoint(UNET_CHECKPOINT, device)
    unet_size = int((unet_payload.get("preprocessing") or {}).get("image_size") or 512)
    cache: dict[Path, dict[int, Any]] = {}

    per_image: list[dict[str, Any]] = []
    for cell_line in args.cell_lines:
        manifest = MANIFESTS[cell_line]
        if not manifest.exists():
            print(f"  SKIP {cell_line}: no manifest at {manifest}", flush=True)
            continue
        records = load_manifest(manifest)[:IMAGES]
        for severity in SEVERITIES:
            for position, record in enumerate(records):
                truth = _labels_for(record, cache)
                # One generator per image position, seeded identically at every
                # severity, so the same cells land in the damaged arm all along
                # the axis being swept. Re-drawing the split would let cell
                # selection vary with the variable being measured.
                rng = np.random.default_rng(1000 + position)
                damaged, intact = damage.split_cells(truth, fraction=0.5, rng=rng)
                if not damaged or not intact:
                    continue
                with Image.open(record.image_path) as handle:
                    grey = np.asarray(handle.convert("L"), dtype=np.float32)
                degraded = damage.apply_damage(
                    grey, truth, damaged, _damage_for(args.axis, severity), rng=rng
                )
                picture = Image.fromarray(degraded.astype(np.uint8))

                predictions = {
                    "three_class": predict_instances(
                        model,
                        picture,
                        image_size=512,
                        device=device,
                        interior_threshold=INTERIOR_THRESHOLD,
                        min_area=MIN_AREA,
                    ),
                    "unet_watershed": _unet_instances(unet, picture, unet_size, device),
                    "cellpose": external.cellpose_instances(
                        np.asarray(picture), diameter=CELLPOSE_DIAMETER
                    ),
                }
                for name, predicted in predictions.items():
                    shares = damage.detection_coverage(predicted, truth)
                    row: dict[str, Any] = {
                        "cell_line": cell_line,
                        "severity": severity,
                        "segmenter": name,
                        "image_id": record.image_id,
                        "acquisition_group": acquisition_group(record),
                        "cells_damaged": len(damaged),
                        "cells_intact": len(intact),
                    }
                    for threshold in COVERAGE_THRESHOLDS:
                        intact_rate, damaged_rate = _rates(shares, damaged, intact, threshold)
                        key = f"{threshold:.1f}"
                        row[f"intact@{key}"] = intact_rate
                        row[f"damaged@{key}"] = damaged_rate
                    per_image.append(row)
            print(f"  {cell_line} severity {severity:.1f} done", flush=True)

    summary = _summarise(per_image)
    out = args.out or Path(f"runs/damage/blindness_{args.axis}.json")
    report = {
        "design": {
            "axis": args.axis,
            "images_per_cell_line": IMAGES,
            "cell_lines": args.cell_lines,
            "severities": list(SEVERITIES),
            "coverage_thresholds": list(COVERAGE_THRESHOLDS),
            "split": "each annotated cell assigned to damaged or intact by coin flip",
            "unit_of_analysis": (
                "acquisition group (one field of view at one timestamp). The paired "
                "difference is computed per image and the interval resamples whole "
                "groups, because cells within a frame are not independent trials."
            ),
            "detection_rule": (
                "a cell is found when one predicted object covers at least the "
                "coverage threshold of it"
            ),
            "checkpoint": provenance.relative_to_repo(CHECKPOINT),
            "unet_checkpoint": provenance.relative_to_repo(UNET_CHECKPOINT),
            "segmenters": list(SEGMENTERS),
            "preregistration": fingerprint,
        },
        "per_image": per_image,
        "summary": summary,
        "caveat": (
            "Synthetic damage is not real damage. What is measured is how a segmenter "
            "responds to a controlled loss of contrast, edge definition and interior "
            "texture. Whether dying cells in a real plate degrade along the same axes is "
            "what a real experiment would settle."
        ),
        "environment": provenance.environment(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _print_summary(summary)
    print(f"\nWROTE {out}")


def _print_summary(summary: dict[str, Any]) -> None:
    for label, row in sorted(summary.items()):
        interval = row["paired_difference"]
        if not interval.get("evaluated", True):
            continue
        print(
            f"  {label:36s} gap {interval['difference']:+.3f} "
            f"[{interval['ci_low']:+.3f}, {interval['ci_high']:+.3f}] "
            f"{interval['clusters']} groups"
        )


if __name__ == "__main__":
    main()
