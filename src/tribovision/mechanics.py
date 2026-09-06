"""Tissue mechanics inferred from cell outlines.

The vertex model of epithelial tissue predicts a rigidity transition governed by
a single dimensionless shape index,

    q = perimeter / sqrt(area)

with a critical value near 3.81. Below it a monolayer is jammed — solid,
stable, cells locked in place. Above it the tissue unjams and becomes fluid-like
and migratory, the mechanical state associated with invasion.

The useful property for this project is that q needs no dye, no stain and no
force measurement. It is computed from cell outlines, which is exactly what a
segmenter produces. That makes the mechanical state of a monolayer readable from
an ordinary label-free image — provided the outlines are accurate, which is a
much stronger requirement than it sounds.

**The discretisation trap.** Measuring the perimeter by counting pixel edges
inflates it by roughly 4/pi for a smooth boundary. q is linear in perimeter, so a
naive pixel measurement inflates q by about 27% — and because the threshold is a
fixed number rather than a relative one, that is enough to push essentially every
cell over it and report a fluid tissue no matter what the tissue is doing. This
module measures q both ways and applies an explicit correction, because a
mechanical conclusion drawn from an uncorrected pixel perimeter is an artifact.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

#: Vertex-model rigidity transition. Above this a monolayer is fluid-like.
JAMMING_THRESHOLD = 3.81
#: The physical minimum: no shape has a smaller perimeter for its area.
CIRCLE_SHAPE_INDEX = 2.0 * math.sqrt(math.pi)
#: A regular hexagon, the densest space-filling packing of equal cells.
HEXAGON_SHAPE_INDEX = math.sqrt(8.0 * math.sqrt(3.0))
#: Ratio by which a pixel-edge ("crack") perimeter exceeds a smooth boundary.
CRACK_PERIMETER_FACTOR = 4.0 / math.pi
#: Confluence above which crowding, rather than post-plating spreading, dominates.
#:
#: A monolayer does two different things over a time-lapse. Freshly seeded cells
#: are round and sparse; as they attach and spread the shape index *rises*. Only
#: once the field fills does crowding take over and drive it back down. Pooling
#: the two regimes cancels the second against the first and hides the jamming
#: signature entirely, so trajectories are reported split at this confluence.
CROWDING_CONFLUENCE = 0.5

_TIMESTAMP = re.compile(r"_(\d+)d(\d+)h(\d+)m_")


class MechanicsError(ValueError):
    """Raised when a shape index cannot be computed meaningfully."""


def shape_index(area: float, perimeter: float) -> float:
    """Return the dimensionless shape index q = P / sqrt(A)."""
    if area <= 0:
        raise MechanicsError(f"Area must be positive, got {area}.")
    if perimeter <= 0:
        raise MechanicsError(f"Perimeter must be positive, got {perimeter}.")
    return float(perimeter / math.sqrt(area))


def polygon_area_perimeter(ring: Sequence[float]) -> tuple[float, float]:
    """Shoelace area and closed-path perimeter of one flat COCO polygon ring."""
    if len(ring) < 6 or len(ring) % 2:
        raise MechanicsError("A polygon ring needs at least three x/y pairs.")
    xs = np.asarray(ring[0::2], dtype=float)
    ys = np.asarray(ring[1::2], dtype=float)
    area = 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))
    closed_x = np.append(xs, xs[0])
    closed_y = np.append(ys, ys[0])
    perimeter = float(np.sum(np.hypot(np.diff(closed_x), np.diff(closed_y))))
    return area, perimeter


def polygon_shape_index(segmentation: Any) -> float | None:
    """Shape index from a COCO polygon segmentation, or None if it is RLE."""
    if not isinstance(segmentation, list) or not segmentation:
        return None
    ring = segmentation[0] if isinstance(segmentation[0], list) else segmentation
    if not isinstance(ring, list) or len(ring) < 6:
        return None
    try:
        area, perimeter = polygon_area_perimeter(ring)
        return shape_index(area, perimeter)
    except MechanicsError:
        return None


def corrected_pixel_shape_index(
    area_pixels: float, crack_perimeter: float, *, factor: float = CRACK_PERIMETER_FACTOR
) -> float:
    """Shape index from a raster mask, with the discretisation bias divided out.

    The correction is the ratio between a pixel-edge perimeter and the smooth
    boundary it approximates. It is exact for a straight diagonal edge and
    asymptotically right for a large smooth object; it does not rescue a mask
    whose boundary is genuinely wrong.
    """
    return shape_index(area_pixels, crack_perimeter / factor)


def parse_hours(file_name: str) -> float | None:
    """Recover the time-lapse timestamp from a LIVECell file name, in hours."""
    match = _TIMESTAMP.search(file_name)
    if match is None:
        return None
    days, hours, minutes = (int(value) for value in match.groups())
    return days * 24.0 + hours + minutes / 60.0


def summarise(values: Sequence[float], *, seed: int = 0, resamples: int = 2000) -> dict[str, Any]:
    """Distribution summary with a bootstrap interval on the median."""
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if array.size == 0:
        return {"cells": 0}
    rng = np.random.default_rng(seed)
    medians = [
        float(np.median(rng.choice(array, size=array.size, replace=True)))
        for _ in range(resamples if array.size > 1 else 0)
    ]
    interval = (
        [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]
        if medians
        else None
    )
    return {
        "cells": int(array.size),
        "median_q": float(np.median(array)),
        "median_q_ci95": interval,
        "mean_q": float(array.mean()),
        "iqr": [float(np.percentile(array, 25)), float(np.percentile(array, 75))],
        "fraction_unjammed": float(np.mean(array > JAMMING_THRESHOLD)),
        "fraction_below_circle": float(np.mean(array < CIRCLE_SHAPE_INDEX)),
        "state": "unjammed (fluid-like)"
        if float(np.median(array)) > JAMMING_THRESHOLD
        else "jammed (solid-like)",
    }


def interpret(summary: dict[str, Any]) -> str:
    """One sentence a reader can check against the numbers."""
    if not summary.get("cells"):
        return "No cells measured."
    median = summary["median_q"]
    margin = median - JAMMING_THRESHOLD
    direction = "above" if margin > 0 else "below"
    return (
        f"Median shape index {median:.3f}, {abs(margin):.3f} {direction} the "
        f"q* = {JAMMING_THRESHOLD} rigidity transition, with "
        f"{summary['fraction_unjammed']:.0%} of cells above it. This describes the "
        "mechanical state of the monolayer as imaged; it is not a measurement of "
        "invasion, and nothing here shows that changing the state changes an outcome."
    )


def measure_annotation_file(
    path: Any,
    *,
    cell_type: str,
    min_area_pixels: float = 50.0,
    max_cells: int | None = None,
) -> dict[str, Any]:
    """Measure the shape index of every annotated cell in one COCO file.

    Only the polygon outlines are read, so no images are needed. Cells smaller
    than *min_area_pixels* are dropped: at a few tens of pixels the outline is
    dominated by discretisation and q is not meaningful.
    """
    import json
    from collections import defaultdict
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in payload.get("images", [])}
    per_image: dict[int, list[float]] = defaultdict(list)
    per_image_area: dict[int, list[float]] = defaultdict(list)
    values: list[float] = []
    skipped_rle = 0
    skipped_small = 0

    for annotation in payload.get("annotations", []):
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list) or not segmentation:
            skipped_rle += 1
            continue
        ring = segmentation[0] if isinstance(segmentation[0], list) else segmentation
        if not isinstance(ring, list) or len(ring) < 6:
            skipped_rle += 1
            continue
        try:
            area, perimeter = polygon_area_perimeter(ring)
        except MechanicsError:
            skipped_rle += 1
            continue
        if area < min_area_pixels:
            skipped_small += 1
            continue
        try:
            q = shape_index(area, perimeter)
        except MechanicsError:
            continue
        values.append(q)
        per_image[int(annotation["image_id"])].append(q)
        per_image_area[int(annotation["image_id"])].append(area)
        if max_cells is not None and len(values) >= max_cells:
            break

    # Confluence proxy: how many cells share each field of view, and the median
    # shape index in that field. Jamming theory predicts these move together.
    density_rows: list[dict[str, Any]] = []
    for image_id, cell_values in per_image.items():
        image = images.get(image_id)
        if image is None or len(cell_values) < 10:
            continue
        area = float(image.get("width", 0)) * float(image.get("height", 0))
        if area <= 0:
            continue
        areas = per_image_area.get(image_id, [])
        density_rows.append(
            {
                "image_id": image_id,
                "file_name": str(image.get("file_name", "")),
                "cells": len(cell_values),
                "cells_per_megapixel": len(cell_values) / (area / 1e6),
                # Confluence is the honest crowding measure: the fraction of the
                # field the cells actually cover. Counting cells conflates
                # crowding with cell size, and larger cells are also more
                # elongated, so a count-based correlation can be an artifact of
                # area rather than a signature of jamming.
                "confluence": float(sum(areas) / area) if areas else None,
                "median_area_pixels": float(np.median(areas)) if areas else None,
                "median_q": float(np.median(cell_values)),
                "hours": parse_hours(str(image.get("file_name", ""))),
            }
        )

    return {
        "cell_type": cell_type,
        "source": str(path),
        "summary": summarise(values),
        "skipped_rle_or_malformed": skipped_rle,
        "skipped_below_min_area": skipped_small,
        "min_area_pixels": min_area_pixels,
        "images": density_rows,
    }


def _guarded_spearman(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Spearman that reports an undefined correlation instead of warning.

    A trajectory whose shape index never moves is a real possibility - a fully
    jammed monolayer, or a synthetic control - and a correlation is genuinely
    undefined there rather than merely weak.
    """
    from scipy import stats

    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return {"spearman_rho": None, "p_value": None, "note": "input had no variance"}
    result = stats.spearmanr(x, y)
    return {"spearman_rho": float(result.statistic), "p_value": float(result.pvalue)}


def _partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> dict[str, Any]:
    """Spearman correlation of x and y after removing what *control* explains.

    Rank-transform everything, regress out the control linearly, and correlate the
    residuals. This is the standard partial correlation and it is what separates
    "crowding changes cell shape" from "big cells are both rarer and more
    elongated".
    """
    from scipy import stats

    ranks = [stats.rankdata(value) for value in (x, y, control)]
    residuals = []
    design = np.column_stack([np.ones_like(ranks[2]), ranks[2]])
    for series in ranks[:2]:
        coefficients, *_ = np.linalg.lstsq(design, series, rcond=None)
        residuals.append(series - design @ coefficients)
    result = stats.pearsonr(residuals[0], residuals[1])
    return {"rho": float(result.statistic), "p_value": float(result.pvalue)}


def density_relationship(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Does the tissue jam as it crowds?

    Reported three ways, because the obvious one is confounded. Cell *count* per
    field falls when cells are large, and large cells are also more elongated, so
    a count-based correlation can reproduce the jamming signature without any
    jamming. Confluence — the fraction of the field covered — does not have that
    problem, and the partial correlation removes cell size explicitly.
    """
    from scipy import stats

    usable = [row for row in rows if row.get("cells_per_megapixel") and row.get("median_q")]
    if len(usable) < 5:
        return {"evaluated": False, "reason": "Fewer than five fields of view."}
    count = np.array([row["cells_per_megapixel"] for row in usable], dtype=float)
    values = np.array([row["median_q"] for row in usable], dtype=float)
    if float(np.std(count)) < 1e-9 or float(np.std(values)) < 1e-9:
        return {"evaluated": False, "reason": "No variation in density or shape index."}

    by_count = stats.spearmanr(count, values)
    report: dict[str, Any] = {
        "evaluated": True,
        "fields": len(usable),
        "by_cell_count": {
            "spearman_rho": float(by_count.statistic),
            "p_value": float(by_count.pvalue),
            "caveat": "Confounded with cell size; use confluence instead.",
        },
        "hypothesis": (
            "Vertex-model jamming predicts a negative correlation: as a monolayer "
            "crowds, cells become more regular and the shape index falls toward q*."
        ),
    }
    with_confluence = [
        row for row in usable if row.get("confluence") and row.get("median_area_pixels")
    ]
    if len(with_confluence) >= 5:
        confluence = np.array([row["confluence"] for row in with_confluence], dtype=float)
        area = np.array([row["median_area_pixels"] for row in with_confluence], dtype=float)
        shape = np.array([row["median_q"] for row in with_confluence], dtype=float)
        if float(np.std(confluence)) > 1e-12:
            primary = stats.spearmanr(confluence, shape)
            report["by_confluence"] = {
                "spearman_rho": float(primary.statistic),
                "p_value": float(primary.pvalue),
            }
            report["controlling_for_cell_size"] = _partial_spearman(confluence, shape, area)
            report["supports_hypothesis"] = bool(primary.statistic < 0 and primary.pvalue < 0.05)
            report["survives_size_control"] = bool(
                report["controlling_for_cell_size"]["rho"] < 0
                and report["controlling_for_cell_size"]["p_value"] < 0.05
            )
            return report
    report["supports_hypothesis"] = bool(by_count.statistic < 0 and by_count.pvalue < 0.05)
    return report


def time_course(
    paths: list[Any],
    *,
    cell_type: str,
    min_area_pixels: float = 50.0,
    min_timepoints: int = 6,
    crowding_confluence: float = CROWDING_CONFLUENCE,
) -> dict[str, Any]:
    """Follow the mechanical state of each well as its monolayer fills in.

    LIVECell is a time-lapse: the same wells are imaged repeatedly, and the
    timestamp is in the file name. That makes the strongest available test of the
    jamming hypothesis possible — *within* a well, does the shape index fall as
    the monolayer crowds? A within-well trajectory removes every between-well
    confound at once, which the cross-sectional correlation cannot.

    Each well contributes one trajectory. Wells are the unit; fields imaged at the
    same timestamp are averaged first, so imaging more fields cannot inflate n.
    """
    import json
    from collections import defaultdict
    from pathlib import Path

    from tribovision.manifest import parse_file_name

    # (well, hours) -> shape indices and areas pooled across fields
    grouped_q: dict[tuple[str, float], list[float]] = defaultdict(list)
    grouped_area: dict[tuple[str, float], list[float]] = defaultdict(list)
    frame_area: dict[tuple[str, float], float] = {}
    fields_seen: dict[tuple[str, float], set[int]] = defaultdict(set)

    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        images = {}
        for image in payload.get("images", []):
            file_name = str(image.get("file_name", ""))
            hours = parse_hours(file_name)
            parts = parse_file_name(file_name)
            if hours is None or parts is None:
                continue
            images[int(image["id"])] = (
                parts["well"],
                hours,
                float(image.get("width", 0)) * float(image.get("height", 0)),
            )
        for annotation in payload.get("annotations", []):
            meta = images.get(int(annotation.get("image_id", -1)))
            if meta is None:
                continue
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, list) or not segmentation:
                continue
            ring = segmentation[0] if isinstance(segmentation[0], list) else segmentation
            if not isinstance(ring, list) or len(ring) < 6:
                continue
            try:
                area, perimeter = polygon_area_perimeter(ring)
                if area < min_area_pixels:
                    continue
                q = shape_index(area, perimeter)
            except MechanicsError:
                continue
            well, hours, image_area = meta
            key = (well, hours)
            grouped_q[key].append(q)
            grouped_area[key].append(area)
            frame_area[key] = image_area
            fields_seen[key].add(int(annotation["image_id"]))

    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (well, hours), values in sorted(grouped_q.items()):
        if len(values) < 10:
            continue
        fields = max(1, len(fields_seen[(well, hours)]))
        total_frame_area = frame_area[(well, hours)] * fields
        trajectories[well].append(
            {
                "hours": hours,
                "cells": len(values),
                "fields": fields,
                "median_q": float(np.median(values)),
                "confluence": float(sum(grouped_area[(well, hours)]) / total_frame_area),
            }
        )

    wells: dict[str, Any] = {}
    for well, points in trajectories.items():
        if len(points) < min_timepoints:
            continue
        points.sort(key=lambda row: row["hours"])
        timeline = np.array([row["hours"] for row in points], dtype=float)
        shape = np.array([row["median_q"] for row in points], dtype=float)
        confluence = np.array([row["confluence"] for row in points], dtype=float)
        against_time = _guarded_spearman(timeline, shape)
        against_confluence = _guarded_spearman(confluence, shape)
        crowded = confluence >= crowding_confluence
        crowded_result: dict[str, Any] = {
            "reached": bool(crowded.sum() >= 5),
            "timepoints": int(crowded.sum()),
            "threshold": crowding_confluence,
        }
        if crowded_result["reached"]:
            restricted = _guarded_spearman(confluence[crowded], shape[crowded])
            crowded_result.update(restricted)
            crowded_result["jams"] = bool(
                restricted["spearman_rho"] is not None
                and restricted["spearman_rho"] < 0
                and restricted["p_value"] < 0.05
            )
        else:
            crowded_result["note"] = (
                "This well never crowds enough for jamming to be the dominant process, "
                "so its trajectory measures post-plating spreading instead. A positive "
                "correlation here is not evidence against jamming."
            )
        wells[well] = {
            "timepoints": len(points),
            "hours_range": [float(timeline.min()), float(timeline.max())],
            "confluence_range": [float(confluence.min()), float(confluence.max())],
            "q_start": float(shape[0]),
            "q_end": float(shape[-1]),
            "q_change": float(shape[-1] - shape[0]),
            "crossed_transition": bool(
                (shape[0] > JAMMING_THRESHOLD) != (shape[-1] > JAMMING_THRESHOLD)
            ),
            "vs_time": against_time,
            "vs_confluence": against_confluence,
            "crowded_regime": crowded_result,
            "jams_as_it_crowds": bool(crowded_result.get("jams", False)),
            "trajectory": points,
        }

    reached = [w for w, v in wells.items() if v["crowded_regime"]["reached"]]
    jamming = [w for w in reached if wells[w]["jams_as_it_crowds"]]
    return {
        "cell_type": cell_type,
        "wells": wells,
        "wells_measured": len(wells),
        "wells_reaching_crowded_regime": len(reached),
        "wells_that_jam_as_they_crowd": len(jamming),
        "consistent": bool(reached) and len(jamming) in (0, len(reached)),
        "verdict": (
            "no well had enough timepoints"
            if not wells
            else f"{len(jamming)} of {len(reached)} wells that reach confluence "
            f"{CROWDING_CONFLUENCE} jam as they crowd "
            f"({len(wells) - len(reached)} well(s) never crowd that far)"
        ),
        "note": (
            "Each well is one trajectory and the unit of analysis; fields imaged at "
            "the same timestamp are averaged first. A well-level result from a single "
            "well is descriptive, not replicated - check wells_measured before "
            "quoting it."
        ),
    }
