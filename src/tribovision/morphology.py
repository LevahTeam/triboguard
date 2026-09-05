"""Per-object morphology measured on the native image grid.

Two honesty constraints shape this module.

*Regions are not automatically cells.* Plain connected-component labelling merges
cells that touch, which is common at the confluences LIVECell contains. Features
extracted that way describe predicted *regions*. A marker-controlled splitter is
provided and every row records which method produced it, so a reader can tell
whether a number is a region or a separated object rather than having to assume.

*Units are pixels until someone supplies a calibration.* Micrometre columns
appear only when ``micrometers_per_pixel`` is given by the operator; nothing here
guesses a microscope's scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

INSTANCE_METHODS = ("connected_components", "watershed_split")

#: A perfectly round *digitised* object scores about this, not 1.0, because the
#: crack perimeter of a pixel grid exceeds the circumference it approximates.
#: Reports quote it so that a circularity of 0.59 is not read as "quite ragged".
DIGITISED_DISC_CIRCULARITY = 0.59


@dataclass(frozen=True)
class Calibration:
    """Physical scale supplied by the operator, or absent."""

    micrometers_per_pixel: float | None = None

    def area(self, pixels: float) -> float | None:
        if self.micrometers_per_pixel is None:
            return None
        return float(pixels) * self.micrometers_per_pixel**2

    def length(self, pixels: float) -> float | None:
        if self.micrometers_per_pixel is None:
            return None
        return float(pixels) * self.micrometers_per_pixel


def connected_components(mask: np.ndarray) -> np.ndarray:
    """Label 8-connected foreground components."""
    structure = np.ones((3, 3), dtype=bool)
    labels, _ = ndimage.label(np.asarray(mask).astype(bool), structure=structure)
    return labels.astype(np.int32)


def _seed_markers(distance: np.ndarray, min_distance: int) -> np.ndarray:
    """One marker per local maximum of the distance transform."""
    footprint = np.ones((2 * min_distance + 1, 2 * min_distance + 1), dtype=bool)
    peaks = (distance >= ndimage.maximum_filter(distance, footprint=footprint)) & (
        distance > min_distance
    )
    markers, _ = ndimage.label(peaks, structure=np.ones((3, 3), dtype=bool))
    return markers.astype(np.int32)


def _flood(labels: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Grow labels into *allowed* until nothing more can be claimed."""
    footprint = np.ones((3, 3), dtype=bool)
    while True:
        grown = ndimage.grey_dilation(labels, footprint=footprint)
        updated = np.where((labels == 0) & allowed & (grown > 0), grown, labels)
        if np.array_equal(updated, labels):
            return labels
        labels = updated


def watershed_split(mask: np.ndarray, *, min_distance: int = 3, levels: int = 24) -> np.ndarray:
    """Separate touching objects by flooding from distance-transform maxima.

    This is a marker-controlled watershed written directly on top of
    ``scipy.ndimage`` so that every step stays inspectable. Where two catchment
    basins meet, the higher label wins; that boundary rule is arbitrary but
    consistent, and it affects only a one-pixel-wide seam.
    """
    foreground = np.asarray(mask).astype(bool)
    if not foreground.any():
        return np.zeros(foreground.shape, dtype=np.int32)
    distance = ndimage.distance_transform_edt(foreground)
    markers = _seed_markers(distance, min_distance)
    if markers.max() == 0:
        return connected_components(foreground)
    labels = markers.copy()
    maximum = float(distance.max())
    for level in np.linspace(maximum, 0.0, max(2, levels)):
        labels = _flood(labels, foreground & (distance >= level))
    labels = _flood(labels, foreground)
    # Any component with no marker at all keeps its own identity.
    orphans = foreground & (labels == 0)
    if orphans.any():
        extra = connected_components(orphans)
        labels = np.where(orphans, extra + labels.max(), labels)
    return relabel(labels)


def relabel(labels: np.ndarray) -> np.ndarray:
    """Renumber labels to a dense 1..n range."""
    values = np.unique(labels)
    values = values[values > 0]
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lookup[values] = np.arange(1, len(values) + 1, dtype=np.int32)
    return lookup[labels]


def label_objects(
    mask: np.ndarray, *, method: str = "connected_components", min_area: int = 0
) -> np.ndarray:
    if method not in INSTANCE_METHODS:
        raise ValueError(f"method must be one of {INSTANCE_METHODS}, got {method!r}.")
    labels = watershed_split(mask) if method == "watershed_split" else connected_components(mask)
    if min_area > 0 and labels.max() > 0:
        areas = np.bincount(labels.ravel())
        drop = np.where(areas < min_area)[0]
        drop = drop[drop > 0]
        if drop.size:
            labels = np.where(np.isin(labels, drop), 0, labels)
            labels = relabel(labels)
    return labels


def crack_perimeter(component: np.ndarray) -> int:
    """Count 4-connected boundary edges of a binary component.

    This "crack" perimeter systematically exceeds the perimeter of the underlying
    smooth shape (a digitised disc measures roughly 4/pi times its true
    circumference). It is reported as-is, and circularity derived from it is
    therefore comparable *between conditions measured the same way* rather than
    interpretable as an absolute shape constant.
    """
    binary = np.asarray(component).astype(bool)
    padded = np.pad(binary, 1)
    horizontal = int(np.count_nonzero(padded[1:-1, 1:-1] & ~padded[1:-1, 2:]))
    horizontal += int(np.count_nonzero(padded[1:-1, 1:-1] & ~padded[1:-1, :-2]))
    vertical = int(np.count_nonzero(padded[1:-1, 1:-1] & ~padded[2:, 1:-1]))
    vertical += int(np.count_nonzero(padded[1:-1, 1:-1] & ~padded[:-2, 1:-1]))
    return horizontal + vertical


def _axes_from_moments(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float, float]:
    """Return (major axis, minor axis, orientation) from second central moments."""
    if len(xs) < 2:
        return 1.0, 1.0, 0.0
    coordinates = np.vstack((xs - xs.mean(), ys - ys.mean()))
    covariance = coordinates @ coordinates.T / len(xs)
    # +1/12 corrects the discretisation variance of unit pixels.
    covariance += np.eye(2) / 12.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    minor, major = np.sqrt(eigenvalues) * 4.0
    orientation = float(math.degrees(math.atan2(*eigenvectors[:, 1][::-1])))
    return float(major), float(minor), orientation


def measure(
    labels: np.ndarray,
    intensity: np.ndarray | None = None,
    *,
    calibration: Calibration | None = None,
    method: str = "connected_components",
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Measure every labelled object and return one row per object."""
    calibration = calibration or Calibration()
    labels = np.asarray(labels)
    count = int(labels.max())
    if count == 0:
        return []
    objects = ndimage.find_objects(labels)
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(objects, start=1):
        if window is None:
            continue
        component = labels[window] == index
        area = int(np.count_nonzero(component))
        if area == 0:
            continue
        ys, xs = np.nonzero(component)
        ys = ys + window[0].start
        xs = xs + window[1].start
        perimeter = crack_perimeter(component)
        major, minor, orientation = _axes_from_moments(ys.astype(np.float64), xs.astype(np.float64))
        circularity = (4 * math.pi * area / perimeter**2) if perimeter else 0.0
        row: dict[str, Any] = {
            **(extra or {}),
            "object_id": index,
            "instance_method": method,
            "area_pixels": area,
            "perimeter_pixels": perimeter,
            "centroid_x": float(xs.mean()),
            "centroid_y": float(ys.mean()),
            "equivalent_diameter_pixels": float(2 * math.sqrt(area / math.pi)),
            "major_axis_pixels": major,
            "minor_axis_pixels": minor,
            "aspect_ratio": float(major / minor) if minor > 0 else float("nan"),
            "orientation_degrees": orientation,
            # 1.0 for a perfectly round digitised object of this size; lower means
            # more elongated or more ragged.
            "circularity": float(min(circularity, 1.0)),
            "solidity": _solidity(ys, xs, area),
        }
        if intensity is not None:
            values = np.asarray(intensity, dtype=np.float64)[ys, xs]
            row["mean_intensity"] = float(values.mean())
            row["std_intensity"] = float(values.std())
            row["min_intensity"] = float(values.min())
            row["max_intensity"] = float(values.max())
        if calibration.micrometers_per_pixel is not None:
            row["area_um2"] = calibration.area(area)
            row["perimeter_um"] = calibration.length(perimeter)
            row["equivalent_diameter_um"] = calibration.length(row["equivalent_diameter_pixels"])
            row["micrometers_per_pixel"] = calibration.micrometers_per_pixel
        rows.append(row)
    return rows


def _solidity(ys: np.ndarray, xs: np.ndarray, area: int) -> float:
    """Object area divided by the area of its convex hull.

    A cell that spreads and ruffles has a lower solidity than one that has
    rounded up, so this is one of the more direct "rounding" indicators. The hull
    is taken over pixel corners, which keeps the ratio at 1.0 for convex shapes
    instead of leaving a systematic half-pixel deficit.
    """
    if area < 3:
        return 1.0
    corners = np.concatenate(
        [
            np.column_stack((xs + dx, ys + dy))
            for dx, dy in ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5))
        ]
    )
    try:
        hull_area = float(ConvexHull(corners).volume)
    except (QhullError, ValueError):
        # Collinear or otherwise degenerate objects have no two-dimensional hull.
        return 1.0
    if hull_area <= 0:
        return 1.0
    return float(min(1.0, area / hull_area))


def summarise_image(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-object rows into the image-level numbers used for dose response."""
    if not rows:
        return {"objects": 0}

    def stat(key: str) -> dict[str, float]:
        values = np.array([float(row[key]) for row in rows if row.get(key) is not None])
        if values.size == 0:
            return {}
        return {
            f"{key}_mean": float(values.mean()),
            f"{key}_median": float(np.median(values)),
            f"{key}_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }

    summary: dict[str, Any] = {"objects": len(rows)}
    for key in (
        "area_pixels",
        "circularity",
        "aspect_ratio",
        "solidity",
        "equivalent_diameter_pixels",
        "mean_intensity",
    ):
        if key in rows[0]:
            summary.update(stat(key))
    summary["total_area_pixels"] = float(sum(float(row["area_pixels"]) for row in rows))
    return summary
