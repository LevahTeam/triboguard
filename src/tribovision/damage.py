"""Are cell segmenters blind to the cells that matter most?

Every segmenter in common use is trained on images of healthy cultures, because
that is what annotated datasets contain. A dying cell does not look like a
healthy one: it loses contrast against the medium, its membrane stops being a
crisp edge, and its interior fills with vacuoles. If a segmenter finds those
cells less reliably than healthy ones, then any viability measure built on
segmentation carries a bias with a direction -- it under-reports damage, and so
makes a treatment look weaker than it is.

That is a testable claim and it needs no laboratory. LIVECell ships a hand-drawn
outline for every cell, so damage can be applied to known cells in known amounts
and the detection rate read off exactly.

**The design is paired within an image.** Each annotated cell is assigned by coin
flip to damaged or untouched, and both live in the same frame under the same
illumination, focus and confluence. Comparing detection between the two arms
therefore controls for every image-level confound at once; comparing damaged
images against separate healthy images would not.

**Synthetic damage is not real damage, and this cannot pretend otherwise.** What
is measured is how a segmenter responds to a controlled reduction in contrast and
edge definition. Whether a dying cell in a real plate degrades along the same
axes is exactly the question a real experiment would settle, and the honest
reading of any result here is a hypothesis about real images rather than a
measurement of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: A ground-truth cell counts as found when a single predicted object covers at
#: least this share of it. Deliberately generous: the question is whether the
#: segmenter saw the cell at all, not whether it outlined it well.
DETECTION_COVERAGE = 0.5

#: Radius, in pixels, of the neighbourhood whose median stands in for "the
#: background this cell would fade into". Large enough to reach past a cell,
#: small enough to track uneven illumination across a frame.
BACKGROUND_RADIUS = 12


class DamageError(ValueError):
    """Raised when a damage model or a measurement cannot be applied."""


@dataclass(frozen=True)
class Damage:
    """How far a cell has been pushed toward invisibility.

    ``fade`` mixes the cell's pixels toward the local background: 0 leaves it
    untouched, 1 erases it completely. ``blur`` softens the boundary, standing in
    for a membrane that has stopped being a crisp edge. ``vacuoles`` adds bright
    interior blobs, the feature the source study's own figure points arrows at.
    """

    fade: float = 0.0
    blur: float = 0.0
    vacuoles: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (("fade", self.fade), ("blur", self.blur), ("vacuoles", self.vacuoles)):
            if not 0.0 <= value <= 1.0 or not np.isfinite(value):
                raise DamageError(f"{name} must lie in [0, 1], got {value}.")

    @property
    def severity(self) -> float:
        """A single number for plotting, the mean of the three axes."""
        return float((self.fade + self.blur + self.vacuoles) / 3.0)

    @staticmethod
    def at(severity: float) -> Damage:
        """The default trajectory: all three axes move together.

        A real dying cell fades, softens and vacuolates at once, so sweeping one
        axis alone would answer a narrower question than the one asked.
        """
        return Damage(fade=severity, blur=severity, vacuoles=severity)


def _local_background(image: np.ndarray, radius: int = BACKGROUND_RADIUS) -> np.ndarray:
    """A smooth estimate of what the medium looks like behind each pixel."""
    from scipy import ndimage

    # A median filter ignores the cells sitting on top of the background far
    # better than a mean would, which matters at the confluence LIVECell images
    # are taken at.
    return ndimage.median_filter(image.astype(np.float32), size=2 * radius + 1, mode="reflect")


def apply_damage(
    image: np.ndarray,
    labels: np.ndarray,
    damaged: set[int],
    damage: Damage,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return ``image`` with the cells in ``damaged`` degraded by ``damage``.

    Untouched cells keep every original pixel, so the two arms differ only in
    the transformation and not in how they were written back.
    """
    from scipy import ndimage

    generator = np.random.default_rng() if rng is None else rng
    picture = np.asarray(image, dtype=np.float32)
    if picture.shape != labels.shape:
        raise DamageError(f"image {picture.shape} and labels {labels.shape} disagree.")
    if not damaged:
        return picture.copy()

    out = picture.copy()
    background = _local_background(picture)
    selected = np.isin(labels, list(damaged))

    if damage.fade > 0:
        # Mix toward the medium. At fade 1 the cell is gone.
        out[selected] = (1.0 - damage.fade) * picture[selected] + damage.fade * background[selected]

    if damage.vacuoles > 0:
        # Bright interior blobs, placed only inside cells and only where the
        # cell is well away from its own edge, which is where real vacuoles sit.
        interior = ndimage.binary_erosion(selected, np.ones((5, 5), bool))
        if interior.any():
            spots = generator.random(picture.shape) < (0.02 * damage.vacuoles)
            spots &= interior
            if spots.any():
                blobs = ndimage.gaussian_filter(spots.astype(np.float32), sigma=1.6)
                blobs /= max(float(blobs.max()), 1e-9)
                lift = damage.vacuoles * (background - picture)
                out += blobs * lift * selected

    if damage.blur > 0:
        # Soften across the boundary rather than inside the cell, so the edge
        # stops being crisp without the whole cell turning to mush.
        band = ndimage.binary_dilation(selected, np.ones((5, 5), bool))
        smoothed = ndimage.gaussian_filter(out, sigma=0.5 + 2.5 * damage.blur)
        out[band] = smoothed[band]

    return np.clip(out, 0.0, 255.0)


def detected_cells(
    predicted: np.ndarray, truth: np.ndarray, *, coverage: float = DETECTION_COVERAGE
) -> set[int]:
    """Which ground-truth cells a prediction actually found.

    A cell counts as found when one predicted object covers at least ``coverage``
    of it. Using a single object rather than the union of all foreground is what
    stops a prediction that merges a whole cluster into one blob from counting as
    having found every cell in it.
    """
    if predicted.shape != truth.shape:
        raise DamageError(f"predicted {predicted.shape} and truth {truth.shape} disagree.")
    if not 0 < coverage <= 1:
        raise DamageError(f"coverage must lie in (0, 1], got {coverage}.")

    found: set[int] = set()
    for cell in np.unique(truth):
        if cell == 0:
            continue
        mask = truth == cell
        area = int(mask.sum())
        if area == 0:
            continue
        overlaps = predicted[mask]
        overlaps = overlaps[overlaps > 0]
        if overlaps.size == 0:
            continue
        _, counts = np.unique(overlaps, return_counts=True)
        if int(counts.max()) / area >= coverage:
            found.add(int(cell))
    return found


def split_cells(
    labels: np.ndarray, *, fraction: float = 0.5, rng: np.random.Generator | None = None
) -> tuple[set[int], set[int]]:
    """Assign each cell to the damaged or the untouched arm, by coin flip.

    Both arms then sit in the same frame, so illumination, focus and local
    crowding are shared and cannot explain a difference between them.
    """
    if not 0 <= fraction <= 1:
        raise DamageError(f"fraction must lie in [0, 1], got {fraction}.")
    generator = np.random.default_rng() if rng is None else rng
    cells = [int(value) for value in np.unique(labels) if value != 0]
    if not cells:
        return set(), set()
    picked = generator.random(len(cells)) < fraction
    damaged = {cell for cell, take in zip(cells, picked, strict=True) if take}
    return damaged, set(cells) - damaged


def summarise(found: set[int], damaged: set[int], intact: set[int]) -> dict[str, Any]:
    """Detection rates for the two arms, and the gap between them."""
    damaged_rate = len(found & damaged) / len(damaged) if damaged else None
    intact_rate = len(found & intact) / len(intact) if intact else None
    return {
        "cells_damaged": len(damaged),
        "cells_intact": len(intact),
        "found_damaged": len(found & damaged),
        "found_intact": len(found & intact),
        "detection_damaged": damaged_rate,
        "detection_intact": intact_rate,
        "detection_gap": (
            None if damaged_rate is None or intact_rate is None else intact_rate - damaged_rate
        ),
    }


def recovered_effect(intact_rate: float, damaged_rate: float, affected_fraction: float) -> float:
    """What share of a real morphology change a segmentation readout still sees.

    This models the readout this project actually uses. The treatment pipeline
    measures a *shape* feature averaged over segmented cells, and the damaged
    cells are the ones carrying the signal. Miss them and the average is taken
    over survivors, so the measured shift is diluted toward "nothing happened".

    With ``k`` affected cells whose feature has moved, healthy cells detected at
    rate ``h`` and damaged ones at ``d``, the measured mean shift is a
    detection-weighted average and the share of the true shift recovered is

        d / ((1 - k) * h + k * d)

    which is 1 when the two detection rates are equal, and 0 when damaged cells
    are never found.

    An earlier version of this function modelled a *count* readout instead, and
    got the direction of the bias backwards. Counting is not diluted by
    blindness -- it is inflated by it, because a dying cell that cannot be seen
    is indistinguishable from one that has gone. The two readouts are biased in
    opposite directions by the same blindness, which is worth stating plainly
    rather than collapsing into one headline.
    """
    for name, value in (
        ("intact_rate", intact_rate),
        ("damaged_rate", damaged_rate),
        ("affected_fraction", affected_fraction),
    ):
        if not 0 <= value <= 1 or not np.isfinite(value):
            raise DamageError(f"{name} must lie in [0, 1], got {value}.")
    if affected_fraction == 0:
        # No cells changed, so there is no effect to dilute. Nothing was missed.
        return 1.0
    denominator = (1.0 - affected_fraction) * intact_rate + affected_fraction * damaged_rate
    if denominator <= 0:
        # Nothing was detected at all, so no effect could be measured either way.
        return 0.0
    return float(damaged_rate / denominator)


def apparent_count_drop(intact_rate: float, damaged_rate: float, affected_fraction: float) -> float:
    """The drop a *count* readout reports when a fraction of cells are damaged.

    Kept beside :func:`recovered_effect` because the same blindness pushes the
    two readouts in opposite directions, and reporting only one would be a
    convenient half of the story.

    Damaged cells are assumed still present in the well. If they are as visible
    as healthy ones the count does not move at all, which is correct and often
    surprising: counting cells cannot detect damage while the damaged cells are
    still there. As they become harder to see, the count falls, and the fall
    looks like killing.
    """
    for name, value in (
        ("intact_rate", intact_rate),
        ("damaged_rate", damaged_rate),
        ("affected_fraction", affected_fraction),
    ):
        if not 0 <= value <= 1 or not np.isfinite(value):
            raise DamageError(f"{name} must lie in [0, 1], got {value}.")
    if intact_rate <= 0:
        return 0.0
    before = intact_rate
    after = (1.0 - affected_fraction) * intact_rate + affected_fraction * damaged_rate
    return float(1.0 - after / before)
