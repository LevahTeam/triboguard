"""What an image-based cell count can conclude when the segmenter misses damaged cells.

The birth-death result says a bulk readout cannot separate killing from stalling,
and the design layer recommends imaging because it counts cells instead of
averaging them. That recommendation assumes the segmenter sees every cell still
in the well. The damage-blindness experiment measures that it does not: a cell
that is damaged but alive, and too faint to segment, is counted exactly like a
cell that has gone. So an image-based count carries an identifiability problem
of its own, one layer underneath the first.

Write a treated well relative to its untreated control as three kinds of cell:

* a fraction ``g`` that is genuinely **absent** -- died, or never born because
  division slowed. A perfect counter would report exactly this;
* a fraction ``k`` that is **damaged but present**;
* the rest, intact.

With intact cells detected at rate ``h`` and damaged cells at rate ``d``, the
counted ratio of treated to control is

    R = 1 - g - k * b,        b = 1 - d / h

where ``b`` is the segmenter's **blindness** to damaged cells: 0 when it finds
them as reliably as healthy ones, 1 when it finds none. One observation, two
unknowns. The absent fraction is therefore an interval, not a number:

    g  in  [ max(0, (1 - R - b) / (1 - b)),  1 - R ]

The top is reached if every uncounted cell is really gone; the bottom if as many
as possible are merely damaged and unseen (bounded because damaged cells cannot
outnumber the survivors). At ``b = 0`` the interval closes to ``1 - R`` and the
count means what everyone assumes. At ``b = 1`` it opens to ``[0, 1 - R]`` and
the count cannot say whether anything is absent at all.

``b`` is precisely what the damage sweep measures, so this turns a measured
blindness into a stated range -- and abstains when the range is too wide to carry
a claim. Note what ``g`` is *not*: it is not a killing fraction. Telling death
from slowed division inside ``g`` is the birth-death problem, and needs the
spread between wells. This module is the layer below that one: without it, a
real segmenter cannot even recover ``g``.
"""

from __future__ import annotations

import math
from typing import Any

from triboguard.kinetics import EFFECT_THRESHOLD

#: Widest interval on the absent fraction that still counts as an answer. A
#: count pinned to within ten percentage points is a measurement; wider than
#: that, it is a range pretending to be one.
RESOLUTION = 0.10

RESOLVED = "resolved"
PRESENT_SIZE_UNRESOLVED = "absence established, size unresolved"
NO_MEANINGFUL_ABSENCE = "no meaningful absence"
ABSTAIN = "abstain"


class BlindnessError(ValueError):
    """Raised when a count or a detection rate cannot describe a real measurement."""


def _check_unit(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise BlindnessError(f"{name} must lie in [0, 1], got {value}.")
    return float(value)


def blindness(intact_rate: float, damaged_rate: float) -> float:
    """How much of the damaged population the segmenter fails to see, relative to healthy.

    Clipped to ``[0, 1]``. A damaged rate above the intact one -- which sampling
    noise can produce near zero damage -- means the segmenter is not blind, not
    that it is negatively blind, so it reads as 0.
    """
    intact = _check_unit("intact_rate", intact_rate)
    damaged = _check_unit("damaged_rate", damaged_rate)
    if intact <= 0.0:
        raise BlindnessError(
            "a segmenter that finds no healthy cells has no blindness to measure; "
            "its counts carry no information at all."
        )
    return min(1.0, max(0.0, 1.0 - damaged / intact))


def _check_ratio(name: str, value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise BlindnessError(f"{name} must be finite and non-negative, got {value}.")
    return float(value)


def absent_bounds(ratio: float, blind: float) -> tuple[float, float]:
    """The absent fraction consistent with one counted ratio at one blindness.

    A ratio above 1 -- the treated well counted *more* cells than its control --
    leaves nothing absent, and both bounds are 0.
    """
    r = _check_ratio("ratio", ratio)
    b = _check_unit("blindness", blind)
    top = max(0.0, 1.0 - r)
    if b >= 1.0:
        return 0.0, top
    bottom = max(0.0, (1.0 - r - b) / (1.0 - b))
    return min(bottom, top), top


def absent_interval(
    ratio_low: float, ratio_high: float, blind_low: float, blind_high: float
) -> tuple[float, float]:
    """Worst case over a range of counted ratios and a range of blindness.

    Both bounds of :func:`absent_bounds` fall as the ratio rises, and the lower
    bound also falls as blindness rises, so the extremes of the box give the
    extremes of the interval: the bottom at the highest ratio and the most
    blindness, the top at the lowest ratio.
    """
    r_lo, r_hi = _check_ratio("ratio_low", ratio_low), _check_ratio("ratio_high", ratio_high)
    b_lo, b_hi = _check_unit("blind_low", blind_low), _check_unit("blind_high", blind_high)
    if r_lo > r_hi:
        raise BlindnessError(f"ratio_low {r_lo} exceeds ratio_high {r_hi}.")
    if b_lo > b_hi:
        raise BlindnessError(f"blind_low {b_lo} exceeds blind_high {b_hi}.")
    bottom, _ = absent_bounds(r_hi, b_hi)
    _, top = absent_bounds(r_lo, b_lo)
    return bottom, top


def assess(
    ratio_low: float,
    ratio_high: float,
    blind_low: float,
    blind_high: float,
    *,
    resolution: float = RESOLUTION,
    effect: float = EFFECT_THRESHOLD,
) -> dict[str, Any]:
    """What a counted ratio supports, given what is known about the segmenter.

    Four answers, and the fourth is the one this project exists to give:

    * **resolved** -- the interval is narrow and clear of zero;
    * **absence established, size unresolved** -- something is gone, but how much
      the count cannot say;
    * **no meaningful absence** -- even the top of the interval is below the
      effect threshold;
    * **abstain** -- the interval straddles the threshold: the same count is
      consistent with nothing being absent and with a real effect.
    """
    if not 0.0 < resolution <= 1.0 or not math.isfinite(resolution):
        raise BlindnessError(f"resolution must lie in (0, 1], got {resolution}.")
    if not 0.0 < effect < 1.0 or not math.isfinite(effect):
        raise BlindnessError(f"effect must lie in (0, 1), got {effect}.")
    bottom, top = absent_interval(ratio_low, ratio_high, blind_low, blind_high)
    width = top - bottom
    if top < effect:
        verdict = NO_MEANINGFUL_ABSENCE
    elif bottom >= effect and width <= resolution:
        verdict = RESOLVED
    elif bottom >= effect:
        verdict = PRESENT_SIZE_UNRESOLVED
    else:
        verdict = ABSTAIN
    return {
        "absent_low": bottom,
        "absent_high": top,
        "width": width,
        "verdict": verdict,
        "inputs": {
            "ratio": [float(ratio_low), float(ratio_high)],
            "blindness": [float(blind_low), float(blind_high)],
            "resolution": resolution,
            "effect": effect,
        },
        "reads_as": _reading(verdict, bottom, top),
    }


def _reading(verdict: str, bottom: float, top: float) -> str:
    span = f"between {bottom:.0%} and {top:.0%}"
    if verdict == RESOLVED:
        return f"The count supports {span} of cells absent."
    if verdict == PRESENT_SIZE_UNRESOLVED:
        return f"Cells are genuinely absent, {span}; the count cannot narrow it further."
    if verdict == NO_MEANINGFUL_ABSENCE:
        return f"At most {top:.0%} of cells are absent, below the effect threshold."
    return (
        f"Anywhere {span} of cells could be absent: the same count is consistent with "
        "no effect and with a real one, because the segmenter cannot see the cells "
        "that would tell them apart."
    )
