"""What to measure next, and what it will cost to learn the mechanism.

The two earlier stages leave one conclusion to act on: the mechanism lives in
the variance across wells, and the variance has ``wells - 1`` degrees of
freedom. Everything here follows from that.

**Extra timepoints do not buy the mechanism.** Measured on simulated wells,
going from three timepoints to nine at three wells moved the turnover interval
from 1.999 to 1.979. Going from three wells to twenty-four moved it to 0.082. A
planner that only chose *when* to measure -- the obvious reading of "pick the
next timepoint" -- would be optimising the axis that does not matter, so this
one ranks wells and timepoints against each other and usually answers "wells".

**Whether the assay destroys the well changes the price of a timepoint.** MTS
reads a colour change produced by lysing cells; the well is gone afterwards, so
a five-timepoint course needs five separate sets of wells. Microscopy is
non-destructive and the same wells can be imaged all week. The identical
scientific request therefore costs five times as much under one assay as the
other, and a planner that ignores this will confidently recommend the expensive
half of a design.

A convenient property falls out of the estimator: it reads a mean and a variance
at each timepoint and never needs to follow one well through time. So a
destructive assay loses no information here, only money.

The precision numbers are closed-form rather than simulated. The relative width
of a chi-squared interval depends on the well count alone, so "how many wells do
I need" is answered by inverting a function, not by a search over experiments
that would take hours and return the same answer with noise on it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from triboguard.abstention import minimum_groups

#: Below two wells there is no variance and no mechanism, at any price.
MINIMUM_USEFUL_WELLS = 2

#: Refuse to search past this. A design needing more wells than this per
#: condition is not a design a student laboratory will run, and reporting the
#: honest "out of reach" beats reporting a number nobody can act on.
MAXIMUM_SEARCHED_WELLS = 2000


class DesignError(ValueError):
    """Raised when a design or a budget cannot describe a real experiment."""


@dataclass(frozen=True)
class Assay:
    """What one measurement costs, and whether it consumes the well.

    ``per_well`` covers seeding and reagents for a well that will be followed.
    ``per_measurement`` covers reading one well once.
    """

    per_well: float
    per_measurement: float
    #: True for endpoint assays such as MTS, where reading the well destroys it
    #: and each timepoint needs its own wells.
    destructive: bool = False
    name: str = "assay"

    def __post_init__(self) -> None:
        if self.per_well < 0 or self.per_measurement < 0:
            raise DesignError("costs must be non-negative.")


@dataclass(frozen=True)
class Design:
    """One condition's worth of experiment: how many wells, read when."""

    wells: int
    times: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.wells < 1:
            raise DesignError(f"wells must be at least 1, got {self.wells}.")
        if len(self.times) < 2:
            raise DesignError("a design needs at least two timepoints to give a rate.")
        if len(set(self.times)) != len(self.times):
            raise DesignError("timepoints must be distinct.")
        if any(t < 0 for t in self.times):
            raise DesignError("timepoints must be non-negative.")

    @property
    def timepoints(self) -> int:
        return len(self.times)

    def wells_consumed(self, assay: Assay) -> int:
        """Wells the plate actually gives up.

        A destructive assay cannot re-read a well, so every timepoint needs its
        own set. This is where an endpoint assay becomes expensive, and it is
        invisible if cost is counted per measurement alone.
        """
        return self.wells * self.timepoints if assay.destructive else self.wells

    def cost(self, assay: Assay) -> float:
        measurements = self.wells * self.timepoints
        return self.wells_consumed(assay) * assay.per_well + measurements * assay.per_measurement


def relative_turnover_width(wells: int, confidence: float = 0.95) -> float:
    """Width of the turnover interval as a multiple of the estimate.

    Depends on the well count and nothing else, which is the whole reason this
    module can plan without simulating. Falls to infinity at one well, where
    there is no spread to measure.
    """
    if wells < MINIMUM_USEFUL_WELLS:
        return float("inf")
    if not 0 < confidence < 1:
        raise DesignError(f"confidence must be in (0, 1), got {confidence}.")
    from scipy import stats

    degrees = wells - 1
    tail = (1.0 - confidence) / 2.0
    upper = float(stats.chi2.ppf(1.0 - tail, degrees))
    lower = float(stats.chi2.ppf(tail, degrees))
    return degrees / lower - degrees / upper


def wells_for_width(target: float, confidence: float = 0.95) -> int | None:
    """Fewest wells reaching ``target`` relative width, or None if out of reach.

    The actionable form of the whole analysis: a researcher asks "how many wells
    do I need to tell killing from stalling?" and this answers in wells rather
    than in statistics.
    """
    if target <= 0:
        raise DesignError(f"target width must be positive, got {target}.")
    for wells in range(MINIMUM_USEFUL_WELLS, MAXIMUM_SEARCHED_WELLS + 1):
        if relative_turnover_width(wells, confidence) <= target:
            return wells
    return None


@dataclass(frozen=True)
class Option:
    """One way to spend more, and what it buys."""

    label: str
    design: Design
    cost: float
    added_cost: float
    relative_width: float
    #: Log-ratio of the widths before and after. A ratio because the width is a
    #: multiplicative quantity; a log because gains compose by addition and the
    #: first well helps far more than the fiftieth.
    information: float
    information_per_cost: float
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "wells": self.design.wells,
            "timepoints": list(self.design.times),
            "cost": self.cost,
            "added_cost": self.added_cost,
            "relative_width": (self.relative_width if math.isfinite(self.relative_width) else None),
            "information": self.information,
            "information_per_cost": self.information_per_cost,
            "note": self.note,
        }


def _information(before: float, after: float) -> float:
    if not math.isfinite(before):
        # Going from "no estimate at all" to any estimate is not a finite
        # improvement in ratio terms. Reporting infinity would sort correctly
        # and print badly, so it is capped at a large finite value that still
        # dominates every real option.
        return 100.0 if math.isfinite(after) else 0.0
    if after <= 0 or before <= 0:
        return 0.0
    return math.log(before / after)


def options(
    current: Design,
    assay: Assay,
    *,
    extra_wells: Sequence[int] = (1, 2, 3, 5, 10, 20),
    extra_times: Sequence[float] = (),
    confidence: float = 0.95,
) -> list[Option]:
    """Every augmentation worth considering, ranked by information per unit cost.

    Adding timepoints is included even though it is almost never chosen, because
    a planner that silently omitted the option would be asserting the conclusion
    rather than demonstrating it. The ranking is what shows the difference.
    """
    baseline_width = relative_turnover_width(current.wells, confidence)
    baseline_cost = current.cost(assay)
    found: list[Option] = []

    for extra in extra_wells:
        if extra < 1:
            raise DesignError(f"extra wells must be positive, got {extra}.")
        design = Design(wells=current.wells + extra, times=current.times)
        found.append(
            _option(
                f"add {extra} well{'s' if extra > 1 else ''} per condition",
                design,
                assay,
                baseline_width,
                baseline_cost,
                confidence,
            )
        )

    for moment in extra_times:
        if moment in current.times:
            continue
        design = Design(wells=current.wells, times=tuple(sorted((*current.times, moment))))
        option = _option(
            f"add a timepoint at {moment:g} h",
            design,
            assay,
            baseline_width,
            baseline_cost,
            confidence,
        )
        found.append(
            Option(
                **{
                    **option.__dict__,
                    "note": (
                        "sharpens the growth curve but not the mechanism: the "
                        "turnover interval is set by the well count alone"
                    ),
                }
            )
        )

    found.sort(key=lambda item: item.information_per_cost, reverse=True)
    return found


def _option(
    label: str,
    design: Design,
    assay: Assay,
    baseline_width: float,
    baseline_cost: float,
    confidence: float,
) -> Option:
    width = relative_turnover_width(design.wells, confidence)
    cost = design.cost(assay)
    added = cost - baseline_cost
    information = _information(baseline_width, width)
    # A free improvement would divide by zero. It cannot happen for a real
    # assay, but a zero-cost assay is a legitimate way to ask "what would I
    # learn if money were no object", and it should answer rather than crash.
    per_cost = information / added if added > 0 else (information if information else 0.0)
    return Option(
        label=label,
        design=design,
        cost=cost,
        added_cost=added,
        relative_width=width,
        information=information,
        information_per_cost=per_cost,
    )


def recommend(
    current: Design,
    assay: Assay,
    *,
    target_width: float | None = None,
    confidence: float = 0.95,
    calibration_confidence: float = 0.9,
    **kwargs: Any,
) -> dict[str, Any]:
    """The next measurement to take, with its price and a plain-language reason."""
    ranked = options(current, assay, confidence=confidence, **kwargs)
    if not ranked:
        raise DesignError("no augmentation options were generated.")
    best = ranked[0]
    current_width = relative_turnover_width(current.wells, confidence)
    needed = wells_for_width(target_width, confidence) if target_width else None
    report: dict[str, Any] = {
        "current": {
            "wells": current.wells,
            "timepoints": list(current.times),
            "cost": current.cost(assay),
            "wells_consumed": current.wells_consumed(assay),
            "relative_turnover_width": (current_width if math.isfinite(current_width) else None),
            "mechanism_estimable": current.wells >= MINIMUM_USEFUL_WELLS,
        },
        "assay": {
            "name": assay.name,
            "destructive": assay.destructive,
            "per_well": assay.per_well,
            "per_measurement": assay.per_measurement,
        },
        "recommended": best.as_dict(),
        "options": [option.as_dict() for option in ranked],
        "calibration_experiments_needed": minimum_groups(calibration_confidence),
        "statement": _statement(current, assay, best, needed, target_width),
    }
    if target_width is not None:
        report["target"] = {
            "relative_width": target_width,
            "wells_required": needed,
            "reachable": needed is not None,
            "cost_at_requirement": (
                Design(wells=needed, times=current.times).cost(assay)
                if needed is not None
                else None
            ),
        }
    return report


def _statement(
    current: Design,
    assay: Assay,
    best: Option,
    needed: int | None,
    target_width: float | None,
) -> str:
    """The recommendation in words a lab notebook could take instructions from."""
    parts: list[str] = []
    if current.wells < MINIMUM_USEFUL_WELLS:
        parts.append(
            f"With {current.wells} well per condition the mechanism cannot be "
            "estimated at all, at any number of timepoints."
        )
    parts.append(f"The best next spend is to {best.label}.")
    if assay.destructive:
        parts.append(
            f"{assay.name} destroys the well it reads, so each timepoint needs its "
            f"own wells: this design consumes {best.design.wells_consumed(assay)} "
            "wells rather than "
            f"{best.design.wells}."
        )
    if target_width is not None:
        if needed is None:
            parts.append(
                f"A relative width of {target_width:g} is out of reach below "
                f"{MAXIMUM_SEARCHED_WELLS} wells per condition."
            )
        else:
            parts.append(
                f"Reaching a relative width of {target_width:g} needs {needed} wells per condition."
            )
    return " ".join(parts)
