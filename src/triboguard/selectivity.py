"""Selectivity that comes from the cells dividing, not from the compound.

The standard way to show a compound is selective for cancer is to run it against
a cancer cell line and against normal cells, and report that the cancer cells
died and the normal ones did not. The source study does exactly this: RAW264.7
against mouse peritoneal lymphocytes.

The two populations differ in a way that has nothing to do with the compound.
A cancer line divides continuously; primary lymphocytes in culture largely do
not. And "% cytotoxicity" is not measured against the starting population -- it
is measured against an untreated control read **at the same time**, which has
been dividing all day in the cancer plate and sitting still in the lymphocyte
plate.

That asymmetry is enough on its own. A compound that stops division and kills
nothing at all removes cells the cancer control *would have gained* and produces
a large "% cytotoxicity"; the same compound applied to cells that were not going
to divide anyway produces none. Perfect apparent selectivity, zero real
selectivity, from a compound with no selective action whatsoever.

This module puts a number on that. Everything here is closed-form from the
birth-death mean, which depends on the net rate alone -- so none of it needs a
replicate count, and it can therefore be applied to a study that never reported
one.

**What is assumed, and where it can break.** MTS absorbance is treated as
proportional to the number of viable cells, which is the assumption every
paper using it makes implicitly when it calls the readout "% cytotoxicity". It
is not guaranteed: the assay reports mitochondrial dehydrogenase activity, so a
compound that slows metabolism per cell without touching a single one produces
the same fall. That is a *second* independent route to a false selectivity claim
and this module does not model it -- it is named here so the result is not read
as the only way the conclusion can fail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from triboguard.kinetics import BirthDeath, apply_treatment

#: A population whose net rate is within this of zero is treated as static.
#: Primary lymphocytes in short-term culture are the case this exists for.
STATIC_TOLERANCE = 1e-9


class SelectivityError(ValueError):
    """Raised when a selectivity comparison cannot be posed as asked."""


@dataclass(frozen=True)
class Population:
    """One arm of a selectivity experiment: what the cells do when left alone."""

    name: str
    control: BirthDeath

    @property
    def divides(self) -> bool:
        return self.control.birth > STATIC_TOLERANCE


def signal_ratio(control: BirthDeath, treated: BirthDeath, hours: float) -> float:
    """Treated signal as a fraction of the same-time control signal.

    Under the proportionality assumption above this is the ratio of viable cell
    numbers, and because the birth-death mean depends on the net rate alone it
    collapses to a single exponential in the *difference* of net rates.
    """
    if not math.isfinite(hours) or hours < 0:
        raise SelectivityError(f"hours must be finite and non-negative, got {hours}.")
    return math.exp((treated.net - control.net) * hours)


def apparent_cytotoxicity(control: BirthDeath, treated: BirthDeath, hours: float) -> float:
    """What a plate reader reports as "% cytotoxicity", as a fraction.

    Named *apparent* throughout because nothing here establishes that a cell
    died. It is one minus the same-time signal ratio, which is the definition
    the field uses and the definition that makes division rate matter.
    """
    return 1.0 - signal_ratio(control, treated, hours)


def required_net_change(cytotoxicity: float, hours: float) -> float:
    """The change in net rate that reproduces an observed "% cytotoxicity".

    Inverts :func:`apparent_cytotoxicity`. Returned as a negative number: the
    treated population grows more slowly, by whatever combination of slowed
    division and added death.
    """
    if not 0 <= cytotoxicity < 1 or not math.isfinite(cytotoxicity):
        raise SelectivityError(f"cytotoxicity must lie in [0, 1), got {cytotoxicity}.")
    if not math.isfinite(hours) or hours <= 0:
        raise SelectivityError(f"hours must be finite and positive, got {hours}.")
    return math.log(1.0 - cytotoxicity) / hours


def birth_rate_for_pure_stalling(cytotoxicity: float, hours: float) -> float:
    """Slowest-dividing control for which stalling alone explains the result.

    A purely cytostatic compound can at most stop every division, so it can
    lower the net rate by exactly the birth rate and no further. Setting that
    limit equal to the required change gives the smallest birth rate a control
    can have and still produce this "% cytotoxicity" with no cell death at all.

    Any control dividing faster than this has an explanation for the reported
    number that involves nothing dying.
    """
    return -required_net_change(cytotoxicity, hours)


def stalling_fraction(control: BirthDeath, cytotoxicity: float, hours: float) -> float | None:
    """Share of divisions that must stop to reproduce the result without killing.

    ``None`` when the control does not divide fast enough for stalling alone to
    account for the observation -- which is the case where the reported number
    really does require cells to die.
    """
    needed = birth_rate_for_pure_stalling(cytotoxicity, hours)
    if control.birth <= 0:
        return None
    fraction = needed / control.birth
    return None if fraction > 1.0 else fraction


def pure_stalling_partner(control: BirthDeath, cytotoxicity: float, hours: float) -> BirthDeath:
    """The treated rates that reproduce the result by slowing division only.

    Raises when the control cannot divide fast enough, rather than returning a
    negative birth rate and letting the caller discover it later.
    """
    fraction = stalling_fraction(control, cytotoxicity, hours)
    if fraction is None:
        raise SelectivityError(
            f"A control dividing at {control.birth:.4f}/h cannot produce "
            f"{cytotoxicity:.0%} in {hours:g} h by stalling alone; it would need at least "
            f"{birth_rate_for_pure_stalling(cytotoxicity, hours):.4f}/h."
        )
    return apply_treatment(control, birth_scale=1.0 - fraction)


def pure_killing_partner(control: BirthDeath, cytotoxicity: float, hours: float) -> BirthDeath:
    """The treated rates that reproduce the same result by killing only.

    Always exists: death is unbounded above, so any observation can be explained
    by enough killing. That asymmetry is the whole reason the two readings are
    not equally easy to rule out.
    """
    return apply_treatment(control, death_increase=-required_net_change(cytotoxicity, hours))


def illusion(
    cancer: Population,
    normal: Population,
    *,
    hours: float,
    birth_scale: float = 1.0,
    death_increase: float = 0.0,
) -> dict[str, Any]:
    """Apply one identical, non-selective treatment to both arms and report it.

    The treatment is by construction the same compound at the same strength in
    both populations: the same fractional slowdown of division and the same
    added death rate. Any difference in the reported numbers is therefore
    manufactured by the experiment rather than caused by the compound.
    """
    treated = {
        arm.name: apply_treatment(
            arm.control, birth_scale=birth_scale, death_increase=death_increase
        )
        for arm in (cancer, normal)
    }
    reported = {
        arm.name: apparent_cytotoxicity(arm.control, treated[arm.name], hours)
        for arm in (cancer, normal)
    }
    gap = reported[cancer.name] - reported[normal.name]
    return {
        "hours": hours,
        "treatment": {"birth_scale": birth_scale, "death_increase": death_increase},
        "truly_selective": False,
        "arms": {
            arm.name: {
                "control_birth": arm.control.birth,
                "control_death": arm.control.death,
                "control_doubling_hours": arm.control.doubling_hours,
                "divides": arm.divides,
                "apparent_cytotoxicity": reported[arm.name],
                "cells_actually_killed": _killed_fraction(arm.control, treated[arm.name], hours),
            }
            for arm in (cancer, normal)
        },
        "apparent_gap": gap,
        # A ratio is what the field reports, and it is exactly the quantity that
        # blows up when the denominator is a population that was not dividing.
        # Reported as None rather than as a large number, because "infinite" is
        # the honest reading and a big float invites being averaged.
        "apparent_ratio": (
            None
            if abs(reported[normal.name]) < STATIC_TOLERANCE
            else reported[cancer.name] / reported[normal.name]
        ),
    }


def _killed_fraction(control: BirthDeath, treated: BirthDeath, hours: float) -> float:
    """Share of the cells present at the start that the treatment actually kills.

    Separate from the apparent number because this is the quantity the claim is
    about and the two come apart completely. With no added death it is zero
    however large the apparent cytotoxicity gets.
    """
    extra_death = treated.death - control.death
    if extra_death <= 0:
        return 0.0
    return 1.0 - math.exp(-extra_death * hours)


def explaining_family(
    control: BirthDeath, cytotoxicity: float, hours: float, *, steps: int = 11
) -> list[dict[str, Any]]:
    """Every mechanism consistent with one reported "% cytotoxicity".

    The mean trajectory fixes the net rate and nothing else, so the reported
    number pins down one combination and leaves a one-parameter family behind.
    This walks that family from the pure-stalling end to the pure-killing end so
    a reader can see the range rather than take the claim on trust.
    """
    if steps < 2:
        raise SelectivityError(f"steps must be at least 2, got {steps}.")
    if control.birth <= 0:
        # Nothing to stall, so there is no family: one explanation and no trade
        # to walk along. Returning several identical rows would suggest a range
        # that does not exist.
        only = _family_row(control, control.birth, cytotoxicity, hours, 1.0)
        # A non-dividing control always admits the killing explanation, so this
        # cannot be None; asserting it keeps the type honest without inventing a
        # branch that no input can reach.
        assert only is not None
        return [only]
    lowest = stalling_fraction(control, cytotoxicity, hours)
    if lowest is None:
        # Stalling cannot reach it, so the family starts at "stop every division"
        # and the remainder must be killing.
        lowest = 1.0
    rows = []
    for index in range(steps):
        share = lowest * index / (steps - 1)
        birth_scale = 1.0 - share
        row = _family_row(control, control.birth * birth_scale, cytotoxicity, hours, birth_scale)
        if row is not None:
            rows.append(row)
    return rows


def _family_row(
    control: BirthDeath,
    treated_birth: float,
    cytotoxicity: float,
    hours: float,
    birth_scale: float,
) -> dict[str, Any] | None:
    """One member of the family, or ``None`` if it would need a negative death rate."""
    treated_death = treated_birth - (control.net + required_net_change(cytotoxicity, hours))
    if treated_death < 0:
        return None
    treated = BirthDeath(birth=treated_birth, death=treated_death)
    return {
        "birth_scale": birth_scale,
        "death_rate": treated.death,
        "cells_actually_killed": _killed_fraction(control, treated, hours),
        "apparent_cytotoxicity": apparent_cytotoxicity(control, treated, hours),
    }


def time_course_consistency(points: list[tuple[float, float]]) -> dict[str, Any]:
    """Whether one unchanging treatment could produce a reported time course.

    A treatment with constant rates gives an apparent cytotoxicity of
    ``1 - exp(dr * t)``, which is monotone in time and implies the *same* ``dr``
    at every timepoint. Recovering a different ``dr`` from each reported point
    means no single constant-rate treatment reproduces the series -- the effect
    changed over the experiment, or the numbers disagree with each other.
    """
    if len(points) < 2:
        raise SelectivityError("a time course needs at least two points.")
    implied = []
    for hours, cytotoxicity in points:
        implied.append(
            {
                "hours": hours,
                "cytotoxicity": cytotoxicity,
                "implied_net_change": required_net_change(cytotoxicity, hours),
            }
        )
    rates = [row["implied_net_change"] for row in implied]
    strongest, weakest = min(rates), max(rates)
    # Both are negative; the spread is what a constant-rate reading has to explain.
    spread = strongest / weakest if weakest != 0 else math.inf
    monotone = all(
        b["cytotoxicity"] >= a["cytotoxicity"]
        for a, b in zip(
            sorted(implied, key=lambda r: r["hours"]),
            sorted(implied, key=lambda r: r["hours"])[1:],
            strict=False,
        )
    )
    return {
        "points": implied,
        "monotone_in_time": monotone,
        "implied_rate_spread": spread,
        "constant_rate_consistent": bool(spread <= 1.25 and monotone),
    }
