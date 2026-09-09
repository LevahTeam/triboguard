"""The birth-death model, and the reason mechanism inference is hard at all.

A population where cells divide at rate ``birth`` and die at rate ``death`` has
mean size

    m(t) = N0 * exp((birth - death) * t)

The mean depends on the *difference* of the two rates and on nothing else. So a
treatment that halves division and one that doubles death can produce
byte-identical mean trajectories. No amount of averaging separates them, and no
schedule of extra timepoints helps, because every timepoint measures the same
degenerate combination. This is a property of the model, not a shortage of data.

What breaks the degeneracy is the *variance*:

    v(t) = N0 * (birth + death) / (birth - death)
           * exp((birth - death) * t) * (exp((birth - death) * t) - 1)

which depends on the *sum*. Knowing the difference and the sum determines both
rates. So the information needed to name a mechanism lives in the spread across
replicate wells, not in the average of them -- which is exactly the quantity a
bulk assay throws away by averaging, and exactly the quantity an experiment with
two wells per condition cannot estimate.

That is the whole argument for this package: replicates are not a statistical
nicety here, they are the only channel through which the mechanism is visible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Mechanism labels. "mixed" means both rates moved, which is the honest answer
#: more often than either pure label.
CYTOTOXIC = "cytotoxic"
CYTOSTATIC = "cytostatic"
MIXED = "mixed"
NO_EFFECT = "no_effect"
MECHANISMS = (NO_EFFECT, CYTOSTATIC, CYTOTOXIC, MIXED)

#: Below this relative change a rate counts as unchanged. A treatment that moves
#: a rate by 2% is not "cytostatic"; it is noise with a label attached.
EFFECT_THRESHOLD = 0.15


class KineticsError(ValueError):
    """Raised when a rate or a schedule cannot describe a real experiment."""


@dataclass(frozen=True)
class BirthDeath:
    """Per-capita division and death rates, in units of 1/hour."""

    birth: float
    death: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.birth) or self.birth < 0:
            raise KineticsError(f"birth rate must be finite and non-negative, got {self.birth}.")
        if not math.isfinite(self.death) or self.death < 0:
            raise KineticsError(f"death rate must be finite and non-negative, got {self.death}.")

    @property
    def net(self) -> float:
        """Growth rate. This is all the mean trajectory can tell you."""
        return self.birth - self.death

    @property
    def turnover(self) -> float:
        """Total event rate. Only the variance across replicates sees this."""
        return self.birth + self.death

    @property
    def doubling_hours(self) -> float | None:
        """Hours to double, or None for a population that is not growing."""
        return math.log(2.0) / self.net if self.net > 0 else None


def mean_count(params: BirthDeath, initial: float, times: Sequence[float]) -> np.ndarray:
    """Expected population size. A function of ``net`` alone -- that is the problem."""
    t = np.asarray(times, dtype=float)
    if np.any(t < 0):
        raise KineticsError("times must be non-negative.")
    return float(initial) * np.exp(params.net * t)


def variance_count(params: BirthDeath, initial: float, times: Sequence[float]) -> np.ndarray:
    """Variance across independent wells. This is where the mechanism hides.

    The ``birth == death`` case is the limit of the general expression and has
    to be written separately, because the general one divides by ``net``.
    """
    t = np.asarray(times, dtype=float)
    if np.any(t < 0):
        raise KineticsError("times must be non-negative.")
    n0 = float(initial)
    net, turnover = params.net, params.turnover
    if abs(net) < 1e-12:
        return n0 * turnover * t
    growth = np.exp(net * t)
    return n0 * (turnover / net) * growth * (growth - 1.0)


def simulate(
    params: BirthDeath,
    initial: int,
    times: Sequence[float],
    *,
    replicates: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Exact stochastic trajectories, shape ``(replicates, len(times))``.

    Not a Gillespie loop. A linear birth-death process started from ``n``
    ancestors is the sum of ``n`` independent single-ancestor processes, and the
    single-ancestor law at time ``t`` is known in closed form: extinct with
    probability ``alpha``, otherwise geometric on {1, 2, ...}. Sampling that
    directly is exact and costs one draw per ancestor per interval, instead of
    one draw per division event -- which matters because a well holds 10^5 cells
    and a Gillespie simulation of it would dominate the runtime of everything
    here.
    """
    generator = np.random.default_rng() if rng is None else rng
    schedule = np.asarray(times, dtype=float)
    if schedule.ndim != 1 or schedule.size == 0:
        raise KineticsError("times must be a non-empty one-dimensional sequence.")
    if np.any(np.diff(schedule) <= 0):
        raise KineticsError("times must be strictly increasing.")
    if schedule[0] < 0:
        raise KineticsError("times must be non-negative.")
    if replicates < 1:
        raise KineticsError(f"replicates must be at least 1, got {replicates}.")
    if initial < 1:
        raise KineticsError(f"initial must be at least 1, got {initial}.")

    counts = np.zeros((replicates, schedule.size), dtype=np.int64)
    # Each well is walked forward interval by interval, so the state at one
    # timepoint is the ancestor set for the next. Sampling every timepoint from
    # time zero instead would give correct marginals and impossible
    # trajectories: a well could shrink and then exceed its own earlier size.
    current = np.full(replicates, int(initial), dtype=np.int64)
    previous = 0.0
    for index, moment in enumerate(schedule):
        current = _advance(params, current, moment - previous, generator)
        counts[:, index] = current
        previous = float(moment)
    return counts


def _advance(
    params: BirthDeath, populations: np.ndarray, duration: float, rng: np.random.Generator
) -> np.ndarray:
    """One exact step of the process for each well."""
    if duration <= 0:
        return populations.copy()
    alpha, beta = _extinction_and_geometric(params, duration)
    out = np.empty_like(populations)
    for index, size in enumerate(populations):
        if size <= 0:
            out[index] = 0
            continue
        # Of `size` ancestors, those that leave descendants each contribute a
        # geometric number of them.
        survivors = int(rng.binomial(int(size), 1.0 - alpha))
        if survivors == 0:
            out[index] = 0
            continue
        out[index] = int(rng.geometric(1.0 - beta, size=survivors).sum())
    return out


def _extinction_and_geometric(params: BirthDeath, duration: float) -> tuple[float, float]:
    """``(alpha, beta)`` for the single-ancestor law after ``duration``."""
    birth, death = params.birth, params.death
    net = params.net
    if abs(net) < 1e-12:
        # Critical case: both probabilities converge to the same value.
        shared = birth * duration / (1.0 + birth * duration)
        return shared, shared
    growth = math.exp(net * duration)
    # Negative for any declining population -- with birth 0 it is exactly -death
    # -- and the numerators are negative there too, so both ratios stay in
    # [0, 1]. An earlier version rejected a negative denominator as degenerate
    # and refused to simulate pure killing, which is the case this project cares
    # about most. It cannot reach zero: the two branches of `net` are handled
    # above, and away from `birth == death` the expression keeps its sign.
    denominator = birth * growth - death
    if denominator == 0.0:
        raise KineticsError("Degenerate birth-death parameters produced an undefined law.")
    alpha = death * (growth - 1.0) / denominator
    beta = birth * (growth - 1.0) / denominator
    # Floating point can push these a hair outside [0, 1); the samplers do not
    # tolerate that, and clipping is safer than a rare mid-run exception.
    return min(max(alpha, 0.0), 1.0), min(max(beta, 0.0), 1.0 - 1e-12)


def apply_treatment(
    control: BirthDeath, *, birth_scale: float = 1.0, death_increase: float = 0.0
) -> BirthDeath:
    """A treated population, described the way a biologist would describe it.

    ``birth_scale`` below 1 slows division (cytostatic). ``death_increase`` adds
    to the death rate (cytotoxic). Both together is the mixed case, which real
    compounds usually are.
    """
    if birth_scale < 0:
        raise KineticsError(f"birth_scale must be non-negative, got {birth_scale}.")
    if death_increase < 0:
        raise KineticsError(f"death_increase must be non-negative, got {death_increase}.")
    return BirthDeath(birth=control.birth * birth_scale, death=control.death + death_increase)


def classify(
    control: BirthDeath, treated: BirthDeath, *, threshold: float = EFFECT_THRESHOLD
) -> str:
    """Which mechanism a pair of rate sets represents.

    Used to label simulated ground truth, not to judge an inferred fit -- an
    inference has uncertainty and gets a *set* of labels, which is what the
    calibration layer is for.
    """
    if threshold < 0:
        raise KineticsError(f"threshold must be non-negative, got {threshold}.")
    birth_drop = (control.birth - treated.birth) / control.birth if control.birth > 0 else 0.0
    # Death is compared against the total event rate rather than against the
    # control death rate: a control that barely dies has a near-zero
    # denominator, and every treatment would look infinitely cytotoxic.
    scale = control.turnover if control.turnover > 0 else 1.0
    death_rise = (treated.death - control.death) / scale
    slowed = birth_drop > threshold
    killed = death_rise > threshold
    if slowed and killed:
        return MIXED
    if killed:
        return CYTOTOXIC
    if slowed:
        return CYTOSTATIC
    return NO_EFFECT


def indistinguishable_partner(observed: BirthDeath, *, birth: float) -> BirthDeath:
    """The rates that produce an identical mean trajectory at a different birth rate.

    Exists to make the identifiability problem concrete rather than assert it:
    for any birth rate you choose, there is a death rate giving the same mean
    curve. Returned so tests and figures can show the two curves lying on top of
    each other, and the variances pulling apart.
    """
    if birth < 0:
        raise KineticsError(f"birth must be non-negative, got {birth}.")
    death = birth - observed.net
    if death < 0:
        raise KineticsError(
            f"No non-negative death rate matches net {observed.net:.4f} at birth {birth:.4f}."
        )
    return BirthDeath(birth=birth, death=death)


def summarise(counts: np.ndarray) -> dict[str, Any]:
    """Mean and variance per timepoint, the two channels the fit will use."""
    array = np.asarray(counts, dtype=float)
    if array.ndim != 2:
        raise KineticsError(f"counts must be (replicates, times), got shape {array.shape}.")
    replicates = array.shape[0]
    return {
        "replicates": replicates,
        "timepoints": int(array.shape[1]),
        "mean": array.mean(axis=0).tolist(),
        # ddof=1 needs at least two wells. One well gives a mean and no variance,
        # which is precisely the case where the mechanism is unidentifiable.
        "variance": (array.var(axis=0, ddof=1).tolist() if replicates > 1 else None),
        "variance_estimable": replicates > 1,
    }
