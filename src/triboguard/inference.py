"""Recovering birth and death rates from counted wells, with honest uncertainty.

Two channels carry the answer and they are not equally reliable.

The **mean** trajectory determines ``net = birth - death``. It is estimated from
an average over wells, so its precision improves quickly with replicates and it
is usually the easy half.

The **variance** across wells determines ``turnover = birth + death``. A sample
variance from *r* wells has *r - 1* degrees of freedom, so three wells buy two,
and the estimate is correspondingly terrible. Since the mechanism is exactly the
part that only turnover can see, the uncertainty on the mechanism is dominated
by the replicate count and barely improves with more timepoints.

That asymmetry is the point. A study can pin its growth curve down beautifully
and still have no idea whether the drug kills or stalls, and reporting a single
mechanism label from such a study is where unsupported conclusions come from.

Estimation is method-of-moments rather than maximum likelihood: each step is a
formula whose reference value :mod:`triboguard.kinetics` already tests, so a
wrong number here is traceable to a line rather than to an optimiser that
converged somewhere unhelpful.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from triboguard import kinetics
from triboguard.kinetics import BirthDeath, KineticsError

#: Fewer wells than this and the across-well variance has no degrees of freedom
#: at all, so turnover -- and therefore the mechanism -- is not estimable.
MINIMUM_REPLICATES_FOR_VARIANCE = 2

#: Default bootstrap size. Resampling is over wells, because the well is the
#: independent unit; resampling timepoints would treat one trajectory as many.
RESAMPLES = 2000


class InferenceError(ValueError):
    """Raised when an experiment cannot support the estimate being asked for."""


@dataclass(frozen=True)
class RateEstimate:
    """Fitted rates for one condition, with intervals that admit their limits."""

    birth: float
    death: float
    net: float
    turnover: float
    initial: float
    net_interval: tuple[float, float]
    turnover_interval: tuple[float, float]
    replicates: int
    timepoints: int
    #: True when the observed spread was too small to be consistent with the
    #: fitted mean and turnover had to be clamped to |net|. Under-dispersion is
    #: not a rounding detail: it means the data contradict the model, and any
    #: mechanism read off such a fit is reading off the clamp.
    variance_floor_hit: bool
    #: False when there is only one well, in which case death and birth are not
    #: separable at all and the reported split is a convention, not a finding.
    turnover_estimable: bool

    @property
    def rates(self) -> BirthDeath:
        return BirthDeath(birth=self.birth, death=self.death)

    def as_dict(self) -> dict[str, Any]:
        return {
            "birth": self.birth,
            "death": self.death,
            "net": self.net,
            "turnover": self.turnover,
            "initial": self.initial,
            "net_interval": list(self.net_interval),
            # The upper bound is infinite when there is only one well, which JSON
            # cannot represent. None reads as "unbounded", which is the fact.
            "turnover_interval": [
                self.turnover_interval[0],
                self.turnover_interval[1] if np.isfinite(self.turnover_interval[1]) else None,
            ],
            "replicates": self.replicates,
            "timepoints": self.timepoints,
            "variance_floor_hit": self.variance_floor_hit,
            "turnover_estimable": self.turnover_estimable,
        }


def _validate(counts: np.ndarray, times: np.ndarray) -> None:
    if counts.ndim != 2:
        raise InferenceError(f"counts must be (replicates, times), got shape {counts.shape}.")
    if times.ndim != 1 or times.size != counts.shape[1]:
        raise InferenceError("times must be one per column of counts.")
    if times.size < 2:
        raise InferenceError("at least two timepoints are needed to estimate a rate.")
    if np.any(np.diff(times) <= 0):
        raise InferenceError("times must be strictly increasing.")
    if np.any(counts < 0):
        raise InferenceError("counts must be non-negative.")


def _fit_net(counts: np.ndarray, times: np.ndarray, initial: float | None) -> tuple[float, float]:
    """Slope of log mean count against time, and the implied starting size.

    Taking the log of the mean rather than the mean of the logs is deliberate:
    the model states ``E[N] = N0 exp(net t)``, and ``E[log N] < log E[N]`` by
    Jensen, so averaging logs would bias the growth rate downward -- most in the
    small, noisy populations where the bias is least affordable.
    """
    means = counts.mean(axis=0)
    usable = means > 0
    if usable.sum() < 2:
        raise InferenceError(
            "fewer than two timepoints have any surviving cells; no rate is estimable."
        )
    t = times[usable]
    y = np.log(means[usable])
    if initial is not None:
        if initial <= 0:
            raise InferenceError(f"initial must be positive, got {initial}.")
        # The seeding density is known from the protocol, so the intercept is
        # not a free parameter and fitting it would spend information on
        # something already measured.
        net = float(np.sum((y - np.log(initial)) * t) / np.sum(t * t)) if np.any(t) else 0.0
        return net, float(initial)
    slope, intercept = np.polyfit(t, y, 1)
    return float(slope), float(np.exp(intercept))


def _fit_turnover(
    counts: np.ndarray, times: np.ndarray, net: float, initial: float
) -> tuple[float, bool]:
    """Least-squares turnover from the across-well variance, given ``net``.

    ``variance(t)`` is linear in turnover once ``net`` is fixed, so this is a
    one-parameter regression through the origin rather than an optimisation.
    """
    replicates = counts.shape[0]
    if replicates < MINIMUM_REPLICATES_FOR_VARIANCE:
        # One well has no spread to measure. Turnover is unidentifiable; the
        # caller is told rather than handed a number that looks like an estimate.
        return abs(net), True
    observed = counts.var(axis=0, ddof=1)
    # Predicted variance per unit of turnover, from the same expression the
    # forward model uses.
    unit = kinetics.variance_count(_unit_turnover(net), initial, times)
    denominator = float(np.sum(unit * unit))
    if denominator <= 0:
        return abs(net), True
    turnover = float(np.sum(observed * unit) / denominator)
    # birth and death are both non-negative, so turnover cannot fall below the
    # magnitude of net. Hitting the floor means the wells agreed with each other
    # more than the model allows.
    if turnover < abs(net):
        return abs(net), True
    return turnover, False


def _unit_turnover(net: float) -> BirthDeath:
    """Rates with the given ``net`` and a turnover of exactly 1.

    Used only to evaluate the variance expression at unit turnover, so the
    regression coefficient can be read straight off.
    """
    birth = (net + 1.0) / 2.0
    death = (1.0 - net) / 2.0
    if birth < 0 or death < 0:
        # |net| > 1 per hour is not a cell culture; guard rather than construct
        # an invalid parameter pair.
        raise InferenceError(f"net rate {net} is outside the range this model describes.")
    return BirthDeath(birth=birth, death=death)


def _point_estimate(
    counts: np.ndarray, times: np.ndarray, initial: float | None
) -> tuple[float, float, float, bool]:
    net, start = _fit_net(counts, times, initial)
    turnover, floored = _fit_turnover(counts, times, net, start)
    return net, turnover, start, floored


def estimate_rates(
    counts: np.ndarray | Sequence[Sequence[float]],
    times: Sequence[float],
    *,
    initial: float | None = None,
    resamples: int = RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> RateEstimate:
    """Fit birth and death for one condition from its replicate wells.

    ``initial`` is the seeding density if the protocol recorded it. Supplying it
    removes a fitted parameter and tightens everything downstream; omitting it
    is supported because published figures often do not report it.
    """
    array = np.asarray(counts, dtype=float)
    schedule = np.asarray(times, dtype=float)
    _validate(array, schedule)
    if not 0 < confidence < 1:
        raise InferenceError(f"confidence must be in (0, 1), got {confidence}.")

    net, turnover, start, floored = _point_estimate(array, schedule, initial)
    replicates = int(array.shape[0])

    nets: list[float] = []
    rng = np.random.default_rng(seed)
    if resamples > 0 and replicates >= MINIMUM_REPLICATES_FOR_VARIANCE:
        for _ in range(resamples):
            picked = rng.integers(0, replicates, replicates)
            try:
                b_net, _ = _fit_net(array[picked], schedule, initial)
            except InferenceError:
                continue
            nets.append(b_net)

    tail = (1.0 - confidence) / 2.0
    net_interval = _interval(nets, net, tail)
    turnover_interval = turnover_bounds(turnover, replicates, confidence)

    birth = (turnover + net) / 2.0
    death = (turnover - net) / 2.0
    return RateEstimate(
        birth=max(birth, 0.0),
        death=max(death, 0.0),
        net=net,
        turnover=turnover,
        initial=start,
        net_interval=net_interval,
        turnover_interval=turnover_interval,
        replicates=replicates,
        timepoints=int(schedule.size),
        variance_floor_hit=floored,
        turnover_estimable=replicates >= MINIMUM_REPLICATES_FOR_VARIANCE,
    )


def turnover_bounds(
    turnover: float, replicates: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Interval for a variance-derived quantity, from chi-squared rather than bootstrap.

    Resampling wells with replacement is the wrong tool here, and quietly so. A
    variance needs the wells to differ, but a resample of three wells draws
    duplicates almost always -- measured at 100% of draws collapsing onto the
    variance floor -- so the bootstrap distribution becomes a point mass and the
    interval comes out *narrower* for three wells than for twenty-four. A tool
    whose job is to know when evidence is thin cannot report its highest
    confidence exactly where evidence is thinnest.

    The textbook interval does behave: with ``r`` wells, ``(r-1) s^2 / sigma^2``
    is chi-squared on ``r-1`` degrees of freedom, which widens without limit as
    ``r`` falls to two.

    Degrees of freedom come from the wells alone. Extra timepoints are measured
    on those same wells and their variances are strongly correlated, so counting
    them would inflate the degrees of freedom and re-introduce the
    overconfidence this exists to remove. Conservative on purpose.
    """
    from scipy import stats

    if replicates < MINIMUM_REPLICATES_FOR_VARIANCE:
        # No spread to measure at all: bounded below by the floor, unbounded above.
        return (turnover, float("inf"))
    degrees = replicates - 1
    tail = (1.0 - confidence) / 2.0
    upper_chi = float(stats.chi2.ppf(1.0 - tail, degrees))
    lower_chi = float(stats.chi2.ppf(tail, degrees))
    return (turnover * degrees / upper_chi, turnover * degrees / lower_chi)


def _interval(draws: list[float], fallback: float, tail: float) -> tuple[float, float]:
    """Percentile interval, or a degenerate one when there was nothing to resample."""
    if not draws:
        return (fallback, fallback)
    values = np.asarray(draws, dtype=float)
    return (
        float(np.percentile(values, 100 * tail)),
        float(np.percentile(values, 100 * (1 - tail))),
    )


def mechanism_distribution(
    control: np.ndarray | Sequence[Sequence[float]],
    treated: np.ndarray | Sequence[Sequence[float]],
    times: Sequence[float],
    *,
    initial: float | None = None,
    threshold: float = kinetics.EFFECT_THRESHOLD,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    """How often each mechanism survives resampling of the wells.

    Returns a distribution rather than a label. A study whose data cannot
    separate killing from stalling will show mass on both, and that spread is
    the quantity the abstention layer thresholds -- collapsing it to an argmax
    here would throw away the only evidence that the answer is unclear.

    Control and treated wells are resampled independently, because they are
    different wells; pairing them would invent a correlation the plate does not
    have.
    """
    control_array = np.asarray(control, dtype=float)
    treated_array = np.asarray(treated, dtype=float)
    schedule = np.asarray(times, dtype=float)
    _validate(control_array, schedule)
    _validate(treated_array, schedule)

    point = _mechanism_of(control_array, treated_array, schedule, initial, threshold)
    counts = dict.fromkeys(kinetics.MECHANISMS, 0)
    rng = np.random.default_rng(seed)

    # The two rates are drawn from different laws because they are estimated
    # from different things. `net` comes from an average, which the bootstrap
    # handles well. `turnover` comes from a variance, which the bootstrap
    # handles catastrophically at small well counts -- see turnover_bounds --
    # so it is drawn from its chi-squared sampling distribution instead.
    control_fit = _point_estimate(control_array, schedule, initial)
    treated_fit = _point_estimate(treated_array, schedule, initial)
    drawn = 0
    for _ in range(max(resamples, 0)):
        try:
            before = _draw_rates(control_array, schedule, initial, control_fit[1], rng)
            after = _draw_rates(treated_array, schedule, initial, treated_fit[1], rng)
        except InferenceError:
            continue
        counts[kinetics.classify(before, after, threshold=threshold)] += 1
        drawn += 1
    frequencies = (
        {name: counts[name] / drawn for name in kinetics.MECHANISMS}
        if drawn
        else dict.fromkeys(kinetics.MECHANISMS, 0.0)
    )
    estimable = (
        control_array.shape[0] >= MINIMUM_REPLICATES_FOR_VARIANCE
        and treated_array.shape[0] >= MINIMUM_REPLICATES_FOR_VARIANCE
    )
    return {
        "point_estimate": point,
        "frequencies": frequencies,
        "draws": drawn,
        "turnover_estimable": estimable,
        "threshold": threshold,
        "note": (
            "net is resampled over wells; turnover is drawn from its chi-squared "
            "sampling law, because bootstrapping a variance from a handful of "
            "wells collapses onto the variance floor and understates the doubt. "
            "These are sampling frequencies, not calibrated coverage; the "
            "abstention layer is what turns them into a set with a guarantee."
        ),
    }


def _draw_rates(
    counts: np.ndarray,
    times: np.ndarray,
    initial: float | None,
    turnover: float,
    rng: np.random.Generator,
) -> BirthDeath:
    """One plausible rate pair, given what this experiment could pin down.

    ``net`` is resampled over wells. ``turnover`` is scaled by a chi-squared
    draw, which is the sampling law of a variance and is what makes a three-well
    experiment look as uncertain as it is.
    """
    replicates = counts.shape[0]
    picked = rng.integers(0, replicates, replicates)
    net, _ = _fit_net(counts[picked], times, initial)
    degrees = replicates - 1
    if degrees >= 1:
        draw = float(rng.chisquare(degrees))
        # A vanishing draw means "the data are consistent with enormous
        # turnover", which is true but numerically unusable; cap it rather than
        # divide by ~0 and produce an infinite rate.
        turnover = turnover * degrees / max(draw, 1e-6)
    return _rates_from(net, max(turnover, abs(net)))


def _mechanism_of(
    control: np.ndarray,
    treated: np.ndarray,
    times: np.ndarray,
    initial: float | None,
    threshold: float,
) -> str:
    control_net, control_turnover, _, _ = _point_estimate(control, times, initial)
    treated_net, treated_turnover, _, _ = _point_estimate(treated, times, initial)
    try:
        before = _rates_from(control_net, control_turnover)
        after = _rates_from(treated_net, treated_turnover)
    except KineticsError as error:  # pragma: no cover - guarded by clamping above
        raise InferenceError(str(error)) from error
    return kinetics.classify(before, after, threshold=threshold)


def _rates_from(net: float, turnover: float) -> BirthDeath:
    return BirthDeath(
        birth=max((turnover + net) / 2.0, 0.0), death=max((turnover - net) / 2.0, 0.0)
    )
