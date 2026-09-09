"""Turning a mechanism distribution into a set of answers with a guarantee.

The frequencies from :mod:`triboguard.inference` describe sampling spread. They
are not a promise: nothing in them says how often the true mechanism is the one
with the highest frequency. Split conformal prediction supplies that promise.

The construction is small. On experiments whose true mechanism is known, score
each by how little frequency the estimator gave the correct answer. Take a
quantile of those scores. On a new experiment, return every mechanism scoring
better than that quantile. Under exchangeability the returned set contains the
truth at least ``confidence`` of the time -- and it earns that by returning
*more than one label* when the data cannot separate them, which is the behaviour
this project exists to produce.

Two decisions matter more than the algorithm.

**The exchangeable unit is a whole experiment.** Timepoints from one trajectory
are not independent draws, and calibrating on them would manufacture a guarantee
out of correlated repeats. Records carrying the same ``group`` are collapsed to
one score before the quantile is taken.

**Too little calibration data returns everything.** Asking for 95% coverage from
five calibration experiments is asking for a promise the data cannot support.
The conformal quantile is undefined there, and the honest output is the full set
-- "this tells you nothing" -- rather than a narrower one that quietly breaks its
guarantee.

**And the calibration must match the design it is applied to.** Exchangeability
is not a formality here. Calibrating on thirty-well experiments and predicting
on three-well ones was measured at 53% coverage against a 90% target, with 45%
of verdicts confidently wrong -- worse than useless, because it is confident. A
three-well experiment is a different kind of object from a thirty-well one, and
the threshold that governs it has to be learned from its own kind. Calibration
is therefore stratified, and a stratum with too few experiments abstains
entirely rather than borrowing a threshold from a richer one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from triboguard import kinetics

#: Sets at least this large are reported as an explicit refusal to choose rather
#: than as a narrowed answer. Two of four mechanisms is still information; all
#: four is not.
UNINFORMATIVE_SET_SIZE = len(kinetics.MECHANISMS)


class AbstentionError(ValueError):
    """Raised when a calibration set cannot support the guarantee being asked for."""


@dataclass(frozen=True)
class CalibrationRecord:
    """One experiment whose true mechanism is known.

    ``group`` names the experiment. Several records may share one -- the same
    drug scored at two doses, say -- and they are collapsed before calibration
    because they are not independent evidence about the estimator.

    ``stratum`` names the experimental design, and defaults to the number of
    wells because that is what dominates how well the mechanism can be resolved.
    Records are only ever compared against others in their own stratum.
    """

    frequencies: Mapping[str, float]
    truth: str
    group: str
    stratum: str = ""

    @staticmethod
    def wells(replicates: int) -> str:
        """The conventional stratum name for a design with ``replicates`` wells."""
        return f"wells={replicates}"


@dataclass(frozen=True)
class Calibration:
    """Thresholds per experimental design, and an honest account of their worth."""

    thresholds: Mapping[str, float]
    confidence: float
    groups_per_stratum: Mapping[str, int]
    records: int

    @property
    def threshold(self) -> float:
        """The only threshold, for the common single-stratum case."""
        if len(self.thresholds) != 1:
            raise AbstentionError(
                "this calibration covers several designs; ask for a stratum by name."
            )
        return next(iter(self.thresholds.values()))

    @property
    def groups(self) -> int:
        return sum(self.groups_per_stratum.values())

    @property
    def degenerate(self) -> bool:
        """True when no stratum can support the requested confidence."""
        return all(not math.isfinite(value) for value in self.thresholds.values())

    def threshold_for(self, stratum: str) -> float:
        """The threshold governing ``stratum``, or infinity when it has none.

        An unknown stratum is not an error and must not silently fall back to
        another design's threshold. It returns infinity, which yields the full
        set -- the honest answer for an experiment unlike anything calibrated.
        """
        return self.thresholds.get(stratum, float("inf"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "thresholds": {
                name: (value if math.isfinite(value) else None)
                for name, value in self.thresholds.items()
            },
            "confidence": self.confidence,
            "groups_per_stratum": dict(self.groups_per_stratum),
            "records": self.records,
            "degenerate": self.degenerate,
            "note": (
                "Thresholds are per experimental design. A design with too few "
                "calibration experiments has no finite threshold and returns "
                "every mechanism, rather than borrowing a threshold from a "
                "richer design -- which was measured at 53% coverage against a "
                "90% target when tried."
            ),
        }


def _score(frequencies: Mapping[str, float], label: str) -> float:
    """How badly the estimator missed ``label``. Lower is better."""
    if label not in kinetics.MECHANISMS:
        raise AbstentionError(f"{label!r} is not one of {kinetics.MECHANISMS}.")
    return 1.0 - float(frequencies.get(label, 0.0))


def minimum_groups(confidence: float) -> int:
    """Fewest distinct experiments that can support ``confidence``.

    The conformal quantile needs ``ceil((n + 1) * confidence) <= n``. Below that
    there is no finite threshold, so the guarantee is only available from this
    many experiments upward. Exposed because "collect more calibration data" is
    actionable advice and "the threshold is infinite" is not.
    """
    if not 0 < confidence < 1:
        raise AbstentionError(f"confidence must be in (0, 1), got {confidence}.")
    # The closed form is n >= c / (1 - c), and in floating point that is wrong
    # exactly where it matters: 0.9 / (1 - 0.9) evaluates to 9.000000000000002,
    # whose ceiling is 10, so the advertised minimum would be one experiment
    # more than necessary at every round confidence anyone actually asks for.
    # The estimate seeds a short search and the definition decides.
    estimate = max(int(confidence / (1.0 - confidence)) - 2, 1)
    for count in range(estimate, estimate + 8):
        if math.ceil((count + 1) * confidence) <= count:
            return count
    raise AbstentionError(  # pragma: no cover - unreachable for confidence < 1
        f"no group count satisfies the conformal condition at confidence {confidence}."
    )


def calibrate(records: Iterable[CalibrationRecord], *, confidence: float = 0.9) -> Calibration:
    """Fit one conformal threshold per experimental design.

    Strata are fitted independently and never pooled. Pooling would let a
    plentiful design lend its threshold to a scarce one, which is precisely the
    substitution that produced 53% coverage at a 90% target.
    """
    if not 0 < confidence < 1:
        raise AbstentionError(f"confidence must be in (0, 1), got {confidence}.")
    collected = list(records)
    if not collected:
        raise AbstentionError("calibration needs at least one record.")

    # One score per experiment, per stratum. The worst record in a group is the
    # one kept: a drug scored at five doses is one piece of evidence about the
    # estimator, and averaging its easy doses against its hard one would buy a
    # tighter threshold with repetition rather than with information.
    worst: dict[str, dict[str, float]] = {}
    for record in collected:
        score = _score(record.frequencies, record.truth)
        bucket = worst.setdefault(record.stratum, {})
        bucket[record.group] = max(bucket.get(record.group, 0.0), score)

    thresholds: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for stratum, groups in worst.items():
        scores = np.array(sorted(groups.values()), dtype=float)
        count = scores.size
        sizes[stratum] = count
        rank = math.ceil((count + 1) * confidence)
        # No finite quantile exists below the minimum. Returning everything is
        # the only honest option; silently lowering the confidence to whatever
        # this many experiments can support would be a guarantee the caller did
        # not ask for and would not be told about.
        thresholds[stratum] = float("inf") if rank > count else float(scores[rank - 1])
    return Calibration(
        thresholds=thresholds,
        confidence=confidence,
        groups_per_stratum=sizes,
        records=len(collected),
    )


def predict_set(
    frequencies: Mapping[str, float],
    calibration: Calibration,
    *,
    stratum: str = "",
    turnover_estimable: bool = True,
) -> dict[str, Any]:
    """Every mechanism the evidence cannot rule out, under this design's threshold.

    ``stratum`` must name the same experimental design the calibration recorded
    for experiments like this one. A design the calibration never saw gets the
    full set rather than a neighbour's threshold.

    ``turnover_estimable`` is the escape hatch for an experiment with one well
    per condition. There, birth and death are not separable at all, so the
    estimator's frequencies describe a quantity it never measured. The full set
    is returned regardless of what the numbers say, because a calibrated
    threshold applied to an unidentified parameter is a guarantee about nothing.
    """
    if not turnover_estimable:
        return _verdict(
            list(kinetics.MECHANISMS),
            calibration,
            reason="turnover is not estimable from a single well per condition",
        )
    threshold = calibration.threshold_for(stratum)
    if not math.isfinite(threshold):
        return _verdict(
            list(kinetics.MECHANISMS),
            calibration,
            reason=(
                f"no threshold is calibrated for design {stratum!r}; too few "
                "calibration experiments share this design"
            ),
        )
    admitted = [name for name in kinetics.MECHANISMS if _score(frequencies, name) <= threshold]
    if not admitted:
        # Conformal sets can come out empty when an experiment is unlike
        # anything in calibration. Reporting nothing would read as "no mechanism
        # is possible", which is never the finding; the honest reading is that
        # this experiment is outside what the calibration covers.
        return _verdict(
            list(kinetics.MECHANISMS),
            calibration,
            reason=(
                "no mechanism cleared the threshold; this experiment is unlike the calibration set"
            ),
        )
    return _verdict(admitted, calibration, reason=None)


def _verdict(
    admitted: Sequence[str], calibration: Calibration, reason: str | None
) -> dict[str, Any]:
    ordered = [name for name in kinetics.MECHANISMS if name in set(admitted)]
    sufficient = len(ordered) == 1
    return {
        "mechanisms": ordered,
        "sufficient_evidence": sufficient,
        "uninformative": len(ordered) >= UNINFORMATIVE_SET_SIZE,
        "confidence": calibration.confidence,
        "confidence_note": (
            "coverage is guaranteed for experiments of this design, drawn like the calibration set"
        ),
        "reason": reason,
        "statement": _statement(ordered, sufficient, calibration.confidence),
    }


def _statement(mechanisms: Sequence[str], sufficient: bool, confidence: float) -> str:
    """The finding in words, phrased so it cannot be over-read."""
    percent = f"{confidence:.0%}"
    if sufficient:
        return (
            f"The evidence supports a {mechanisms[0].replace('_', ' ')} response, "
            f"at {percent} coverage."
        )
    if len(mechanisms) >= UNINFORMATIVE_SET_SIZE:
        return (
            "The evidence does not distinguish any mechanism from any other. No "
            "claim about killing versus growth inhibition is supported."
        )
    listed = " or ".join(name.replace("_", " ") for name in mechanisms)
    return (
        f"The evidence supports {listed}, at {percent} coverage, but does not "
        "single one out. A conclusion naming just one is not supported."
    )


def evaluate(
    records: Iterable[CalibrationRecord],
    calibration: Calibration,
) -> dict[str, Any]:
    """Empirical coverage and the costs paid for it, on held-out experiments.

    Coverage alone is trivial to achieve -- always return every mechanism -- so
    it is reported next to the set size that bought it and, more importantly,
    next to the rate of *confidently wrong* answers: a single-mechanism verdict
    that was incorrect. That is the number a researcher would be misled by, and
    the one a method claiming to prevent unsupported conclusions has to keep low.
    """
    held = list(records)
    if not held:
        raise AbstentionError("evaluation needs at least one record.")
    covered = 0
    sizes: list[int] = []
    abstained = 0
    confidently_wrong = 0
    for record in held:
        verdict = predict_set(record.frequencies, calibration, stratum=record.stratum)
        names = verdict["mechanisms"]
        sizes.append(len(names))
        if record.truth in names:
            covered += 1
        if not verdict["sufficient_evidence"]:
            abstained += 1
        elif names[0] != record.truth:
            confidently_wrong += 1
    total = len(held)
    decisive = total - abstained
    return {
        "experiments": total,
        "coverage": covered / total,
        "target_coverage": calibration.confidence,
        "mean_set_size": float(np.mean(sizes)),
        "abstention_rate": abstained / total,
        "confidently_wrong_rate": confidently_wrong / total,
        # The rate a researcher actually faces. They do not act on the
        # experiments where the system abstained; they act on the ones where it
        # named a mechanism, and this is how often it was wrong when it did.
        "decisive_rate": decisive / total,
        "decisive_error_rate": (confidently_wrong / decisive) if decisive else None,
        "note": (
            "Coverage is free if every set contains everything, so read it "
            "beside mean_set_size. confidently_wrong_rate is a share of all "
            "experiments; decisive_error_rate is the share of the *named* ones, "
            "which is what a reader acting on a verdict is exposed to. Split "
            "conformal guarantees the first kind of coverage and not the second."
        ),
    }
