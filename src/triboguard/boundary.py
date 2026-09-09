"""Where mechanism inference works, where it abstains, and where it misleads.

A method that reports its own uncertainty is only useful if that report can be
trusted across the designs people actually run. This sweeps designs and effect
sizes and sorts each combination into one of four regimes.

The regime that matters is **overconfident**: the system names a single
mechanism and is wrong, often enough to mislead. Every other failure is cheap.
An abstention costs a researcher some money and no credibility; a confident
wrong answer costs the reverse, and it is the failure that a tool claiming to
prevent unsupported conclusions must not have. So the boundary that gets drawn
here is the one between "safe to act on" and "quietly wrong", not the one
between "accurate" and "inaccurate".

Every cell calibrates on its own design. That is not a detail: pooling
calibration across designs was measured at 53% coverage against a 90% target,
and a map built on pooled calibration would chart the pooling bug rather than
the method.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from triboguard import abstention, inference, kinetics
from triboguard.abstention import CalibrationRecord
from triboguard.kinetics import BirthDeath

#: Tolerance on both error rates below, as a share. A finite evaluation sample
#: fluctuates -- forty experiments resolve errors only to one part in forty -- so
#: some allowance is unavoidable. It is slack, not permission to undershoot.
COVERAGE_SLACK = 0.10

#: An earlier version flagged any cell with more than 5% confident errors. That
#: was measuring the cutoff rather than the method: at 90% target coverage, 10%
#: of answers are *expected* to be wrong, so the rule demanded better than the
#: guarantee promises and painted ten of twenty-five cells red including the
#: easiest ones. What replaces it is the rate a researcher is actually exposed
#: to -- among experiments where the system named one mechanism, how often it
#: was wrong -- judged against the same nominal level. Split conformal
#: guarantees marginal coverage and not conditional coverage, so cells where
#: this fails while overall coverage holds are a real limitation of the method
#: and precisely what the map is for.
DECISIVE_ERROR_TOLERANCE = COVERAGE_SLACK

#: Above this mean set size the answer is honest and useless -- most of the
#: mechanism space survives, so nothing has been ruled out.
UNINFORMATIVE_ABOVE = 3.0

RELIABLE = "reliable"
ABSTAINS = "abstains"
UNINFORMATIVE = "uninformative"
OVERCONFIDENT = "overconfident"
#: An effect too small to move any rate past the classifier's own threshold
#: leaves "no effect" as the only truth in the cell. Nothing there tests whether
#: mechanisms can be told apart, and scoring it as though it did was how an
#: earlier sweep reported "overconfident" for columns that contained a single
#: class and one or two committed answers out of eighty.
NO_CONTRAST = "no_contrast"
REGIMES = (RELIABLE, ABSTAINS, UNINFORMATIVE, OVERCONFIDENT, NO_CONTRAST)


class BoundaryError(ValueError):
    """Raised when a sweep is specified in a way that cannot be run."""


@dataclass(frozen=True)
class Scenario:
    """One point in design space, and the size of the effect being looked for."""

    wells: int
    times: tuple[float, ...]
    effect: float
    seeded: int = 2000
    control: BirthDeath = BirthDeath(birth=0.055, death=0.005)

    def stratum(self) -> str:
        return f"wells={self.wells}|times={len(self.times)}"


def _treatment(scenario: Scenario, rng: np.random.Generator) -> tuple[BirthDeath, str]:
    """A treatment drawn uniformly from the four mechanisms at this effect size.

    Drawing uniformly rather than in proportion to how often each occurs in
    nature is deliberate: the map is about whether the method *can* separate
    them, and a rare class would otherwise be invisible in the averages.
    """
    slowed = bool(rng.integers(0, 2))
    killed = bool(rng.integers(0, 2))
    control = scenario.control
    # The two perturbations are scaled to be comparable in their effect on the
    # net rate, so neither mechanism is mechanically easier to detect than the
    # other at the same nominal effect size.
    shift = scenario.effect * control.birth
    birth_scale = (control.birth - shift) / control.birth if slowed else 1.0
    death_increase = shift if killed else 0.0
    treated = kinetics.apply_treatment(
        control, birth_scale=max(birth_scale, 0.0), death_increase=death_increase
    )
    return treated, kinetics.classify(control, treated)


def one_experiment(scenario: Scenario, seed: int, *, resamples: int = 100) -> CalibrationRecord:
    """Simulate a control and a treated condition, and infer their mechanism."""
    rng = np.random.default_rng(seed)
    treated, truth = _treatment(scenario, rng)
    control_wells = kinetics.simulate(
        scenario.control,
        scenario.seeded,
        scenario.times,
        replicates=scenario.wells,
        rng=np.random.default_rng(seed + 500_000),
    )
    treated_wells = kinetics.simulate(
        treated,
        scenario.seeded,
        scenario.times,
        replicates=scenario.wells,
        rng=np.random.default_rng(seed + 900_000),
    )
    distribution = inference.mechanism_distribution(
        control_wells,
        treated_wells,
        scenario.times,
        initial=scenario.seeded,
        resamples=resamples,
        seed=seed,
    )
    return CalibrationRecord(
        frequencies=distribution["frequencies"],
        truth=truth,
        group=f"exp{seed}",
        stratum=scenario.stratum(),
    )


def regime(report: dict[str, Any], confidence: float, *, distinct_truths: int = 2) -> str:
    """Sort one cell's metrics into a regime.

    Order matters. A cell with only one true mechanism is set aside first: it
    cannot speak to separating mechanisms whatever it scores. After that,
    misleading beats every other failure, because a cell that misleads is
    misleading regardless of how good its other numbers look.
    """
    if distinct_truths <= 1:
        return NO_CONTRAST
    if report["coverage"] < confidence - COVERAGE_SLACK:
        # The marginal guarantee is broken outright.
        return OVERCONFIDENT
    decisive_error = report.get("decisive_error_rate")
    if (
        decisive_error is not None
        and decisive_error > (1.0 - confidence) + DECISIVE_ERROR_TOLERANCE
    ):
        # Marginal coverage holds, but the answers the system was willing to
        # commit to are unreliable -- the wide sets are carrying the guarantee
        # while the narrow ones mislead. This is the failure a reader meets.
        return OVERCONFIDENT
    if report["mean_set_size"] > UNINFORMATIVE_ABOVE:
        return UNINFORMATIVE
    if report["abstention_rate"] > 0.5:
        return ABSTAINS
    return RELIABLE


def evaluate_cell(
    scenario: Scenario,
    *,
    calibration_experiments: int = 40,
    test_experiments: int = 40,
    confidence: float = 0.9,
    resamples: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """Calibrate and evaluate one point in design space, on its own design."""
    if calibration_experiments < 1 or test_experiments < 1:
        raise BoundaryError("both calibration and test need at least one experiment.")
    base = seed * 1_000_000
    calibration_records = [
        one_experiment(scenario, base + index, resamples=resamples)
        for index in range(calibration_experiments)
    ]
    calibration = abstention.calibrate(calibration_records, confidence=confidence)
    held_out = [
        one_experiment(scenario, base + 500_000 + index, resamples=resamples)
        for index in range(test_experiments)
    ]
    report = abstention.evaluate(held_out, calibration)
    truths = [record.truth for record in held_out]
    distinct = len(set(truths))
    return {
        "wells": scenario.wells,
        "timepoints": len(scenario.times),
        "effect": scenario.effect,
        "threshold": calibration.thresholds.get(scenario.stratum()),
        "calibration_degenerate": calibration.degenerate,
        **{key: value for key, value in report.items() if key != "note"},
        "regime": regime(report, confidence, distinct_truths=distinct),
        # A cell whose draws happened to contain one mechanism tells you nothing
        # about separating mechanisms, so the spread is recorded beside the score.
        "distinct_truths": distinct,
    }


def sweep(
    *,
    wells: Sequence[int],
    effects: Sequence[float],
    times: Sequence[float] = (0.0, 12.0, 24.0, 36.0, 48.0),
    calibration_experiments: int = 40,
    test_experiments: int = 40,
    confidence: float = 0.9,
    resamples: int = 100,
    seed: int = 0,
    progress: bool = False,
) -> dict[str, Any]:
    """The map: every combination of well count and effect size."""
    if not wells or not effects:
        raise BoundaryError("a sweep needs at least one well count and one effect size.")
    if any(count < 1 for count in wells):
        raise BoundaryError("well counts must be positive.")
    if any(effect < 0 for effect in effects):
        raise BoundaryError("effect sizes must be non-negative.")

    cells: list[dict[str, Any]] = []
    for index, well_count in enumerate(wells):
        for position, effect in enumerate(effects):
            scenario = Scenario(wells=well_count, times=tuple(times), effect=effect)
            cell = evaluate_cell(
                scenario,
                calibration_experiments=calibration_experiments,
                test_experiments=test_experiments,
                confidence=confidence,
                resamples=resamples,
                seed=seed + index * 97 + position,
            )
            cells.append(cell)
            if progress:
                exposure = cell["decisive_error_rate"]
                shown = "n/a" if exposure is None else f"{exposure:.2f}"
                print(
                    f"  wells={well_count:3d} effect={effect:.2f} -> {cell['regime']:14s} "
                    f"coverage {cell['coverage']:.2f} decisive {cell['decisive_rate']:.2f} "
                    f"error-when-decisive {shown}",
                    flush=True,
                )
    return {
        "cells": cells,
        "wells": list(wells),
        "effects": list(effects),
        "timepoints": list(times),
        "confidence": confidence,
        "thresholds": {
            "coverage_slack": COVERAGE_SLACK,
            "decisive_error_tolerance": DECISIVE_ERROR_TOLERANCE,
            "uninformative_above": UNINFORMATIVE_ABOVE,
        },
        "summary": summarise(cells),
    }


def null_behaviour(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """What the system does when the truth is "no effect" everywhere.

    These cells are set aside by :func:`regime` because they cannot test whether
    mechanisms are separable. They do test something else, and it was nearly
    lost behind that label: whether the system invents a mechanism when there is
    none. The rate rises with the well count -- zero at three wells, above ten
    percent at forty-eight -- which reads alarming and is mostly not.

    The effect at the smallest sweep step is a real change of about ten percent
    in a rate, sitting below the fifteen percent the classifier needs before it
    calls anything a mechanism. Ground truth therefore says "no effect" while a
    well-powered experiment is precise enough to resolve the change and gets
    scored wrong for succeeding. That is a hard threshold on a continuous
    quantity behaving as hard thresholds do, and it is a limitation of the
    scoring rather than of the inference.

    It is still worth reporting, because every laboratory that reports
    "cytotoxic" or "no effect" is applying some threshold, and the boundary
    behaves the same way for them.
    """
    return [
        {
            "wells": cell["wells"],
            "effect": cell["effect"],
            "coverage": cell["coverage"],
            "commits_on": cell["decisive_rate"],
            "wrong_when_it_commits": cell["decisive_error_rate"],
            "spurious_mechanism_rate": cell["decisive_rate"] * (cell["decisive_error_rate"] or 0.0),
        }
        for cell in cells
        if cell["regime"] == NO_CONTRAST
    ]


def summarise(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts per regime, and the safe boundary in wells."""
    counts = {name: 0 for name in REGIMES}
    for cell in cells:
        counts[cell["regime"]] += 1
    overconfident = [cell for cell in cells if cell["regime"] == OVERCONFIDENT]
    reliable = [cell for cell in cells if cell["regime"] == RELIABLE]
    return {
        "cells": len(cells),
        "regimes": counts,
        # The largest design that still misleads, and the smallest that does not.
        # Quoting only the second would suggest a clean cutoff that a sweep of
        # noisy cells does not actually have.
        "worst_overconfident_wells": (
            max(cell["wells"] for cell in overconfident) if overconfident else None
        ),
        "smallest_reliable_wells": (min(cell["wells"] for cell in reliable) if reliable else None),
        "max_confidently_wrong_rate": (
            max(cell["confidently_wrong_rate"] for cell in cells) if cells else None
        ),
        # Reported beside the regimes because the no-contrast label would
        # otherwise hide it entirely.
        "null_behaviour": null_behaviour(cells),
        "max_decisive_error_rate": (
            max(
                (cell["decisive_error_rate"] for cell in cells if cell["decisive_error_rate"]),
                default=None,
            )
        ),
    }
