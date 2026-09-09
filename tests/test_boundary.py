"""The regime map, and the asymmetry it is built around.

Not all failures cost the same. An abstention wastes money; a confident wrong
answer wastes credibility and cannot be taken back. The classifier here has to
encode that asymmetry, so most of these tests are about whether a cell that
misleads is called out even when its other numbers look respectable.
"""

from __future__ import annotations

import numpy as np
import pytest

from triboguard import boundary, kinetics
from triboguard.boundary import (
    ABSTAINS,
    OVERCONFIDENT,
    RELIABLE,
    UNINFORMATIVE,
    BoundaryError,
    Scenario,
)

TIMES = (0.0, 12.0, 24.0, 36.0, 48.0)


def _report(**overrides: float) -> dict:
    base = {
        "coverage": 0.95,
        "mean_set_size": 1.2,
        "abstention_rate": 0.1,
        "confidently_wrong_rate": 0.0,
        "decisive_error_rate": 0.0,
        "decisive_rate": 0.9,
    }
    return {**base, **overrides}


class TestRegimeClassification:
    def test_a_clean_cell_is_reliable(self) -> None:
        assert boundary.regime(_report(), 0.9) == RELIABLE

    def test_unreliable_verdicts_are_caught_even_when_coverage_holds(self) -> None:
        """Marginal coverage can hold while the *named* answers mislead.

        Split conformal guarantees coverage over all experiments, not over the
        subset where the system committed to one mechanism. Wide sets can carry
        the average while narrow ones are wrong, and the narrow ones are what a
        reader acts on.
        """
        misleading = _report(coverage=0.95, mean_set_size=1.0, decisive_error_rate=0.4)
        assert boundary.regime(misleading, 0.9) == OVERCONFIDENT

    def test_an_error_rate_the_guarantee_permits_is_not_flagged(self) -> None:
        """At 90% coverage, 10% wrong is the method working, not failing.

        The first version of this classifier used a fixed 5% cutoff and painted
        ten of twenty-five cells red, including the easiest ones, because it
        demanded better than the guarantee promises.
        """
        expected = _report(coverage=0.92, decisive_error_rate=0.10)
        assert boundary.regime(expected, 0.9) == RELIABLE

    def test_broken_coverage_is_overconfident_even_without_confident_errors(self) -> None:
        """The misses landed in multi-label sets. The guarantee is still broken."""
        assert boundary.regime(_report(coverage=0.5), 0.9) == OVERCONFIDENT

    def test_honest_but_useless_is_its_own_regime(self) -> None:
        """Returning most of the mechanism space rules nothing out."""
        assert boundary.regime(_report(mean_set_size=3.8), 0.9) == UNINFORMATIVE

    def test_frequent_abstention_is_reported_as_abstention(self) -> None:
        assert boundary.regime(_report(abstention_rate=0.8), 0.9) == ABSTAINS

    def test_abstaining_beats_misleading_in_the_ordering(self) -> None:
        """A cell that both abstains a lot and misleads is called misleading."""
        both = _report(abstention_rate=0.9, coverage=0.5, decisive_error_rate=0.5)
        assert boundary.regime(both, 0.9) == OVERCONFIDENT

    def test_a_cell_that_never_commits_has_no_exposure_to_measure(self) -> None:
        """Abstaining everywhere is useless, not dangerous, and is named so."""
        silent = _report(abstention_rate=1.0, mean_set_size=2.0, decisive_error_rate=None)
        assert boundary.regime(silent, 0.9) == ABSTAINS

    def test_the_coverage_allowance_is_slack_not_permission(self) -> None:
        """Just inside the allowance passes; well outside does not."""
        assert boundary.regime(_report(coverage=0.85), 0.9) != OVERCONFIDENT
        assert boundary.regime(_report(coverage=0.70), 0.9) == OVERCONFIDENT


class TestScenario:
    def test_the_stratum_names_the_design(self) -> None:
        scenario = Scenario(wells=6, times=TIMES, effect=0.3)
        assert scenario.stratum() == "wells=6|times=5"

    def test_designs_differing_in_wells_are_different_strata(self) -> None:
        """Otherwise calibration pools across designs and the guarantee breaks."""
        assert Scenario(3, TIMES, 0.3).stratum() != Scenario(6, TIMES, 0.3).stratum()

    def test_designs_differing_in_timepoints_are_different_strata(self) -> None:
        assert Scenario(3, TIMES, 0.3).stratum() != Scenario(3, TIMES[:3], 0.3).stratum()


class TestTreatmentDraws:
    def test_all_four_mechanisms_appear_at_a_large_effect(self) -> None:
        scenario = Scenario(wells=6, times=TIMES, effect=1.0)
        seen = {boundary._treatment(scenario, np.random.default_rng(seed))[1] for seed in range(40)}
        assert seen == set(kinetics.MECHANISMS)

    def test_a_zero_effect_produces_no_mechanism(self) -> None:
        """There has to be a regime where the honest answer is 'nothing happened'."""
        scenario = Scenario(wells=6, times=TIMES, effect=0.0)
        seen = {boundary._treatment(scenario, np.random.default_rng(seed))[1] for seed in range(20)}
        assert seen == {kinetics.NO_EFFECT}

    def test_killing_and_stalling_are_scaled_to_be_equally_detectable(self) -> None:
        """Neither mechanism may be mechanically easier to find at one effect size."""
        scenario = Scenario(wells=6, times=TIMES, effect=0.5)
        control = scenario.control
        shift = 0.5 * control.birth
        killing = kinetics.apply_treatment(control, death_increase=shift)
        stalling = kinetics.apply_treatment(
            control, birth_scale=(control.birth - shift) / control.birth
        )
        assert killing.net == pytest.approx(stalling.net)


class TestCells:
    def test_a_cell_reports_its_design_and_its_regime(self) -> None:
        scenario = Scenario(wells=6, times=TIMES, effect=0.5)
        cell = boundary.evaluate_cell(
            scenario, calibration_experiments=12, test_experiments=12, resamples=40
        )
        assert cell["wells"] == 6
        assert cell["timepoints"] == len(TIMES)
        assert cell["regime"] in boundary.REGIMES
        assert cell["distinct_truths"] >= 1

    def test_a_cell_records_whether_its_calibration_was_degenerate(self) -> None:
        """Below the minimum group count the threshold is infinite, and it shows."""
        scenario = Scenario(wells=6, times=TIMES, effect=0.5)
        cell = boundary.evaluate_cell(
            scenario, calibration_experiments=3, test_experiments=6, resamples=20, confidence=0.9
        )
        assert cell["calibration_degenerate"] is True
        assert cell["mean_set_size"] == float(len(kinetics.MECHANISMS))

    def test_an_empty_cell_specification_is_refused(self) -> None:
        scenario = Scenario(wells=6, times=TIMES, effect=0.5)
        with pytest.raises(BoundaryError, match="at least one experiment"):
            boundary.evaluate_cell(scenario, calibration_experiments=0)


class TestSweep:
    def test_a_sweep_covers_every_combination(self) -> None:
        report = boundary.sweep(
            wells=(3, 6),
            effects=(0.3, 0.9),
            calibration_experiments=12,
            test_experiments=12,
            resamples=30,
        )
        assert len(report["cells"]) == 4
        assert report["summary"]["cells"] == 4

    def test_the_thresholds_used_are_recorded_with_the_map(self) -> None:
        """A regime boundary is meaningless without the cutoffs that drew it."""
        report = boundary.sweep(
            wells=(3,),
            effects=(0.3,),
            calibration_experiments=12,
            test_experiments=12,
            resamples=30,
        )
        assert report["thresholds"]["decisive_error_tolerance"] == boundary.DECISIVE_ERROR_TOLERANCE
        assert report["thresholds"]["uninformative_above"] == boundary.UNINFORMATIVE_ABOVE

    def test_the_summary_reports_the_worst_case_not_only_the_best(self) -> None:
        report = boundary.sweep(
            wells=(3, 6),
            effects=(0.3,),
            calibration_experiments=12,
            test_experiments=12,
            resamples=30,
        )
        assert "worst_overconfident_wells" in report["summary"]
        assert "max_confidently_wrong_rate" in report["summary"]

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"wells": (), "effects": (0.3,)}, "at least one well count"),
            ({"wells": (3,), "effects": ()}, "at least one well count"),
            ({"wells": (0,), "effects": (0.3,)}, "well counts must be positive"),
            ({"wells": (3,), "effects": (-0.1,)}, "effect sizes must be non-negative"),
        ],
    )
    def test_an_unrunnable_sweep_is_refused(self, kwargs: dict, message: str) -> None:
        with pytest.raises(BoundaryError, match=message):
            boundary.sweep(**kwargs)

    def test_the_map_is_reproducible_from_its_seed(self) -> None:
        settings = {
            "wells": (3,),
            "effects": (0.5,),
            "calibration_experiments": 12,
            "test_experiments": 12,
            "resamples": 30,
            "seed": 4,
        }
        first = boundary.sweep(**settings)
        second = boundary.sweep(**settings)
        assert first["cells"] == second["cells"]


class TestNoContrast:
    """A cell containing one true mechanism cannot speak about separating them.

    An earlier sweep reported "overconfident" for two whole columns that
    contained only ``no_effect``: the system committed on one or two experiments
    out of eighty and was wrong, which is a real observation about a degenerate
    column and not about the method's ability to tell killing from stalling.
    """

    def test_a_single_class_cell_is_set_aside(self) -> None:
        assert boundary.regime(_report(), 0.9, distinct_truths=1) == boundary.NO_CONTRAST

    def test_it_is_set_aside_even_when_the_numbers_look_terrible(self) -> None:
        awful = _report(coverage=0.1, decisive_error_rate=1.0)
        assert boundary.regime(awful, 0.9, distinct_truths=1) == boundary.NO_CONTRAST

    def test_a_cell_with_contrast_is_judged_normally(self) -> None:
        assert boundary.regime(_report(), 0.9, distinct_truths=4) == RELIABLE

    def test_an_effect_below_the_classifier_threshold_has_no_contrast(self) -> None:
        """The regime exists because this is reachable, not as a theoretical case."""
        scenario = Scenario(wells=6, times=TIMES, effect=0.10)
        seen = {boundary._treatment(scenario, np.random.default_rng(seed))[1] for seed in range(30)}
        assert seen == {kinetics.NO_EFFECT}

    def test_an_effect_above_it_does_have_contrast(self) -> None:
        scenario = Scenario(wells=6, times=TIMES, effect=0.40)
        seen = {boundary._treatment(scenario, np.random.default_rng(seed))[1] for seed in range(30)}
        assert len(seen) > 1
