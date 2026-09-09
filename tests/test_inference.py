"""Recovering rates from wells, and admitting when the wells cannot support it.

The tests that matter most are the negative ones. An estimator that returns a
confident mechanism from two wells is worse than no estimator, because it
launders a guess into a number, so several tests here exist to check that the
uncertainty *grows* when it should.
"""

from __future__ import annotations

import numpy as np
import pytest

from triboguard import inference, kinetics
from triboguard.inference import InferenceError
from triboguard.kinetics import BirthDeath

CONTROL = BirthDeath(birth=0.055, death=0.005)
HOURS = [0.0, 12.0, 24.0, 36.0, 48.0]
SEEDED = 2000


def _wells(params: BirthDeath, replicates: int, seed: int, times: list[float] | None = None):
    return kinetics.simulate(
        params, SEEDED, times or HOURS, replicates=replicates, rng=np.random.default_rng(seed)
    )


class TestRecoveringRates:
    def test_the_net_rate_is_recovered_from_many_wells(self) -> None:
        counts = _wells(CONTROL, 60, seed=0)
        got = inference.estimate_rates(counts, HOURS, initial=SEEDED, resamples=200)
        assert got.net == pytest.approx(CONTROL.net, abs=0.004)

    def test_the_turnover_is_recovered_from_many_wells(self) -> None:
        """The hard half: it exists only in the spread between wells."""
        counts = _wells(CONTROL, 400, seed=1)
        got = inference.estimate_rates(counts, HOURS, initial=SEEDED, resamples=200)
        assert got.turnover == pytest.approx(CONTROL.turnover, rel=0.35)

    def test_both_rates_follow_from_the_two_summaries(self) -> None:
        counts = _wells(CONTROL, 400, seed=2)
        got = inference.estimate_rates(counts, HOURS, initial=SEEDED, resamples=200)
        assert got.birth == pytest.approx((got.turnover + got.net) / 2)
        assert got.death == pytest.approx((got.turnover - got.net) / 2)
        assert got.rates.net == pytest.approx(got.net)

    def test_a_declining_population_is_recovered_as_declining(self) -> None:
        dying = BirthDeath(birth=0.01, death=0.06)
        got = inference.estimate_rates(
            _wells(dying, 60, seed=3), HOURS, initial=SEEDED, resamples=200
        )
        assert got.net < 0
        assert got.death > got.birth

    def test_the_starting_size_is_fitted_when_not_supplied(self) -> None:
        counts = _wells(CONTROL, 60, seed=4)
        got = inference.estimate_rates(counts, HOURS, resamples=100)
        assert got.initial == pytest.approx(SEEDED, rel=0.1)

    def test_supplying_the_seeding_density_does_not_move_the_answer(self) -> None:
        """It removes a free parameter; it must not change the biology."""
        counts = _wells(CONTROL, 60, seed=5)
        fitted = inference.estimate_rates(counts, HOURS, resamples=100)
        known = inference.estimate_rates(counts, HOURS, initial=SEEDED, resamples=100)
        assert known.net == pytest.approx(fitted.net, abs=0.005)


class TestKnowingWhatItCannotKnow:
    def test_one_well_cannot_estimate_turnover(self) -> None:
        counts = _wells(CONTROL, 1, seed=6)
        got = inference.estimate_rates(counts, HOURS, initial=SEEDED, resamples=200)
        assert got.turnover_estimable is False
        assert got.variance_floor_hit is True

    def test_fewer_wells_give_a_wider_turnover_interval(self) -> None:
        """The core promise: uncertainty must grow when evidence shrinks."""
        wide = inference.estimate_rates(
            _wells(CONTROL, 3, seed=7), HOURS, initial=SEEDED, resamples=400
        )
        narrow = inference.estimate_rates(
            _wells(CONTROL, 40, seed=7), HOURS, initial=SEEDED, resamples=400
        )
        wide_span = wide.turnover_interval[1] - wide.turnover_interval[0]
        narrow_span = narrow.turnover_interval[1] - narrow.turnover_interval[0]
        assert wide_span > narrow_span

    def test_more_timepoints_barely_help_the_mechanism(self) -> None:
        """Timepoints buy the mean. Only wells buy the mechanism.

        This is the result that makes the measurement-selection stage honest: if
        extra timepoints fixed the problem, the cheap advice would be "measure
        more often" and no design tool would be needed.
        """
        dense = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]
        few_wells_many_times = inference.estimate_rates(
            _wells(CONTROL, 3, seed=8, times=dense), dense, initial=SEEDED, resamples=400
        )
        many_wells_few_times = inference.estimate_rates(
            _wells(CONTROL, 24, seed=8, times=[0.0, 24.0, 48.0]),
            [0.0, 24.0, 48.0],
            initial=SEEDED,
            resamples=400,
        )
        by_times = (
            few_wells_many_times.turnover_interval[1] - few_wells_many_times.turnover_interval[0]
        )
        by_wells = (
            many_wells_few_times.turnover_interval[1] - many_wells_few_times.turnover_interval[0]
        )
        assert by_wells < by_times

    def test_identical_wells_hit_the_variance_floor(self) -> None:
        """Zero spread contradicts the model, and is reported rather than fitted."""
        one = kinetics.mean_count(CONTROL, SEEDED, HOURS)
        counts = np.vstack([one, one, one])
        got = inference.estimate_rates(counts, HOURS, initial=SEEDED, resamples=50)
        assert got.variance_floor_hit is True
        assert got.turnover == pytest.approx(abs(got.net))

    def test_an_extinct_population_is_refused_rather_than_fitted(self) -> None:
        counts = np.zeros((4, len(HOURS)))
        counts[:, 0] = SEEDED
        with pytest.raises(InferenceError, match="no rate is estimable"):
            inference.estimate_rates(counts, HOURS, initial=SEEDED, resamples=10)


class TestValidation:
    @pytest.mark.parametrize(
        "counts,times,message",
        [
            (np.ones((3,)), [0.0, 1.0], "replicates, times"),
            (np.ones((3, 2)), [0.0], "one per column"),
            (np.ones((3, 1)), [0.0], "at least two timepoints"),
            (np.ones((3, 2)), [1.0, 0.0], "strictly increasing"),
            (-np.ones((3, 2)), [0.0, 1.0], "non-negative"),
        ],
    )
    def test_an_experiment_that_cannot_be_described_is_refused(
        self, counts: np.ndarray, times: list[float], message: str
    ) -> None:
        with pytest.raises(InferenceError, match=message):
            inference.estimate_rates(counts, times, resamples=0)

    def test_a_single_timepoint_cannot_give_a_rate(self) -> None:
        with pytest.raises(InferenceError, match="at least two timepoints"):
            inference.estimate_rates(np.ones((3, 1)), [0.0], resamples=0)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
    def test_an_impossible_confidence_level_is_refused(self, bad: float) -> None:
        with pytest.raises(InferenceError, match="confidence"):
            inference.estimate_rates(
                _wells(CONTROL, 3, seed=9), HOURS, confidence=bad, resamples=10
            )

    def test_a_non_positive_seeding_density_is_refused(self) -> None:
        with pytest.raises(InferenceError, match="initial must be positive"):
            inference.estimate_rates(_wells(CONTROL, 3, seed=9), HOURS, initial=0, resamples=10)


class TestMechanismDistribution:
    def test_clear_killing_is_mostly_called_killing(self) -> None:
        treated = kinetics.apply_treatment(CONTROL, death_increase=0.05)
        got = inference.mechanism_distribution(
            _wells(CONTROL, 250, seed=10),
            _wells(treated, 250, seed=11),
            HOURS,
            initial=SEEDED,
            resamples=300,
        )
        assert got["frequencies"][kinetics.CYTOTOXIC] > 0.5

    def test_a_distribution_is_returned_rather_than_a_label(self) -> None:
        """Collapsing to an argmax here would discard the evidence of doubt."""
        treated = kinetics.apply_treatment(CONTROL, death_increase=0.02)
        got = inference.mechanism_distribution(
            _wells(CONTROL, 3, seed=12),
            _wells(treated, 3, seed=13),
            HOURS,
            initial=SEEDED,
            resamples=300,
        )
        assert set(got["frequencies"]) == set(kinetics.MECHANISMS)
        assert got["frequencies"][max(got["frequencies"], key=got["frequencies"].get)] < 1.0

    def test_the_frequencies_are_a_distribution(self) -> None:
        got = inference.mechanism_distribution(
            _wells(CONTROL, 6, seed=14),
            _wells(kinetics.apply_treatment(CONTROL, birth_scale=0.4), 6, seed=15),
            HOURS,
            initial=SEEDED,
            resamples=200,
        )
        assert sum(got["frequencies"].values()) == pytest.approx(1.0)

    def test_three_wells_are_less_certain_than_forty(self) -> None:
        """The claim the abstention layer will act on, checked at its source."""
        treated = kinetics.apply_treatment(CONTROL, death_increase=0.02)

        def spread(replicates: int, seed: int) -> float:
            got = inference.mechanism_distribution(
                _wells(CONTROL, replicates, seed=seed),
                _wells(treated, replicates, seed=seed + 100),
                HOURS,
                initial=SEEDED,
                resamples=300,
            )
            return max(got["frequencies"].values())

        assert spread(3, 16) < spread(250, 16)

    def test_two_wells_report_turnover_as_not_estimable(self) -> None:
        got = inference.mechanism_distribution(
            _wells(CONTROL, 1, seed=17),
            _wells(CONTROL, 4, seed=18),
            HOURS,
            initial=SEEDED,
            resamples=50,
        )
        assert got["turnover_estimable"] is False


class TestTheBootstrapCollapse:
    """Regression tests for a bug that made the tool confident where it was blindest.

    Turnover was originally interval-estimated by resampling wells. A resample of
    three wells draws duplicates almost always -- measured at 100% of draws
    collapsing onto the variance floor -- so the bootstrap distribution became a
    point mass and the interval came out *narrower* for three wells than for
    twenty-four. For a tool whose entire purpose is knowing when evidence is
    thin, reporting maximum confidence at minimum evidence is the worst
    available failure, and it passed every test that only checked the point
    estimate.
    """

    def test_three_wells_are_far_less_certain_than_twenty_four(self) -> None:
        def width(replicates: int, seed: int) -> float:
            got = inference.estimate_rates(
                _wells(CONTROL, replicates, seed=seed), HOURS, initial=SEEDED, resamples=100
            )
            return got.turnover_interval[1] - got.turnover_interval[0]

        assert width(3, 20) > 10 * width(24, 20)

    def test_the_interval_narrows_monotonically_with_wells(self) -> None:
        """Relative width, because absolute width also scales with the estimate.

        The chi-squared factor is what the well count controls, and it is
        strictly monotonic. Multiplying it by a point estimate that wanders from
        seed to seed can reorder the absolute widths without anything being
        wrong, so the invariant is asserted where it actually holds.
        """
        relative = [
            (bounds[1] - bounds[0]) / turnover
            for turnover, bounds in (
                (1.0, inference.turnover_bounds(1.0, replicates))
                for replicates in (3, 6, 12, 24, 48)
            )
        ]
        assert relative == sorted(relative, reverse=True)

    def test_the_measured_interval_shrinks_from_three_wells_to_forty_eight(self) -> None:
        """The same property on real fits, stated loosely enough to be true."""
        widths = [
            (lambda e: e.turnover_interval[1] - e.turnover_interval[0])(
                inference.estimate_rates(
                    _wells(CONTROL, replicates, seed=21), HOURS, initial=SEEDED, resamples=0
                )
            )
            for replicates in (3, 48)
        ]
        assert widths[0] > 10 * widths[1]

    def test_a_single_well_is_reported_as_unbounded_above(self) -> None:
        """Not a wide interval -- no interval. One well constrains nothing."""
        got = inference.estimate_rates(
            _wells(CONTROL, 1, seed=22), HOURS, initial=SEEDED, resamples=100
        )
        assert got.turnover_interval[1] == float("inf")
        assert got.as_dict()["turnover_interval"][1] is None

    def test_the_interval_brackets_the_truth_most_of_the_time(self) -> None:
        """Width is worthless if it is not aimed at the right place."""
        covered = 0
        trials = 40
        for seed in range(trials):
            got = inference.estimate_rates(
                _wells(CONTROL, 12, seed=300 + seed), HOURS, initial=SEEDED, resamples=0
            )
            low, high = got.turnover_interval
            covered += low <= CONTROL.turnover <= high
        assert covered >= 0.7 * trials, f"only {covered}/{trials} intervals covered the truth"
