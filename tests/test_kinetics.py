"""The birth-death model, and the degeneracy the whole project is built around.

The most important test here is the one asserting that two different mechanisms
produce the same mean trajectory. If that ever fails, either the model is wrong
or the project has no reason to exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from triboguard import kinetics
from triboguard.kinetics import BirthDeath, KineticsError

CONTROL = BirthDeath(birth=0.055, death=0.005)
HOURS = [0.0, 6.0, 12.0, 24.0, 48.0]


class TestRates:
    def test_net_is_the_difference_and_turnover_is_the_sum(self) -> None:
        rates = BirthDeath(birth=0.06, death=0.02)
        assert rates.net == pytest.approx(0.04)
        assert rates.turnover == pytest.approx(0.08)

    def test_a_shrinking_population_has_no_doubling_time(self) -> None:
        assert BirthDeath(birth=0.01, death=0.05).doubling_hours is None

    def test_doubling_time_inverts_the_net_rate(self) -> None:
        rates = BirthDeath(birth=0.05, death=0.01)
        assert rates.doubling_hours == pytest.approx(np.log(2) / 0.04)

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
    def test_an_impossible_rate_is_refused(self, bad: float) -> None:
        with pytest.raises(KineticsError):
            BirthDeath(birth=bad, death=0.01)
        with pytest.raises(KineticsError):
            BirthDeath(birth=0.01, death=bad)


class TestTheDegeneracy:
    """Why a mean trajectory cannot name a mechanism."""

    def test_killing_and_stalling_produce_the_same_mean_curve(self) -> None:
        """The claim the project exists to address, asserted rather than assumed."""
        killing = kinetics.apply_treatment(CONTROL, death_increase=0.02)
        # Slow division by the same amount the other arm raised death by.
        stalling = kinetics.apply_treatment(
            CONTROL, birth_scale=(CONTROL.birth - 0.02) / CONTROL.birth
        )
        assert kinetics.classify(CONTROL, killing) == kinetics.CYTOTOXIC
        assert kinetics.classify(CONTROL, stalling) == kinetics.CYTOSTATIC
        np.testing.assert_allclose(
            kinetics.mean_count(killing, 1000, HOURS),
            kinetics.mean_count(stalling, 1000, HOURS),
            rtol=1e-12,
        )

    def test_but_their_variances_differ(self) -> None:
        """The channel that does carry the mechanism."""
        killing = kinetics.apply_treatment(CONTROL, death_increase=0.02)
        stalling = kinetics.apply_treatment(
            CONTROL, birth_scale=(CONTROL.birth - 0.02) / CONTROL.birth
        )
        killed = kinetics.variance_count(killing, 1000, HOURS)
        stalled = kinetics.variance_count(stalling, 1000, HOURS)
        # More turnover means more spread across wells at every later time.
        assert killing.turnover > stalling.turnover
        assert np.all(killed[1:] > stalled[1:])

    def test_a_matching_partner_exists_for_any_birth_rate(self) -> None:
        observed = BirthDeath(birth=0.05, death=0.01)
        partner = kinetics.indistinguishable_partner(observed, birth=0.09)
        assert partner.net == pytest.approx(observed.net)
        np.testing.assert_allclose(
            kinetics.mean_count(observed, 500, HOURS),
            kinetics.mean_count(partner, 500, HOURS),
            rtol=1e-12,
        )

    def test_no_partner_exists_when_it_would_need_a_negative_death_rate(self) -> None:
        with pytest.raises(KineticsError, match="No non-negative death rate"):
            kinetics.indistinguishable_partner(BirthDeath(birth=0.05, death=0.0), birth=0.01)


class TestMoments:
    def test_the_mean_is_exponential_in_the_net_rate(self) -> None:
        got = kinetics.mean_count(CONTROL, 100, [0.0, 10.0])
        assert got[0] == pytest.approx(100.0)
        assert got[1] == pytest.approx(100.0 * np.exp(CONTROL.net * 10.0))

    def test_variance_is_zero_at_time_zero(self) -> None:
        """Every well starts at the same seeding density by construction."""
        assert kinetics.variance_count(CONTROL, 100, [0.0])[0] == pytest.approx(0.0)

    def test_a_population_in_balance_still_accumulates_variance(self) -> None:
        """Births and deaths cancel in the mean and add in the spread."""
        balanced = BirthDeath(birth=0.03, death=0.03)
        got = kinetics.variance_count(balanced, 100, [0.0, 10.0])
        assert got[1] == pytest.approx(2 * 100 * 0.03 * 10.0)

    def test_the_balanced_case_is_the_limit_of_the_general_one(self) -> None:
        """The two branches must agree where they meet, or the seam is a bug."""
        nearly = kinetics.variance_count(BirthDeath(0.03, 0.03 - 1e-9), 100, [24.0])[0]
        exactly = kinetics.variance_count(BirthDeath(0.03, 0.03), 100, [24.0])[0]
        assert nearly == pytest.approx(exactly, rel=1e-5)

    @pytest.mark.parametrize("bad", [[-1.0], [0.0, -2.0]])
    def test_negative_times_are_refused(self, bad: list[float]) -> None:
        with pytest.raises(KineticsError):
            kinetics.mean_count(CONTROL, 100, bad)
        with pytest.raises(KineticsError):
            kinetics.variance_count(CONTROL, 100, bad)


class TestSimulation:
    def test_simulated_means_match_the_analytic_mean(self) -> None:
        rng = np.random.default_rng(0)
        counts = kinetics.simulate(CONTROL, 400, [0.0, 12.0, 24.0], replicates=400, rng=rng)
        expected = kinetics.mean_count(CONTROL, 400, [0.0, 12.0, 24.0])
        np.testing.assert_allclose(counts.mean(axis=0), expected, rtol=0.05)

    def test_simulated_variance_matches_the_analytic_variance(self) -> None:
        """The load-bearing check: the second moment is what the fit will use."""
        rng = np.random.default_rng(1)
        counts = kinetics.simulate(CONTROL, 400, [0.0, 12.0, 24.0], replicates=2000, rng=rng)
        expected = kinetics.variance_count(CONTROL, 400, [0.0, 12.0, 24.0])
        np.testing.assert_allclose(counts.var(axis=0, ddof=1)[1:], expected[1:], rtol=0.12)

    def test_trajectories_are_walked_forward_not_resampled(self) -> None:
        """A dying population must not rise again between timepoints.

        Sampling each timepoint independently from time zero gives correct
        marginals and impossible trajectories. This is the check that would
        catch that.
        """
        rng = np.random.default_rng(2)
        dying = BirthDeath(birth=0.0, death=0.08)
        counts = kinetics.simulate(dying, 300, [0.0, 6.0, 12.0, 24.0], replicates=60, rng=rng)
        assert np.all(np.diff(counts, axis=1) <= 0)

    def test_extinction_is_absorbing(self) -> None:
        rng = np.random.default_rng(3)
        counts = kinetics.simulate(
            BirthDeath(birth=0.0, death=1.5), 5, [0.0, 10.0, 20.0], replicates=50, rng=rng
        )
        dead = counts[:, 1] == 0
        assert dead.any(), "test is vacuous unless some wells died out"
        assert np.all(counts[dead, 2] == 0)

    def test_every_well_starts_at_the_seeding_density(self) -> None:
        counts = kinetics.simulate(
            CONTROL, 250, [0.0, 5.0], replicates=8, rng=np.random.default_rng(4)
        )
        assert np.all(counts[:, 0] == 250)

    def test_the_same_seed_reproduces_the_same_wells(self) -> None:
        first = kinetics.simulate(CONTROL, 200, HOURS, replicates=5, rng=np.random.default_rng(7))
        second = kinetics.simulate(CONTROL, 200, HOURS, replicates=5, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"times": [12.0, 6.0]},
            {"times": []},
            {"times": [-1.0, 2.0]},
            {"replicates": 0},
            {"initial": 0},
        ],
    )
    def test_a_schedule_that_cannot_describe_an_experiment_is_refused(self, kwargs: dict) -> None:
        arguments = {"initial": 100, "times": [0.0, 6.0], "replicates": 3, **kwargs}
        with pytest.raises(KineticsError):
            kinetics.simulate(CONTROL, rng=np.random.default_rng(0), **arguments)


class TestClassification:
    def test_raising_death_alone_is_cytotoxic(self) -> None:
        treated = kinetics.apply_treatment(CONTROL, death_increase=0.03)
        assert kinetics.classify(CONTROL, treated) == kinetics.CYTOTOXIC

    def test_slowing_division_alone_is_cytostatic(self) -> None:
        treated = kinetics.apply_treatment(CONTROL, birth_scale=0.4)
        assert kinetics.classify(CONTROL, treated) == kinetics.CYTOSTATIC

    def test_moving_both_is_mixed(self) -> None:
        treated = kinetics.apply_treatment(CONTROL, birth_scale=0.4, death_increase=0.03)
        assert kinetics.classify(CONTROL, treated) == kinetics.MIXED

    def test_a_negligible_change_is_not_a_mechanism(self) -> None:
        """A rate that moved 2% is noise with a label on it."""
        treated = kinetics.apply_treatment(CONTROL, birth_scale=0.98, death_increase=0.0005)
        assert kinetics.classify(CONTROL, treated) == kinetics.NO_EFFECT

    def test_death_is_scaled_by_turnover_not_by_the_control_death_rate(self) -> None:
        """A control that barely dies must not make every treatment look lethal."""
        pristine = BirthDeath(birth=0.05, death=0.0001)
        treated = kinetics.apply_treatment(pristine, death_increase=0.002)
        assert kinetics.classify(pristine, treated) == kinetics.NO_EFFECT

    def test_a_negative_threshold_is_refused(self) -> None:
        with pytest.raises(KineticsError):
            kinetics.classify(CONTROL, CONTROL, threshold=-0.1)


class TestSummarise:
    def test_one_well_yields_no_variance(self) -> None:
        """The unidentifiable case, reported as such rather than as zero spread."""
        counts = kinetics.simulate(CONTROL, 200, HOURS, replicates=1, rng=np.random.default_rng(0))
        summary = kinetics.summarise(counts)
        assert summary["variance"] is None
        assert summary["variance_estimable"] is False

    def test_several_wells_yield_a_variance(self) -> None:
        counts = kinetics.simulate(CONTROL, 200, HOURS, replicates=4, rng=np.random.default_rng(0))
        summary = kinetics.summarise(counts)
        assert summary["variance_estimable"] is True
        assert len(summary["variance"]) == len(HOURS)

    def test_a_one_dimensional_array_is_refused(self) -> None:
        with pytest.raises(KineticsError):
            kinetics.summarise(np.array([1.0, 2.0, 3.0]))
