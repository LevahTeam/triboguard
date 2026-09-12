"""Selectivity manufactured by the control plate rather than by the compound.

The claim these tests defend is easy to state and easy to get backwards: a
compound that kills nothing can still produce a large "% cytotoxicity" in a
dividing population and none at all in a static one. So the tests check the
direction of every quantity, not just its size, and they check that the two
things the module keeps separate -- what the plate reader reports and what
actually happened to the cells -- really do come apart.
"""

from __future__ import annotations

import math

import pytest

from triboguard import selectivity as sel
from triboguard.kinetics import BirthDeath

#: A dividing line and a population that has stopped, the pairing the source
#: study used. Rates are stated here rather than imported so a test failure
#: points at a number a reader can check.
DIVIDING = BirthDeath(birth=0.065, death=0.007)  # doubling every ~12 h
STATIC = BirthDeath(birth=0.0, death=0.005)  # slow attrition, no division


class TestTheIllusion:
    """The result the module exists for."""

    def test_stalling_alone_produces_cytotoxicity_in_a_dividing_population(self) -> None:
        treated = sel.apply_treatment(DIVIDING, birth_scale=0.66)
        assert sel.apparent_cytotoxicity(DIVIDING, treated, 24.0) > 0.35

    def test_the_same_treatment_produces_none_in_a_static_one(self) -> None:
        """No divisions to stop, so a cytostatic compound is invisible."""
        treated = sel.apply_treatment(STATIC, birth_scale=0.66)
        assert sel.apparent_cytotoxicity(STATIC, treated, 24.0) == pytest.approx(0.0)

    def test_a_non_selective_compound_looks_perfectly_selective(self) -> None:
        """Both arms get the identical treatment; the report separates them anyway."""
        result = sel.illusion(
            sel.Population("cancer", DIVIDING),
            sel.Population("normal", STATIC),
            hours=24.0,
            birth_scale=0.66,
        )
        assert result["arms"]["cancer"]["apparent_cytotoxicity"] > 0.35
        assert result["arms"]["normal"]["apparent_cytotoxicity"] == pytest.approx(0.0)
        assert result["apparent_ratio"] is None, "a zero denominator is not a large number"
        assert result["truly_selective"] is False

    def test_and_kills_nothing_in_either_arm(self) -> None:
        """The apparent number is large; the real one is zero. That is the point."""
        result = sel.illusion(
            sel.Population("cancer", DIVIDING),
            sel.Population("normal", STATIC),
            hours=24.0,
            birth_scale=0.66,
        )
        for arm in result["arms"].values():
            assert arm["cells_actually_killed"] == pytest.approx(0.0)

    def test_the_illusion_needs_a_same_time_control(self) -> None:
        """The crux, asserted rather than assumed.

        "% cytotoxicity" is measured against an untreated control read on the
        same day, which has been dividing all along. Measured against the
        *starting* population instead, a cytostatic compound on a growing line
        shows no loss at all -- the treated well still has more cells than it
        began with. The illusion is a property of the normalisation, so a
        version of this module that quietly changed the baseline would still
        pass every other test here.
        """
        treated = sel.apply_treatment(DIVIDING, birth_scale=0.66)
        against_control = sel.apparent_cytotoxicity(DIVIDING, treated, 24.0)
        # Against the starting count: the treated population has still grown.
        grown = math.exp(treated.net * 24.0)
        assert against_control > 0.35
        assert grown > 1.0, "the treated cells did not decline in absolute number"

    def test_killing_does_show_up_in_both_arms(self) -> None:
        """Guards against a module that reports zero killing unconditionally."""
        result = sel.illusion(
            sel.Population("cancer", DIVIDING),
            sel.Population("normal", STATIC),
            hours=24.0,
            death_increase=0.02,
        )
        for arm in result["arms"].values():
            assert arm["cells_actually_killed"] > 0.3

    def test_a_real_selective_compound_still_reads_as_selective(self) -> None:
        """The module must not conclude every selectivity claim is an artifact."""
        strong = sel.apply_treatment(DIVIDING, death_increase=0.03)
        spared = STATIC
        assert sel.apparent_cytotoxicity(DIVIDING, strong, 24.0) > 0.5
        assert sel.apparent_cytotoxicity(STATIC, spared, 24.0) == pytest.approx(0.0)


class TestSignalArithmetic:
    def test_no_treatment_is_no_effect(self) -> None:
        assert sel.apparent_cytotoxicity(DIVIDING, DIVIDING, 24.0) == pytest.approx(0.0)

    def test_at_time_zero_nothing_has_happened_yet(self) -> None:
        treated = sel.apply_treatment(DIVIDING, death_increase=0.05)
        assert sel.apparent_cytotoxicity(DIVIDING, treated, 0.0) == pytest.approx(0.0)

    def test_a_faster_population_makes_the_same_stalling_look_worse(self) -> None:
        """Apparent cytotoxicity tracks the division rate, not the compound."""
        rates = [0.02, 0.04, 0.065, 0.09]
        apparent = [
            sel.apparent_cytotoxicity(
                BirthDeath(birth=b, death=0.005),
                sel.apply_treatment(BirthDeath(birth=b, death=0.005), birth_scale=0.66),
                24.0,
            )
            for b in rates
        ]
        assert apparent == sorted(apparent)

    @pytest.mark.parametrize("bad", [-1.0, float("nan")])
    def test_impossible_durations_are_refused(self, bad: float) -> None:
        with pytest.raises(sel.SelectivityError):
            sel.apparent_cytotoxicity(DIVIDING, DIVIDING, bad)


class TestInvertingTheObservation:
    def test_required_change_round_trips(self) -> None:
        for cytotoxicity in (0.1, 0.4, 0.75):
            change = sel.required_net_change(cytotoxicity, 24.0)
            treated = BirthDeath(birth=DIVIDING.birth, death=DIVIDING.death - change)
            assert sel.apparent_cytotoxicity(DIVIDING, treated, 24.0) == pytest.approx(cytotoxicity)

    def test_the_threshold_birth_rate_is_exactly_sufficient(self) -> None:
        """At the boundary, stopping every division reproduces the observation."""
        needed = sel.birth_rate_for_pure_stalling(0.40, 24.0)
        borderline = BirthDeath(birth=needed, death=0.0)
        stopped = sel.apply_treatment(borderline, birth_scale=0.0)
        assert sel.apparent_cytotoxicity(borderline, stopped, 24.0) == pytest.approx(0.40)

    def test_a_slower_control_cannot_explain_it_by_stalling(self) -> None:
        needed = sel.birth_rate_for_pure_stalling(0.40, 24.0)
        too_slow = BirthDeath(birth=needed * 0.9, death=0.0)
        assert sel.stalling_fraction(too_slow, 0.40, 24.0) is None
        with pytest.raises(sel.SelectivityError, match="stalling alone"):
            sel.pure_stalling_partner(too_slow, 0.40, 24.0)

    def test_a_static_population_can_never_explain_it_by_stalling(self) -> None:
        assert sel.stalling_fraction(STATIC, 0.40, 24.0) is None

    def test_the_stalling_fraction_reproduces_the_observation(self) -> None:
        partner = sel.pure_stalling_partner(DIVIDING, 0.40, 24.0)
        assert sel.apparent_cytotoxicity(DIVIDING, partner, 24.0) == pytest.approx(0.40)
        assert partner.death == pytest.approx(DIVIDING.death), "stalling must not add death"

    def test_longer_exposure_needs_less_division_stopped(self) -> None:
        fractions = [sel.stalling_fraction(DIVIDING, 0.40, h) for h in (12.0, 24.0, 48.0)]
        assert all(f is not None for f in fractions)
        assert fractions == sorted(fractions, reverse=True)

    @pytest.mark.parametrize("bad", [1.0, 1.5, -0.1, float("nan")])
    def test_an_impossible_observation_is_refused(self, bad: float) -> None:
        with pytest.raises(sel.SelectivityError, match="cytotoxicity"):
            sel.required_net_change(bad, 24.0)

    @pytest.mark.parametrize("bad", [0.0, -3.0])
    def test_a_zero_or_negative_duration_is_refused(self, bad: float) -> None:
        with pytest.raises(sel.SelectivityError, match="hours"):
            sel.required_net_change(0.4, bad)


class TestTheExplainingFamily:
    def test_every_member_reproduces_the_same_observation(self) -> None:
        """The identifiability result, on the paper's own number."""
        for row in sel.explaining_family(DIVIDING, 0.40, 24.0, steps=9):
            assert row["apparent_cytotoxicity"] == pytest.approx(0.40)

    def test_the_ends_disagree_completely_about_what_happened(self) -> None:
        family = sel.explaining_family(DIVIDING, 0.40, 24.0, steps=9)
        killed = [row["cells_actually_killed"] for row in family]
        assert max(killed) > 0.35, "one end must be genuine killing"
        assert min(killed) == pytest.approx(0.0), "the other end must kill nothing"

    def test_less_division_means_less_killing_needed(self) -> None:
        family = sel.explaining_family(DIVIDING, 0.40, 24.0, steps=9)
        kept = [row["birth_scale"] for row in family]
        killed = [row["cells_actually_killed"] for row in family]
        assert kept == sorted(kept, reverse=True)
        assert killed == sorted(killed, reverse=True)

    def test_a_static_population_admits_only_killing(self) -> None:
        """Nothing to stall, so the family collapses to a single explanation.

        It must collapse to *one* row rather than several identical ones, which
        would advertise a range of explanations that does not exist.
        """
        family = sel.explaining_family(STATIC, 0.40, 24.0, steps=5)
        assert len(family) == 1
        assert family[0]["cells_actually_killed"] == pytest.approx(0.40)

    def test_pure_killing_kills_exactly_the_reported_share(self) -> None:
        partner = sel.pure_killing_partner(DIVIDING, 0.40, 24.0)
        assert sel._killed_fraction(DIVIDING, partner, 24.0) == pytest.approx(0.40)

    @pytest.mark.parametrize("bad", [0, 1, -2])
    def test_too_few_steps_are_refused(self, bad: int) -> None:
        with pytest.raises(sel.SelectivityError, match="steps"):
            sel.explaining_family(DIVIDING, 0.40, 24.0, steps=bad)


class TestTimeCourse:
    def test_a_constant_rate_course_is_recognised_as_one(self) -> None:
        treated = sel.apply_treatment(DIVIDING, birth_scale=0.66)
        points = [(h, sel.apparent_cytotoxicity(DIVIDING, treated, h)) for h in (3.0, 6.0, 24.0)]
        result = sel.time_course_consistency(points)
        assert result["constant_rate_consistent"]
        assert result["implied_rate_spread"] == pytest.approx(1.0)
        assert result["monotone_in_time"]

    def test_the_reported_course_is_not_one(self) -> None:
        """10% at 3 h, 40% at 6 h, still 40% at 24 h."""
        result = sel.time_course_consistency([(3.0, 0.10), (6.0, 0.40), (24.0, 0.40)])
        assert not result["constant_rate_consistent"]
        assert result["implied_rate_spread"] > 3.0

    def test_a_falling_course_is_flagged_as_non_monotone(self) -> None:
        result = sel.time_course_consistency([(3.0, 0.40), (24.0, 0.10)])
        assert not result["monotone_in_time"]
        assert not result["constant_rate_consistent"]

    def test_a_single_point_is_not_a_time_course(self) -> None:
        with pytest.raises(sel.SelectivityError, match="two points"):
            sel.time_course_consistency([(24.0, 0.40)])


class TestPopulation:
    def test_a_dividing_line_says_so(self) -> None:
        assert sel.Population("cancer", DIVIDING).divides

    def test_a_static_one_says_so(self) -> None:
        assert not sel.Population("normal", STATIC).divides

    def test_a_negative_rate_is_refused_upstream(self) -> None:
        from triboguard.kinetics import KineticsError

        with pytest.raises(KineticsError):
            BirthDeath(birth=-0.01, death=0.0)
