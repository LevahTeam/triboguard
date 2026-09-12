"""The absent-fraction interval, and the abstention built on it.

The interval is the whole claim, so it is tested the way a claim about a bound
has to be: that it is *valid* -- the true value always lies inside it, over
random populations rather than a few chosen ones -- and that it is *tight* --
both ends are actually reached by some population, so it is not merely a wide
range that happens to contain everything.
"""

from __future__ import annotations

import numpy as np
import pytest

from triboguard import blindness as bl


def _counted_ratio(absent: float, damaged: float, blind: float) -> float:
    """The forward model, written out independently of the module under test."""
    return 1.0 - absent - damaged * blind


class TestBlindness:
    def test_equal_detection_is_no_blindness(self) -> None:
        assert bl.blindness(0.9, 0.9) == pytest.approx(0.0)

    def test_finding_no_damaged_cells_is_total_blindness(self) -> None:
        assert bl.blindness(0.9, 0.0) == pytest.approx(1.0)

    def test_it_is_relative_to_the_healthy_rate(self) -> None:
        """Half the healthy rate is half blind, whatever the healthy rate is."""
        assert bl.blindness(0.8, 0.4) == pytest.approx(0.5)
        assert bl.blindness(0.6, 0.3) == pytest.approx(0.5)

    def test_noise_above_the_healthy_rate_is_not_negative_blindness(self) -> None:
        assert bl.blindness(0.85, 0.87) == 0.0

    def test_a_segmenter_that_sees_nothing_is_refused(self) -> None:
        with pytest.raises(bl.BlindnessError, match="no healthy cells"):
            bl.blindness(0.0, 0.0)

    @pytest.mark.parametrize("bad", [-0.1, 1.2, float("nan")])
    def test_impossible_rates_are_refused(self, bad: float) -> None:
        with pytest.raises(bl.BlindnessError):
            bl.blindness(bad, 0.5)


class TestTheInterval:
    def test_a_segmenter_that_sees_everything_pins_it_exactly(self) -> None:
        """At zero blindness the count means what everyone assumes."""
        assert bl.absent_bounds(0.6, 0.0) == pytest.approx((0.4, 0.4))

    def test_total_blindness_cannot_say_whether_anything_is_gone(self) -> None:
        assert bl.absent_bounds(0.6, 1.0) == pytest.approx((0.0, 0.4))

    def test_the_bound_is_valid_for_every_population(self) -> None:
        """The true absent fraction is inside the interval, for 20,000 random wells."""
        rng = np.random.default_rng(0)
        for _ in range(20_000):
            absent = rng.uniform(0.0, 0.99)
            damaged = rng.uniform(0.0, 1.0 - absent)
            blind = rng.uniform(0.0, 1.0)
            low, high = bl.absent_bounds(_counted_ratio(absent, damaged, blind), blind)
            assert low - 1e-12 <= absent <= high + 1e-12

    def test_the_top_is_reached_when_every_missing_cell_is_gone(self) -> None:
        _, high = bl.absent_bounds(_counted_ratio(0.3, 0.0, 0.7), 0.7)
        assert high == pytest.approx(0.3)

    def test_the_bottom_is_reached_when_every_survivor_is_damaged(self) -> None:
        """Tightness: the lower bound is a real population, not slack."""
        blind, absent = 0.5, 0.6
        ratio = _counted_ratio(absent, 1.0 - absent, blind)
        low, _ = bl.absent_bounds(ratio, blind)
        assert low == pytest.approx(absent)

    def test_more_blindness_never_narrows_it(self) -> None:
        widths = []
        for blind in (0.0, 0.2, 0.5, 0.8, 1.0):
            low, high = bl.absent_bounds(0.5, blind)
            widths.append(high - low)
        assert widths == sorted(widths)

    def test_a_count_that_rose_leaves_nothing_absent(self) -> None:
        assert bl.absent_bounds(1.3, 0.4) == (0.0, 0.0)

    def test_the_box_contains_every_point_inside_it(self) -> None:
        low, high = bl.absent_interval(0.55, 0.65, 0.1, 0.3)
        for ratio in np.linspace(0.55, 0.65, 7):
            for blind in np.linspace(0.1, 0.3, 7):
                point_low, point_high = bl.absent_bounds(float(ratio), float(blind))
                assert low - 1e-12 <= point_low and point_high <= high + 1e-12

    def test_reversed_ranges_are_refused(self) -> None:
        with pytest.raises(bl.BlindnessError, match="ratio_low"):
            bl.absent_interval(0.7, 0.6, 0.1, 0.2)
        with pytest.raises(bl.BlindnessError, match="blind_low"):
            bl.absent_interval(0.6, 0.7, 0.3, 0.2)

    @pytest.mark.parametrize("bad", [-0.1, float("inf"), float("nan")])
    def test_impossible_ratios_are_refused(self, bad: float) -> None:
        with pytest.raises(bl.BlindnessError, match="ratio"):
            bl.absent_bounds(bad, 0.5)


class TestTheVerdict:
    def test_a_segmenter_that_sees_damaged_cells_resolves_the_count(self) -> None:
        result = bl.assess(0.6, 0.6, 0.03, 0.06)
        assert result["verdict"] == bl.RESOLVED
        assert result["width"] <= bl.RESOLUTION

    def test_the_same_count_through_a_blind_segmenter_abstains(self) -> None:
        """The whole point: an identical number, a different segmenter, no conclusion."""
        result = bl.assess(0.6, 0.6, 0.95, 1.0)
        assert result["verdict"] == bl.ABSTAIN
        assert result["absent_low"] == pytest.approx(0.0)

    def test_a_large_drop_establishes_absence_even_when_its_size_is_unclear(self) -> None:
        result = bl.assess(0.2, 0.2, 0.5, 0.5)
        assert result["verdict"] == bl.PRESENT_SIZE_UNRESOLVED
        assert result["absent_low"] >= bl.EFFECT_THRESHOLD

    def test_certain_but_possibly_trivial_absence_abstains(self) -> None:
        """Some cells are certainly gone, but perhaps too few to matter.

        The interval here is roughly [8%, 54%]: its bottom is above zero but below
        the effect threshold. Calling that "absence established" would claim an
        effect the count cannot distinguish from one too small to count. Mutation
        testing found this case unguarded.
        """
        result = bl.assess(0.46, 0.46, 0.5, 0.5)
        assert 0.0 < result["absent_low"] < bl.EFFECT_THRESHOLD
        assert result["verdict"] == bl.ABSTAIN

    def test_a_small_drop_is_no_meaningful_absence(self) -> None:
        assert bl.assess(0.95, 0.97, 0.0, 1.0)["verdict"] == bl.NO_MEANINGFUL_ABSENCE

    def test_uncertainty_in_the_count_widens_the_answer(self) -> None:
        point = bl.assess(0.6, 0.6, 0.03, 0.06)
        spread = bl.assess(0.5, 0.7, 0.03, 0.06)
        assert spread["width"] > point["width"]

    def test_the_verdict_is_stated_in_words(self) -> None:
        assert "consistent with no effect" in bl.assess(0.6, 0.6, 0.95, 1.0)["reads_as"]

    @pytest.mark.parametrize("resolution", [0.0, -0.1, 1.5])
    def test_an_impossible_resolution_is_refused(self, resolution: float) -> None:
        with pytest.raises(bl.BlindnessError, match="resolution"):
            bl.assess(0.6, 0.6, 0.1, 0.2, resolution=resolution)
