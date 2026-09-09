"""Planning the next measurement, and refusing to plan the useless one.

The tests that carry weight are the ones asserting that extra timepoints rank
below extra wells, and that a destructive assay is charged for what it destroys.
Both are conclusions the earlier stages forced, and a planner that quietly
ignored either would still look reasonable in its output.
"""

from __future__ import annotations

import math

import pytest

from triboguard.design import (
    MAXIMUM_SEARCHED_WELLS,
    MINIMUM_USEFUL_WELLS,
    Assay,
    Design,
    DesignError,
    options,
    recommend,
    relative_turnover_width,
    wells_for_width,
)

MTS = Assay(per_well=0.80, per_measurement=0.35, destructive=True, name="MTS")
IMAGING = Assay(per_well=0.80, per_measurement=0.20, destructive=False, name="microscopy")
PAPER = Design(wells=3, times=(3.0, 6.0, 12.0, 24.0))


class TestPrecision:
    def test_one_well_leaves_the_mechanism_unbounded(self) -> None:
        assert relative_turnover_width(1) == float("inf")

    def test_precision_improves_monotonically_with_wells(self) -> None:
        widths = [relative_turnover_width(w) for w in (2, 3, 4, 8, 16, 32, 64)]
        assert widths == sorted(widths, reverse=True)

    def test_three_wells_barely_constrain_turnover(self) -> None:
        """The design in most published dose-response work."""
        assert relative_turnover_width(3) > 30

    def test_returns_diminish(self) -> None:
        """The first well is worth far more than the fiftieth."""
        early = relative_turnover_width(3) - relative_turnover_width(4)
        late = relative_turnover_width(50) - relative_turnover_width(51)
        assert early > 100 * late

    def test_a_looser_confidence_gives_a_narrower_interval(self) -> None:
        assert relative_turnover_width(12, 0.80) < relative_turnover_width(12, 0.95)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.3])
    def test_an_impossible_confidence_is_refused(self, bad: float) -> None:
        with pytest.raises(DesignError, match="confidence"):
            relative_turnover_width(10, bad)


class TestWellsForWidth:
    def test_it_inverts_the_width_function(self) -> None:
        needed = wells_for_width(2.0)
        assert needed is not None
        assert relative_turnover_width(needed) <= 2.0
        assert relative_turnover_width(needed - 1) > 2.0

    def test_a_tighter_target_needs_more_wells(self) -> None:
        loose = wells_for_width(2.0)
        tight = wells_for_width(0.5)
        assert loose is not None and tight is not None
        assert tight > loose

    def test_an_unreachable_target_returns_none_rather_than_a_fantasy(self) -> None:
        assert wells_for_width(1e-9) is None

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_a_non_positive_target_is_refused(self, bad: float) -> None:
        with pytest.raises(DesignError, match="target width"):
            wells_for_width(bad)


class TestDestructiveAssays:
    def test_a_destructive_assay_consumes_a_well_per_timepoint(self) -> None:
        """MTS lyses the cells it reads, so a time course cannot reuse wells."""
        assert PAPER.wells_consumed(MTS) == 3 * 4
        assert PAPER.wells_consumed(IMAGING) == 3

    def test_the_same_design_costs_more_under_a_destructive_assay(self) -> None:
        assert PAPER.cost(MTS) > PAPER.cost(IMAGING)

    def test_the_gap_grows_with_the_number_of_timepoints(self) -> None:
        """This is what makes an endpoint assay expensive for kinetics."""
        short = Design(wells=3, times=(0.0, 24.0))
        long = Design(wells=3, times=(0.0, 6.0, 12.0, 18.0, 24.0))
        assert long.cost(MTS) / short.cost(MTS) > long.cost(IMAGING) / short.cost(IMAGING)

    def test_a_negative_cost_is_refused(self) -> None:
        with pytest.raises(DesignError, match="non-negative"):
            Assay(per_well=-1.0, per_measurement=0.2)


class TestDesignValidation:
    @pytest.mark.parametrize(
        "wells,times,message",
        [
            (0, (0.0, 1.0), "wells must be at least 1"),
            (3, (0.0,), "at least two timepoints"),
            (3, (1.0, 1.0), "distinct"),
            (3, (-1.0, 2.0), "non-negative"),
        ],
    )
    def test_a_design_that_cannot_be_run_is_refused(
        self, wells: int, times: tuple[float, ...], message: str
    ) -> None:
        with pytest.raises(DesignError, match=message):
            Design(wells=wells, times=times)


class TestRanking:
    def test_extra_wells_outrank_extra_timepoints(self) -> None:
        """The finding the whole planner rests on, asserted rather than assumed."""
        ranked = options(PAPER, IMAGING, extra_times=(36.0, 48.0))
        first_timepoint = next(i for i, o in enumerate(ranked) if "timepoint" in o.label)
        first_well = next(i for i, o in enumerate(ranked) if "well" in o.label)
        assert first_well < first_timepoint

    def test_a_timepoint_buys_no_mechanism_at_all(self) -> None:
        ranked = options(PAPER, IMAGING, extra_times=(36.0,))
        timepoint = next(o for o in ranked if "timepoint" in o.label)
        assert timepoint.information == 0.0
        assert timepoint.relative_width == relative_turnover_width(PAPER.wells)

    def test_a_timepoint_option_says_why_it_is_useless_here(self) -> None:
        """It is offered so the ranking demonstrates the point, not hides it."""
        ranked = options(PAPER, IMAGING, extra_times=(36.0,))
        timepoint = next(o for o in ranked if "timepoint" in o.label)
        assert "not the mechanism" in timepoint.note

    def test_a_timepoint_already_measured_is_not_offered_again(self) -> None:
        ranked = options(PAPER, IMAGING, extra_times=(12.0,))
        assert not any("timepoint" in o.label for o in ranked)

    def test_more_wells_always_buy_more_information(self) -> None:
        ranked = options(PAPER, IMAGING, extra_wells=(1, 5, 20), extra_times=())
        by_size = sorted(ranked, key=lambda o: o.design.wells)
        assert [o.information for o in by_size] == sorted(o.information for o in by_size)

    def test_the_cheapest_step_wins_on_information_per_cost(self) -> None:
        """Diminishing returns mean the first well is the best buy."""
        ranked = options(PAPER, IMAGING, extra_wells=(1, 2, 5, 20), extra_times=())
        assert ranked[0].design.wells == PAPER.wells + 1

    def test_a_zero_cost_assay_still_ranks(self) -> None:
        """'What if money were no object' is a fair question, not a crash."""
        free = Assay(per_well=0.0, per_measurement=0.0, name="free")
        ranked = options(PAPER, free, extra_wells=(1, 20), extra_times=())
        assert all(math.isfinite(o.information_per_cost) for o in ranked)
        assert ranked[0].information > 0

    def test_a_non_positive_well_increment_is_refused(self) -> None:
        with pytest.raises(DesignError, match="extra wells"):
            options(PAPER, IMAGING, extra_wells=(0,))


class TestRecommendation:
    def test_it_reports_the_paper_design_as_barely_constraining(self) -> None:
        report = recommend(PAPER, MTS, target_width=1.0)
        assert report["current"]["relative_turnover_width"] > 30
        assert report["current"]["mechanism_estimable"] is True

    def test_it_prices_the_target_in_wells(self) -> None:
        report = recommend(PAPER, MTS, target_width=1.0)
        assert report["target"]["reachable"] is True
        assert report["target"]["wells_required"] > PAPER.wells
        assert "wells per condition" in report["statement"]

    def test_an_unreachable_target_is_reported_as_such(self) -> None:
        report = recommend(PAPER, MTS, target_width=1e-9)
        assert report["target"]["reachable"] is False
        assert report["target"]["wells_required"] is None
        assert "out of reach" in report["statement"]

    def test_a_destructive_assay_is_called_out_in_the_statement(self) -> None:
        assert "destroys the well" in recommend(PAPER, MTS)["statement"]
        assert "destroys the well" not in recommend(PAPER, IMAGING)["statement"]

    def test_a_single_well_design_is_told_it_cannot_estimate_anything(self) -> None:
        lonely = Design(wells=1, times=(0.0, 24.0))
        report = recommend(lonely, IMAGING)
        assert report["current"]["mechanism_estimable"] is False
        assert report["current"]["relative_turnover_width"] is None
        assert "cannot be estimated at all" in report["statement"]

    def test_it_says_how_many_calibration_experiments_the_guarantee_needs(self) -> None:
        """Design advice is incomplete without the cost of the guarantee itself."""
        report = recommend(PAPER, IMAGING, calibration_confidence=0.9)
        assert report["calibration_experiments_needed"] == 9

    def test_the_report_survives_a_round_trip_through_json(self) -> None:
        import json

        report = recommend(Design(wells=1, times=(0.0, 24.0)), MTS, target_width=1e-9)
        assert json.loads(json.dumps(report))["current"]["relative_turnover_width"] is None

    def test_the_ceiling_on_the_search_is_stated_not_hidden(self) -> None:
        assert MAXIMUM_SEARCHED_WELLS > 0
        assert MINIMUM_USEFUL_WELLS == 2


class TestRatioVersusWidth:
    """Two different quantities that an earlier version conflated.

    "Known to within a factor of X" names the ratio of the interval's endpoints.
    The relative width is the interval's span divided by the estimate. At three
    wells they are 146 and 39; the earlier code reported the width wherever the
    prose said factor, understating the gap nearly fourfold, and quoted 39 wells
    for "a factor of two" when the ratio does not reach 2.0 until 66.
    """

    def test_the_two_quantities_differ(self) -> None:
        from triboguard.design import turnover_ratio

        assert turnover_ratio(3) > 3 * relative_turnover_width(3)

    def test_the_ratio_at_three_wells_is_about_146(self) -> None:
        from triboguard.design import turnover_ratio

        assert turnover_ratio(3) == pytest.approx(145.7, rel=0.01)

    def test_a_factor_of_two_needs_far_more_wells_than_a_width_of_one(self) -> None:
        from triboguard.design import wells_for_ratio

        assert wells_for_ratio(2.0) == 66
        assert wells_for_width(1.0) == 39

    def test_the_ratio_falls_monotonically_towards_one(self) -> None:
        from triboguard.design import turnover_ratio

        ratios = [turnover_ratio(w) for w in (3, 6, 12, 48, 200)]
        assert ratios == sorted(ratios, reverse=True)
        assert ratios[-1] > 1.0

    def test_one_well_has_no_ratio(self) -> None:
        from triboguard.design import turnover_ratio

        assert turnover_ratio(1) == float("inf")

    def test_an_impossible_target_ratio_is_refused(self) -> None:
        from triboguard.design import wells_for_ratio

        with pytest.raises(DesignError, match="cannot span less than"):
            wells_for_ratio(1.0)
