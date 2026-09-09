"""The coverage guarantee, and the ways a fake one would show up.

A set-valued predictor is easy to fake: return every label and coverage is
perfect. So the tests here always check coverage *against* the set size that
bought it, and the end-to-end test runs the whole pipeline -- simulate, infer,
calibrate, predict -- on experiments the calibration never saw.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from triboguard import abstention, inference, kinetics
from triboguard.abstention import AbstentionError, CalibrationRecord
from triboguard.kinetics import BirthDeath

CONTROL = BirthDeath(birth=0.055, death=0.005)
HOURS = [0.0, 12.0, 24.0, 36.0, 48.0]
SEEDED = 2000


def _record(truth: str, confident: float, group: str) -> CalibrationRecord:
    """A record giving ``confident`` frequency to ``truth`` and sharing the rest."""
    others = [m for m in kinetics.MECHANISMS if m != truth]
    spread = (1.0 - confident) / len(others)
    frequencies = {truth: confident, **{name: spread for name in others}}
    return CalibrationRecord(frequencies=frequencies, truth=truth, group=group)


class TestMinimumGroups:
    @pytest.mark.parametrize(
        "confidence,expected", [(0.5, 1), (0.8, 4), (0.9, 9), (0.95, 19), (0.99, 99)]
    )
    def test_the_threshold_needs_this_many_experiments(
        self, confidence: float, expected: int
    ) -> None:
        assert abstention.minimum_groups(confidence) == expected

    def test_at_the_minimum_the_calibration_is_not_degenerate(self) -> None:
        """The advertised minimum must actually work, or it is bad advice."""
        needed = abstention.minimum_groups(0.9)
        records = [_record(kinetics.CYTOTOXIC, 0.8, f"drug{i}") for i in range(needed)]
        assert abstention.calibrate(records, confidence=0.9).degenerate is False

    def test_one_below_the_minimum_is_degenerate(self) -> None:
        needed = abstention.minimum_groups(0.9)
        records = [_record(kinetics.CYTOTOXIC, 0.8, f"drug{i}") for i in range(needed - 1)]
        assert abstention.calibrate(records, confidence=0.9).degenerate is True


class TestCalibration:
    def test_too_few_experiments_return_everything(self) -> None:
        """Not a narrower set with a broken promise -- the whole set, and a flag."""
        records = [_record(kinetics.CYTOTOXIC, 0.9, "only")]
        calibration = abstention.calibrate(records, confidence=0.95)
        assert calibration.degenerate is True
        verdict = abstention.predict_set({kinetics.CYTOTOXIC: 1.0}, calibration)
        assert verdict["mechanisms"] == list(kinetics.MECHANISMS)
        assert verdict["uninformative"] is True

    def test_repeats_of_one_experiment_do_not_buy_a_tighter_threshold(self) -> None:
        """Twenty doses of one drug are one experiment, not twenty."""
        many_doses = [_record(kinetics.CYTOTOXIC, 0.9, "drugA") for _ in range(20)]
        calibration = abstention.calibrate(many_doses, confidence=0.9)
        assert calibration.groups == 1
        assert calibration.degenerate is True

    def test_the_worst_record_in_a_group_sets_its_score(self) -> None:
        """Averaging an easy dose against a hard one would hide the hard one."""
        mixed = [
            _record(kinetics.CYTOTOXIC, 0.95, "drugA"),
            _record(kinetics.CYTOTOXIC, 0.20, "drugA"),
            *[_record(kinetics.CYTOTOXIC, 0.95, f"drug{i}") for i in range(1, 12)],
        ]
        calibration = abstention.calibrate(mixed, confidence=0.9)
        assert calibration.groups == 12
        # The hard dose is the group's score, so the threshold must admit it.
        assert calibration.threshold >= 0.8 - 1e-9

    def test_an_empty_calibration_is_refused(self) -> None:
        with pytest.raises(AbstentionError, match="at least one record"):
            abstention.calibrate([])

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.4])
    def test_an_impossible_confidence_is_refused(self, bad: float) -> None:
        with pytest.raises(AbstentionError, match="confidence"):
            abstention.calibrate([_record(kinetics.MIXED, 0.5, "a")], confidence=bad)

    def test_an_unknown_mechanism_label_is_refused(self) -> None:
        bogus = CalibrationRecord(frequencies={"apoptosis": 1.0}, truth="apoptosis", group="a")
        with pytest.raises(AbstentionError, match="not one of"):
            abstention.calibrate([bogus])


class TestPredictionSets:
    @pytest.fixture
    def calibration(self) -> abstention.Calibration:
        """A threshold of 0.55: it admits a two-way split but not a four-way one.

        Calibrating on experiments the estimator found easy gives a threshold so
        tight that an ambiguous experiment returns *everything* rather than the
        two candidates it actually supports. That is correct behaviour and a
        useless fixture, so these records are deliberately middling.
        """
        records = [_record(kinetics.CYTOTOXIC, 0.45, f"drug{i}") for i in range(12)]
        return abstention.calibrate(records, confidence=0.9)

    def test_a_decisive_experiment_yields_one_mechanism(
        self, calibration: abstention.Calibration
    ) -> None:
        verdict = abstention.predict_set({kinetics.CYTOTOXIC: 0.99}, calibration)
        assert verdict["mechanisms"] == [kinetics.CYTOTOXIC]
        assert verdict["sufficient_evidence"] is True

    def test_an_ambiguous_experiment_yields_several(
        self, calibration: abstention.Calibration
    ) -> None:
        verdict = abstention.predict_set(
            {kinetics.CYTOTOXIC: 0.5, kinetics.MIXED: 0.5}, calibration
        )
        assert set(verdict["mechanisms"]) == {kinetics.CYTOTOXIC, kinetics.MIXED}
        assert verdict["sufficient_evidence"] is False

    def test_one_well_per_condition_yields_everything(
        self, calibration: abstention.Calibration
    ) -> None:
        """An unidentified parameter cannot be given a calibrated set."""
        verdict = abstention.predict_set(
            {kinetics.CYTOTOXIC: 1.0}, calibration, turnover_estimable=False
        )
        assert verdict["mechanisms"] == list(kinetics.MECHANISMS)
        assert "single well" in verdict["reason"]

    def test_an_experiment_unlike_the_calibration_yields_everything(
        self, calibration: abstention.Calibration
    ) -> None:
        """An empty set must never be read as 'no mechanism is possible'."""
        flat = dict.fromkeys(kinetics.MECHANISMS, 0.0)
        verdict = abstention.predict_set(flat, calibration)
        assert verdict["mechanisms"] == list(kinetics.MECHANISMS)
        assert "unlike the calibration" in verdict["reason"]

    def test_the_returned_order_is_stable(self, calibration: abstention.Calibration) -> None:
        verdict = abstention.predict_set(
            {kinetics.MIXED: 0.5, kinetics.CYTOTOXIC: 0.5}, calibration
        )
        assert verdict["mechanisms"] == [kinetics.CYTOTOXIC, kinetics.MIXED]

    def test_the_statement_refuses_to_name_one_when_several_survive(
        self, calibration: abstention.Calibration
    ) -> None:
        verdict = abstention.predict_set(
            {kinetics.CYTOTOXIC: 0.5, kinetics.MIXED: 0.5}, calibration
        )
        assert "not supported" in verdict["statement"]

    def test_the_statement_says_nothing_is_distinguished_when_all_survive(self) -> None:
        """A flat distribution under a loose threshold: everything genuinely survives."""
        loose = abstention.calibrate(
            [_record(kinetics.CYTOTOXIC, 0.15, f"drug{i}") for i in range(12)], confidence=0.9
        )
        verdict = abstention.predict_set(dict.fromkeys(kinetics.MECHANISMS, 0.25), loose)
        assert verdict["mechanisms"] == list(kinetics.MECHANISMS)
        assert "does not distinguish" in verdict["statement"]


class TestEvaluation:
    def test_always_returning_everything_is_caught_by_set_size(self) -> None:
        """Coverage is free; the metric that exposes the cheat is mean_set_size."""
        useless = abstention.Calibration(
            thresholds={"": 1.0}, confidence=0.9, groups_per_stratum={"": 99}, records=99
        )
        held = [_record(kinetics.CYTOTOXIC, 0.9, f"d{i}") for i in range(10)]
        report = abstention.evaluate(held, useless)
        assert report["coverage"] == 1.0
        assert report["mean_set_size"] == float(len(kinetics.MECHANISMS))
        assert report["abstention_rate"] == 1.0

    def test_a_confidently_wrong_answer_is_counted(self) -> None:
        """The number a reader would act on and be misled by."""
        strict = abstention.Calibration(
            thresholds={"": 0.05}, confidence=0.9, groups_per_stratum={"": 99}, records=99
        )
        wrong = CalibrationRecord(
            frequencies={kinetics.CYTOSTATIC: 1.0}, truth=kinetics.CYTOTOXIC, group="d0"
        )
        report = abstention.evaluate([wrong], strict)
        assert report["confidently_wrong_rate"] == 1.0
        assert report["coverage"] == 0.0

    def test_an_empty_evaluation_is_refused(self) -> None:
        with pytest.raises(AbstentionError, match="at least one record"):
            abstention.evaluate([], abstention.calibrate([_record(kinetics.MIXED, 0.9, "a")]))


class TestEndToEnd:
    """The whole pipeline on experiments the calibration never saw."""

    @staticmethod
    def _experiment(seed: int, replicates: int) -> tuple[dict[str, float], str, bool]:
        rng = np.random.default_rng(seed)
        # A different treatment each time, spanning all four mechanisms.
        birth_scale = float(rng.choice([1.0, 1.0, 0.45, 0.45]))
        death_increase = float(rng.choice([0.0, 0.035, 0.0, 0.035]))
        treated = kinetics.apply_treatment(
            CONTROL, birth_scale=birth_scale, death_increase=death_increase
        )
        truth = kinetics.classify(CONTROL, treated)
        control_wells = kinetics.simulate(
            CONTROL, SEEDED, HOURS, replicates=replicates, rng=np.random.default_rng(seed + 5000)
        )
        treated_wells = kinetics.simulate(
            treated, SEEDED, HOURS, replicates=replicates, rng=np.random.default_rng(seed + 9000)
        )
        got = inference.mechanism_distribution(
            control_wells, treated_wells, HOURS, initial=SEEDED, resamples=200, seed=seed
        )
        return got["frequencies"], truth, got["turnover_estimable"]

    def test_coverage_holds_on_unseen_experiments(self) -> None:
        """The guarantee, checked the only way that means anything."""
        calibration_records = [
            CalibrationRecord(*self._experiment(seed, replicates=30)[:2], group=f"exp{seed}")
            for seed in range(40)
        ]
        calibration = abstention.calibrate(calibration_records, confidence=0.9)
        assert calibration.degenerate is False

        held_out = [
            CalibrationRecord(*self._experiment(seed, replicates=30)[:2], group=f"exp{seed}")
            for seed in range(100, 140)
        ]
        report = abstention.evaluate(held_out, calibration)
        # Conformal guarantees coverage in expectation; a 40-experiment sample
        # fluctuates, so the check allows a margin rather than demanding the
        # nominal level exactly.
        assert report["coverage"] >= 0.8, report
        assert report["mean_set_size"] < len(kinetics.MECHANISMS), "sets are uninformative"

    def test_thin_experiments_abstain_more_than_rich_ones(self) -> None:
        """The behaviour the whole project is for, end to end."""

        def abstention_rate(replicates: int) -> float:
            records = [
                CalibrationRecord(*self._experiment(seed, replicates)[:2], group=f"exp{seed}")
                for seed in range(200, 220)
            ]
            calibration = abstention.calibrate(
                [
                    CalibrationRecord(*self._experiment(seed, 30)[:2], group=f"cal{seed}")
                    for seed in range(40)
                ],
                confidence=0.9,
            )
            return abstention.evaluate(records, calibration)["abstention_rate"]

        assert abstention_rate(3) >= abstention_rate(40)


class TestStratification:
    """A threshold borrowed from a richer design is not a guarantee.

    Calibrating on thirty-well experiments and predicting on three-well ones was
    measured at 53% coverage against a 90% target, with 45% of verdicts
    confidently wrong. Conformal coverage assumes calibration and test data are
    exchangeable, and a three-well experiment is simply not exchangeable with a
    thirty-well one. These tests hold the fix in place.
    """

    @staticmethod
    def _records(stratum: str, confident: float, count: int) -> list[CalibrationRecord]:
        return [
            CalibrationRecord(
                frequencies={
                    kinetics.CYTOTOXIC: confident,
                    **{
                        name: (1 - confident) / 3
                        for name in kinetics.MECHANISMS
                        if name != kinetics.CYTOTOXIC
                    },
                },
                truth=kinetics.CYTOTOXIC,
                group=f"{stratum}-drug{i}",
                stratum=stratum,
            )
            for i in range(count)
        ]

    def test_each_design_gets_its_own_threshold(self) -> None:
        calibration = abstention.calibrate(
            self._records("wells=3", 0.20, 12) + self._records("wells=30", 0.50, 12),
            confidence=0.9,
        )
        assert set(calibration.thresholds) == {"wells=3", "wells=30"}
        # The thin design is harder, so its threshold must be looser.
        assert calibration.thresholds["wells=3"] > calibration.thresholds["wells=30"]

    def test_a_thin_design_abstains_where_a_rich_one_decides(self) -> None:
        calibration = abstention.calibrate(
            self._records("wells=3", 0.20, 12) + self._records("wells=30", 0.50, 12),
            confidence=0.9,
        )
        frequencies = {
            kinetics.CYTOTOXIC: 0.55,
            kinetics.MIXED: 0.35,
            kinetics.CYTOSTATIC: 0.05,
            kinetics.NO_EFFECT: 0.05,
        }
        thin = abstention.predict_set(frequencies, calibration, stratum="wells=3")
        rich = abstention.predict_set(frequencies, calibration, stratum="wells=30")
        assert len(thin["mechanisms"]) > len(rich["mechanisms"])

    def test_an_uncalibrated_design_returns_everything(self) -> None:
        """It must not silently borrow the threshold of a design it resembles."""
        calibration = abstention.calibrate(self._records("wells=30", 0.95, 12), confidence=0.9)
        verdict = abstention.predict_set({kinetics.CYTOTOXIC: 0.99}, calibration, stratum="wells=3")
        assert verdict["mechanisms"] == list(kinetics.MECHANISMS)
        assert "no threshold is calibrated" in verdict["reason"]

    def test_a_design_below_the_minimum_has_no_threshold(self) -> None:
        calibration = abstention.calibrate(
            self._records("wells=3", 0.9, 2) + self._records("wells=30", 0.9, 12),
            confidence=0.9,
        )
        assert calibration.thresholds["wells=3"] == float("inf")
        assert math.isfinite(calibration.thresholds["wells=30"])

    def test_evaluation_uses_each_record_own_design(self) -> None:
        """Evaluating without the stratum would recreate the bug it should catch."""
        calibration = abstention.calibrate(
            self._records("wells=3", 0.20, 12) + self._records("wells=30", 0.50, 12),
            confidence=0.9,
        )
        thin_only = abstention.evaluate(self._records("wells=3", 0.20, 12), calibration)
        rich_only = abstention.evaluate(self._records("wells=30", 0.50, 12), calibration)
        assert thin_only["mean_set_size"] > rich_only["mean_set_size"]

    def test_asking_a_multi_design_calibration_for_one_threshold_is_refused(self) -> None:
        calibration = abstention.calibrate(
            self._records("wells=3", 0.2, 12) + self._records("wells=30", 0.5, 12)
        )
        with pytest.raises(AbstentionError, match="several designs"):
            _ = calibration.threshold
