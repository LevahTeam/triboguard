"""Applying the system to somebody's paper, without overreaching.

The risk here is not a wrong number, it is a wrong register: a tool that reads
as a verdict on the biology when it is only a verdict on what a design can
carry. Several tests below check the wording as carefully as the logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triboguard import case_study
from triboguard.case_study import CaseStudyError

PAPER = Path(__file__).resolve().parents[1] / "case_studies" / "tribonema_2022.json"


@pytest.fixture
def payload() -> dict:
    return case_study.load(PAPER)


class TestLoading:
    def test_the_shipped_case_study_is_valid(self, payload: dict) -> None:
        assert payload["design"]["assay"] == "MTS"
        assert payload["design"]["destructive"] is True

    def test_it_records_what_the_paper_does_not_report(self, payload: dict) -> None:
        """The absences are evidence too, and the biggest one is the replicate count."""
        missing = " ".join(payload["not_reported"]).lower()
        assert "replicate" in missing
        assert "per-well" in missing

    def test_it_records_the_internal_inconsistency_without_resolving_it(
        self, payload: dict
    ) -> None:
        """The methods say 48 hours and every result says 24. Not ours to pick."""
        text = " ".join(payload["internal_inconsistencies"])
        assert "48" in text and "24" in text
        assert "does not choose one for it" in text

    def test_a_capability_that_does_not_exist_is_refused(self, tmp_path: Path) -> None:
        """A typo in a capability name would silently mark a claim unsupported."""
        bad = json.loads(PAPER.read_text(encoding="utf-8"))
        bad["provides"].append("mind_reading")
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(CaseStudyError, match="unknown capabilities"):
            case_study.load(path)

    def test_a_claim_requiring_an_unknown_capability_is_refused(self, tmp_path: Path) -> None:
        bad = json.loads(PAPER.read_text(encoding="utf-8"))
        bad["claims"][0]["requires"].append("telepathy")
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(CaseStudyError, match="requires unknown"):
            case_study.load(path)

    def test_a_missing_section_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "thin.json"
        path.write_text(json.dumps({"source": {}}), encoding="utf-8")
        with pytest.raises(CaseStudyError, match="missing required section"):
            case_study.load(path)


class TestClaimAssessment:
    def test_the_effect_itself_is_supported(self, payload: dict) -> None:
        """The project must not read as a debunking. The reported effect stands."""
        claims = {c["id"]: c for c in case_study.assess_claims(payload)}
        assert claims["affected"]["supported"] is True
        assert claims["comparable_to_doxorubicin"]["supported"] is True

    def test_counting_deaths_is_not_supported_by_a_metabolic_assay(self, payload: dict) -> None:
        claims = {c["id"]: c for c in case_study.assess_claims(payload)}
        died = claims["forty_percent_died"]
        assert died["supported"] is False
        assert "cell_counts" in died["missing_capabilities"]
        assert "not cell number" in died["reasons"][0]

    def test_the_mechanism_claim_fails_for_two_separate_reasons(self, payload: dict) -> None:
        """Wrong readout *and* no replicates. Fixing one would not be enough."""
        claims = {c["id"]: c for c in case_study.assess_claims(payload)}
        mechanism = claims["cytotoxic_not_cytostatic"]
        assert set(mechanism["missing_capabilities"]) == {"cell_counts", "replicate_variance"}

    def test_support_is_set_inclusion_and_nothing_else(self, payload: dict) -> None:
        """Adding the missing capability must flip the verdict, or the rule is decoration."""
        before = {c["id"]: c["supported"] for c in case_study.assess_claims(payload)}
        assert before["forty_percent_died"] is False
        richer = {**payload, "provides": [*payload["provides"], "cell_counts"]}
        after = {c["id"]: c["supported"] for c in case_study.assess_claims(richer)}
        assert after["forty_percent_died"] is True

    def test_removing_a_capability_withdraws_a_supported_claim(self, payload: dict) -> None:
        poorer = {
            **payload,
            "provides": [c for c in payload["provides"] if c != "matched_untreated_control"],
        }
        claims = {c["id"]: c for c in case_study.assess_claims(poorer)}
        assert claims["affected"]["supported"] is False

    def test_every_unsupported_claim_says_why(self, payload: dict) -> None:
        for claim in case_study.assess_claims(payload):
            if not claim["supported"]:
                assert claim["reasons"], claim["id"]
                assert all(reason.strip() for reason in claim["reasons"])


class TestResolvingPower:
    def test_two_wells_barely_constrain_anything(self) -> None:
        rows = {row["wells"]: row for row in case_study.resolving_power()}
        assert rows[2]["relative_turnover_width"] > 100

    def test_it_improves_with_wells(self) -> None:
        widths = [
            row["relative_turnover_width"]
            for row in case_study.resolving_power()
            if row["relative_turnover_width"] is not None
        ]
        assert widths == sorted(widths, reverse=True)

    def test_it_needs_no_measurements_from_the_paper(self) -> None:
        """The whole reason a study without a reported replicate count is analysable."""
        assert case_study.resolving_power(well_counts=(3,))[0]["wells"] == 3


class TestFollowUp:
    def test_the_assumed_replicate_count_is_labelled_an_assumption(self, payload: dict) -> None:
        plan = case_study.follow_up(payload)
        assert plan["assumed_wells"] == 3
        assert "states no replicate count" in plan["assumption_note"]

    def test_switching_to_imaging_costs_less_for_the_same_information(self, payload: dict) -> None:
        """MTS destroys the well it reads; imaging does not."""
        plan = case_study.follow_up(payload)
        mts = plan["as_run_with_mts"]["current"]
        imaging = plan["switched_to_imaging"]["current"]
        assert mts["wells_consumed"] > imaging["wells_consumed"]
        assert mts["cost"] > imaging["cost"]
        assert mts["relative_turnover_width"] == imaging["relative_turnover_width"]

    def test_it_prices_the_answer_in_wells(self, payload: dict) -> None:
        plan = case_study.follow_up(payload)
        assert plan["as_run_with_mts"]["target"]["wells_required"] > 3


class TestTheStatement:
    def test_it_does_not_read_as_a_verdict_on_the_biology(self) -> None:
        """The single most important sentence in the project to get right."""
        statement = case_study.report(PAPER)["statement"]
        assert "the reported effect stands" in statement
        assert "none of this is evidence against the extract" in statement.lower()

    def test_it_names_what_the_assay_actually_measures(self) -> None:
        assert "dehydrogenase" in case_study.report(PAPER)["statement"]

    def test_it_quantifies_the_gap_rather_than_gesturing_at_it(self) -> None:
        statement = case_study.report(PAPER)["statement"]
        # The ratio of the interval's endpoints, not its width relative to the
        # estimate. An earlier version quoted the width here and called it a
        # factor, which understated the gap by nearly fourfold.
        assert "factor of about 146" in statement
        assert "66 wells per condition" in statement

    def test_the_report_survives_a_round_trip_through_json(self) -> None:
        report = case_study.report(PAPER)
        assert json.loads(json.dumps(report))["unsupported"] == report["unsupported"]

    def test_more_conclusions_fail_than_pass(self) -> None:
        report = case_study.report(PAPER)
        assert report["unsupported"] > report["supported"]
        assert report["supported"] >= 1, "a case study that supports nothing is not credible"


class TestTheTwoFactors:
    """The ratio and the width are reported side by side and never interchanged.

    An earlier version printed the relative width under the heading "factor",
    which understated the gap at three wells from 146 to 39. Both numbers are
    real; they answer different questions.
    """

    def test_both_are_reported_for_every_well_count(self) -> None:
        for row in case_study.resolving_power():
            assert row["turnover_ratio"] is not None
            assert row["relative_turnover_width"] is not None

    def test_they_are_not_the_same_number(self) -> None:
        row = next(r for r in case_study.resolving_power() if r["wells"] == 3)
        assert row["turnover_ratio"] > 3 * row["relative_turnover_width"]

    def test_the_ratio_at_three_wells_matches_the_statement(self) -> None:
        """The table and the prose must not disagree with each other."""
        report = case_study.report(PAPER)
        row = next(r for r in report["resolving_power"] if r["wells"] == 3)
        assert round(row["turnover_ratio"]) == 146
        assert "factor of about 146" in report["statement"]

    def test_both_columns_improve_with_wells(self) -> None:
        rows = case_study.resolving_power()
        for key in ("turnover_ratio", "relative_turnover_width"):
            values = [r[key] for r in rows]
            assert values == sorted(values, reverse=True), key
