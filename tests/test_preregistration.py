"""The pre-registered verdict, applied mechanically.

These tests exist because a pre-registration is only worth what its evaluation
is: if the code that applies the criteria can be talked round -- a missing
result counted as a pass, a failed control ignored, an exploratory line
sneaking into the verdict, an artifact scored under an edited plan -- then the
plan protects nothing. Each of those is pinned here.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from tribovision import preregistration as pre

PLAN = Path("preregistration/damage_blindness_v1.json")


def _row(difference: float, low: float, high: float, *, intact_rate: float = 0.9) -> dict[str, Any]:
    return {
        "mean_intact_rate": intact_rate,
        "paired_difference": {
            "evaluated": True,
            "difference": difference,
            "ci_low": low,
            "ci_high": high,
        },
    }


def _summary(plan: dict[str, Any], *, gap: float = 0.5) -> dict[str, Any]:
    """A result that passes everything: no control gap, a gap growing with damage."""
    summary: dict[str, Any] = {}
    for line in plan["confirmatory_cell_lines"] + plan["exploratory_cell_lines"]:
        for segmenter in plan["design"]["segmenters"]:
            for severity in plan["design"]["severities"]:
                for coverage in plan["design"]["coverage_thresholds"]:
                    if severity == 0.0:
                        row = _row(0.0, -0.02, 0.02)
                    else:
                        estimate = gap * severity / 0.6
                        row = _row(estimate, estimate - 0.05, estimate + 0.05)
                    summary[pre.summary_key(line, severity, segmenter, coverage)] = row
    return summary


@pytest.fixture
def plan() -> dict[str, Any]:
    return pre.load_plan(PLAN)


class TestThePlan:
    def test_the_committed_plan_is_well_formed(self, plan: dict[str, Any]) -> None:
        assert plan["confirmatory_cell_lines"]
        assert not set(plan["confirmatory_cell_lines"]) & set(plan["exploratory_cell_lines"])

    def test_the_line_already_seen_is_not_confirmatory(self, plan: dict[str, Any]) -> None:
        """A172 results were seen before the plan was written, so it cannot confirm it."""
        assert "A172" in plan["exploratory_cell_lines"]
        assert "A172" not in plan["confirmatory_cell_lines"]

    def test_a_line_cannot_be_both(self, tmp_path: Path, plan: dict[str, Any]) -> None:
        bad = copy.deepcopy(plan)
        bad["exploratory_cell_lines"].append(bad["confirmatory_cell_lines"][0])
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(pre.PreregistrationError, match="both"):
            pre.load_plan(path)

    def test_a_missing_criterion_is_refused(self, tmp_path: Path, plan: dict[str, Any]) -> None:
        bad = copy.deepcopy(plan)
        del bad["criteria"]["negative_control"]
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(pre.PreregistrationError, match="negative_control"):
            pre.load_plan(path)

    def test_the_primary_segmenter_must_be_in_the_design(
        self, tmp_path: Path, plan: dict[str, Any]
    ) -> None:
        bad = copy.deepcopy(plan)
        bad["primary_endpoint"]["primary_segmenter"] = "something_else"
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(pre.PreregistrationError, match="primary segmenter"):
            pre.load_plan(path)

    def test_the_sweep_runs_the_design_the_plan_describes(self, plan: dict[str, Any]) -> None:
        """The plan and the code that executes it must not be able to drift apart."""
        spec = importlib.util.spec_from_file_location(
            "sweep", Path("scripts/run_damage_blindness.py")
        )
        assert spec is not None and spec.loader is not None
        sweep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sweep)
        design = plan["design"]
        assert design["severities"] == list(sweep.SEVERITIES)
        assert design["images_per_cell_line"] == sweep.IMAGES
        assert design["coverage_thresholds"] == list(sweep.COVERAGE_THRESHOLDS)
        assert design["segmenters"] == list(sweep.SEGMENTERS)
        assert set(_lines(plan)) <= set(sweep.MANIFESTS)
        assert Path(sweep.PREREGISTRATION) == PLAN


def _lines(plan: dict[str, Any]) -> list[str]:
    return plan["confirmatory_cell_lines"] + plan["exploratory_cell_lines"]


class TestTheVerdict:
    def test_a_clean_result_replicates(self, plan: dict[str, Any]) -> None:
        assert pre.evaluate(plan, _summary(plan))["verdict"] == "replicated"

    def test_one_failing_line_is_only_partial(self, plan: dict[str, Any]) -> None:
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        summary[pre.summary_key(line, 0.6, "cellpose", 0.5)] = _row(0.08, 0.03, 0.13)
        outcome = pre.evaluate(plan, summary)
        assert outcome["verdict"] == "partially replicated"
        assert outcome["confirmatory"][line] == pre.FAIL

    def test_failing_everywhere_is_not_replicated(self, plan: dict[str, Any]) -> None:
        summary = _summary(plan)
        for line in plan["confirmatory_cell_lines"]:
            summary[pre.summary_key(line, 0.6, "cellpose", 0.5)] = _row(0.02, -0.03, 0.07)
        assert pre.evaluate(plan, summary)["verdict"] == "not replicated"

    def test_the_minimum_effect_is_strict(self, plan: dict[str, Any]) -> None:
        """A lower bound sitting exactly on the margin has not cleared it."""
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        summary[pre.summary_key(line, 0.6, "cellpose", 0.5)] = _row(0.15, 0.10, 0.20)
        assert pre.evaluate(plan, summary)["confirmatory"][line] == pre.FAIL

    def test_a_failed_control_voids_the_line_instead_of_passing_it(
        self, plan: dict[str, Any]
    ) -> None:
        """A gap with no damage means the split itself is biased; nothing downstream reads."""
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        summary[pre.summary_key(line, 0.0, "cellpose", 0.5)] = _row(0.06, 0.03, 0.09)
        outcome = pre.evaluate(plan, summary)
        assert outcome["confirmatory"][line] == pre.VOID
        assert outcome["verdict"] == "partially replicated", "a void line blocks 'replicated'"

    def test_a_control_biased_the_other_way_also_voids_the_line(self, plan: dict[str, Any]) -> None:
        """Damaged cells found *more* often with no damage applied is a biased split too.

        Mutation testing found this unguarded: every other control case sat above
        the margin, so a check reading only the upper bound passed them all.
        """
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        summary[pre.summary_key(line, 0.0, "cellpose", 0.5)] = _row(-0.06, -0.09, -0.03)
        outcome = pre.evaluate(plan, summary)
        assert outcome["criteria"]["negative_control"][line]["cellpose"]["status"] == pre.FAIL
        assert outcome["confirmatory"][line] == pre.VOID

    def test_a_segmenter_blind_to_healthy_cells_voids_the_line(self, plan: dict[str, Any]) -> None:
        """No gap because nothing is found at all is a broken measurement, not a null."""
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        summary[pre.summary_key(line, 0.0, "cellpose", 0.5)] = _row(
            0.0, -0.01, 0.01, intact_rate=0.3
        )
        summary[pre.summary_key(line, 0.6, "cellpose", 0.5)] = _row(0.02, -0.02, 0.06)
        outcome = pre.evaluate(plan, summary)
        assert outcome["confirmatory"][line] == pre.VOID, "must not read as 'not replicated'"
        assert outcome["criteria"]["baseline_detection"][line]["status"] == pre.FAIL

    def test_baseline_detection_at_the_bar_passes(self, plan: dict[str, Any]) -> None:
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        summary[pre.summary_key(line, 0.0, "cellpose", 0.5)] = _row(
            0.0, -0.01, 0.01, intact_rate=0.5
        )
        assert pre.check_baseline_detection(plan, summary)[line]["status"] == pre.PASS

    def test_a_missing_result_is_never_a_pass(self, plan: dict[str, Any]) -> None:
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        del summary[pre.summary_key(line, 0.6, "cellpose", 0.5)]
        outcome = pre.evaluate(plan, summary)
        assert outcome["confirmatory"][line] == pre.MISSING
        assert outcome["verdict"] != "replicated"

    def test_an_unevaluable_interval_counts_as_missing(self, plan: dict[str, Any]) -> None:
        """Too few groups to bootstrap is an absent result, not a failed one."""
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        summary[pre.summary_key(line, 0.6, "cellpose", 0.5)] = {
            "paired_difference": {"evaluated": False, "reason": "Fewer than three clusters."}
        }
        assert pre.evaluate(plan, summary)["confirmatory"][line] == pre.MISSING

    def test_nothing_at_all_is_not_evaluable(self, plan: dict[str, Any]) -> None:
        assert pre.evaluate(plan, {})["verdict"] == "not evaluable"

    def test_exploratory_lines_never_change_the_verdict(self, plan: dict[str, Any]) -> None:
        summary = _summary(plan)
        for line in plan["exploratory_cell_lines"]:
            for segmenter in plan["design"]["segmenters"]:
                summary[pre.summary_key(line, 0.6, segmenter, 0.5)] = _row(0.0, -0.1, 0.1)
        outcome = pre.evaluate(plan, summary)
        assert outcome["verdict"] == "replicated"
        assert all(status == pre.FAIL for status in outcome["exploratory"].values())

    def test_evaluation_does_not_modify_its_inputs(self, plan: dict[str, Any]) -> None:
        summary = _summary(plan)
        before_plan, before_summary = copy.deepcopy(plan), copy.deepcopy(summary)
        pre.evaluate(plan, summary)
        assert plan == before_plan and summary == before_summary


class TestTheSecondaryCriteria:
    def test_two_of_three_segmenters_is_general(self, plan: dict[str, Any]) -> None:
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        summary[pre.summary_key(line, 0.6, "unet_watershed", 0.5)] = _row(0.05, 0.0, 0.1)
        result = pre.check_segmenter_generality(plan, summary)[line]
        assert result["status"] == pre.PASS
        assert "unet_watershed" not in result["passing"]

    def test_one_of_three_is_not(self, plan: dict[str, Any]) -> None:
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        for segmenter in ("three_class", "unet_watershed"):
            summary[pre.summary_key(line, 0.6, segmenter, 0.5)] = _row(0.05, 0.0, 0.1)
        assert pre.check_segmenter_generality(plan, summary)[line]["status"] == pre.FAIL

    def test_absent_segmenters_that_could_decide_it_leave_it_undecided(
        self, plan: dict[str, Any]
    ) -> None:
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        for segmenter in ("three_class", "unet_watershed"):
            del summary[pre.summary_key(line, 0.6, segmenter, 0.5)]
        assert pre.check_segmenter_generality(plan, summary)[line]["status"] == pre.MISSING

    def test_the_result_must_survive_both_thresholds(self, plan: dict[str, Any]) -> None:
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        summary[pre.summary_key(line, 0.6, "cellpose", 0.7)] = _row(0.08, 0.03, 0.13)
        assert pre.check_threshold_robustness(plan, summary)[line]["status"] == pre.FAIL

    def test_dose_response_tolerates_noise(self, plan: dict[str, Any]) -> None:
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        rising = pre.evaluate(plan, summary)["criteria"]["dose_response"][line]
        assert rising["status"] == pre.PASS
        # A dip smaller than the tolerance between 0.4 and 0.6 is noise.
        at_four = summary[pre.summary_key(line, 0.4, "cellpose", 0.5)]["paired_difference"]
        estimate = at_four["difference"] - 0.01
        summary[pre.summary_key(line, 0.6, "cellpose", 0.5)] = _row(estimate, 0.2, 0.4)
        assert pre.check_dose_response(plan, summary)[line]["status"] == pre.PASS

    def test_dose_response_catches_a_real_fall(self, plan: dict[str, Any]) -> None:
        summary = _summary(plan)
        line = plan["confirmatory_cell_lines"][0]
        at_four = summary[pre.summary_key(line, 0.4, "cellpose", 0.5)]["paired_difference"]
        estimate = at_four["difference"] - 0.10
        summary[pre.summary_key(line, 0.6, "cellpose", 0.5)] = _row(estimate, 0.1, 0.3)
        assert pre.check_dose_response(plan, summary)[line]["status"] == pre.FAIL


class TestMergingArtifacts:
    def _artifact(
        self, tmp_path: Path, name: str, sha: str | None, summary: dict[str, Any]
    ) -> Path:
        design: dict[str, Any] = {}
        if sha is not None:
            design["preregistration"] = {"path": "plan.json", "sha256": sha}
        path = tmp_path / name
        path.write_text(
            json.dumps({"design": design, "summary": summary, "environment": {}}),
            encoding="utf-8",
        )
        return path

    def test_artifacts_under_the_same_plan_merge(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.json"
        plan.write_text("{}", encoding="utf-8")
        sha = pre.plan_fingerprint(plan)["sha256"]
        first = self._artifact(tmp_path, "a.json", sha, {"MCF7/0.6/cellpose/cov0.5": {}})
        second = self._artifact(tmp_path, "b.json", sha, {"SkBr3/0.6/cellpose/cov0.5": {}})
        merged, sources = pre.merge_artifacts([first, second], plan)
        assert set(merged) == {"MCF7/0.6/cellpose/cov0.5", "SkBr3/0.6/cellpose/cov0.5"}
        assert len(sources) == 2

    def test_an_artifact_from_an_edited_plan_is_refused(self, tmp_path: Path) -> None:
        """The whole point: changing the criteria afterwards invalidates old results."""
        plan = tmp_path / "plan.json"
        plan.write_text('{"minimum_effect": 0.1}', encoding="utf-8")
        old_sha = pre.plan_fingerprint(plan)["sha256"]
        artifact = self._artifact(tmp_path, "a.json", old_sha, {})
        plan.write_text('{"minimum_effect": 0.05}', encoding="utf-8")
        with pytest.raises(pre.PreregistrationError, match="different version"):
            pre.merge_artifacts([artifact], plan)

    def test_an_artifact_recording_no_plan_is_refused(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.json"
        plan.write_text("{}", encoding="utf-8")
        artifact = self._artifact(tmp_path, "a.json", None, {})
        with pytest.raises(pre.PreregistrationError, match="records no plan"):
            pre.merge_artifacts([artifact], plan)

    def test_the_same_result_twice_is_refused(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.json"
        plan.write_text("{}", encoding="utf-8")
        sha = pre.plan_fingerprint(plan)["sha256"]
        row = {"MCF7/0.6/cellpose/cov0.5": {}}
        first = self._artifact(tmp_path, "a.json", sha, row)
        second = self._artifact(tmp_path, "b.json", sha, row)
        with pytest.raises(pre.PreregistrationError, match="more than one"):
            pre.merge_artifacts([first, second], plan)

    def test_nothing_to_merge_is_refused(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.json"
        plan.write_text("{}", encoding="utf-8")
        with pytest.raises(pre.PreregistrationError, match="no artifacts"):
            pre.merge_artifacts([], plan)
