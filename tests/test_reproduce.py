"""The reproduce command and the report must describe the same pipeline.

A reproduction script that silently skips an artifact the report quotes, or runs
a stage before the one it depends on, would make "reproducible" a claim rather
than a property. These tests tie the stage list to the report's citations, to
the pre-registered plan, and to the scripts that actually exist.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tribovision import preregistration


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("reproduce", Path("scripts/reproduce.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: @dataclass resolves postponed annotations by
    # looking the defining module up in sys.modules, and a script loaded from a
    # path is not there unless it is put there.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reproduce = _load()


class TestTheStageList:
    def test_stage_names_are_unique(self) -> None:
        names = [stage.name for stage in reproduce.STAGES]
        assert len(names) == len(set(names))

    def test_every_script_a_stage_runs_exists(self) -> None:
        for stage in reproduce.STAGES:
            for command in stage.commands:
                assert (reproduce.REPO / command[0]).is_file(), (stage.name, command[0])

    def test_no_two_stages_write_the_same_file(self) -> None:
        outputs = [path for stage in reproduce.STAGES for path in stage.outputs]
        assert len(outputs) == len(set(outputs))

    def test_a_stage_never_needs_what_a_later_stage_makes(self) -> None:
        order = {stage.name: index for index, stage in enumerate(reproduce.STAGES)}
        for index, stage in enumerate(reproduce.STAGES):
            for path in stage.requires:
                maker = reproduce.producer_of(path)
                if maker is not None:
                    assert order[maker.name] < index, (stage.name, path, maker.name)


class TestAgreementWithTheReport:
    def test_the_report_cites_something(self) -> None:
        assert reproduce.cited_artifacts(), "a report citing nothing would pass vacuously"

    def test_every_artifact_the_report_cites_is_made_by_some_stage(self) -> None:
        """A citation nothing regenerates is a number nobody can check."""
        for path in reproduce.cited_artifacts():
            assert reproduce.producer_of(path) is not None, path


class TestAgreementWithThePlan:
    def test_the_sweep_covers_exactly_the_preregistered_lines(self) -> None:
        plan = preregistration.load_plan(reproduce.REPO / reproduce.PREREGISTRATION)
        planned = set(plan["confirmatory_cell_lines"]) | set(plan["exploratory_cell_lines"])
        assert set(reproduce.DAMAGE_LINES) == planned

    def test_the_verdict_reads_every_file_the_sweep_writes(self) -> None:
        """A sweep output the evaluator's pattern misses is a result silently dropped."""
        spec = importlib.util.spec_from_file_location(
            "evaluate", Path("scripts/evaluate_preregistration.py")
        )
        assert spec is not None and spec.loader is not None
        evaluate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluate)
        pattern = str(evaluate.ARTIFACTS / "blindness_combined_*.json")
        sweep = next(stage for stage in reproduce.STAGES if stage.name == "damage-sweep")
        for output in sweep.outputs:
            assert fnmatch.fnmatch(output, pattern), output


class TestSelection:
    def test_fast_skips_the_slow_stages_and_only_those(self) -> None:
        chosen = {stage.name for stage in reproduce.select(None, True)}
        assert chosen == {stage.name for stage in reproduce.STAGES if not stage.slow}

    def test_named_stages_run_in_dependency_order(self) -> None:
        chosen = reproduce.select(["selectivity", "case-study"], False)
        assert [stage.name for stage in chosen] == ["case-study", "selectivity"]

    def test_an_unknown_stage_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no stage named"):
            reproduce.select(["nonsense"], False)


class TestRunning:
    def test_a_dry_run_executes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(*args: Any, **kwargs: Any) -> None:
            pytest.fail("a dry run started a process")

        monkeypatch.setattr(reproduce.subprocess, "run", refuse)
        reproduce.run(list(reproduce.STAGES), dry_run=True)

    def test_a_failing_table_command_leaves_the_old_table_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(reproduce, "REPO", tmp_path)
        (tmp_path / "table.md").write_text("the old table\n", encoding="utf-8")

        def half_written(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            kwargs["stdout"].write("a truncated new tab")
            return subprocess.CompletedProcess(argv, 1)

        monkeypatch.setattr(reproduce.subprocess, "run", half_written)
        stage = reproduce.Stage("table", (("x.py",),), stdout_to="table.md")
        assert reproduce.run([stage]) == 1
        assert (tmp_path / "table.md").read_text(encoding="utf-8") == "the old table\n"
        assert not list(tmp_path.glob("*.partial")), "the temporary file must be cleaned up"

    def test_a_successful_table_command_replaces_the_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(reproduce, "REPO", tmp_path)
        (tmp_path / "table.md").write_text("the old table\n", encoding="utf-8")

        def complete(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            kwargs["stdout"].write("the new table\n")
            return subprocess.CompletedProcess(argv, 0)

        monkeypatch.setattr(reproduce.subprocess, "run", complete)
        stage = reproduce.Stage("table", (("x.py",),), stdout_to="table.md")
        assert reproduce.run([stage]) == 0
        assert (tmp_path / "table.md").read_text(encoding="utf-8") == "the new table\n"

    def test_a_missing_input_no_stage_makes_is_a_failure_not_a_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(reproduce, "REPO", tmp_path)
        stage = reproduce.Stage("needs-weights", (("x.py",),), requires=("weights.pt",))
        assert reproduce.run([stage]) == 1
        assert "made by no stage here" in capsys.readouterr().out

    def test_a_stage_downstream_of_a_skipped_one_is_skipped_not_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Under --fast the verdict cannot run without the sweep; that is a skip, not an error."""
        monkeypatch.setattr(reproduce, "REPO", tmp_path)
        verdict = next(s for s in reproduce.STAGES if s.name == "preregistered-verdict")
        assert reproduce.run([verdict]) == 0
        assert "SKIP" in capsys.readouterr().out
