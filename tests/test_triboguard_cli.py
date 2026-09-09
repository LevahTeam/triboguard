"""The command line, which is how anyone outside this repository reaches the package.

Before this the only way in was a script that pushed ``src`` onto sys.path, which
works from the repository root and nowhere else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triboguard.cli import main

PAPER = Path(__file__).resolve().parents[1] / "case_studies" / "tribonema_2022.json"


class TestDesign:
    def test_it_reports_a_design(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["design", "--wells", "3", "--times", "0", "24", "48"]) == 0
        out = capsys.readouterr().out
        assert "3 wells x 3 timepoints" in out
        assert "split known to within" in out

    def test_the_two_uncertainty_measures_are_labelled_apart(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Unlabelled and side by side they read as a contradiction."""
        main(["design", "--wells", "3", "--times", "0", "24"])
        out = capsys.readouterr().out
        assert "ratio of the interval's ends" in out
        assert "its span over the estimate" in out

    def test_a_destructive_assay_consumes_more_wells(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["design", "--wells", "3", "--times", "0", "24", "48", "--assay", "mts"])
        mts = capsys.readouterr().out
        main(["design", "--wells", "3", "--times", "0", "24", "48", "--assay", "imaging"])
        imaging = capsys.readouterr().out
        assert "wells consumed        9" in mts
        assert "wells consumed        3" in imaging

    def test_json_output_is_machine_readable(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["design", "--wells", "6", "--times", "0", "24", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["current"]["wells"] == 6
        assert payload["turnover_ratio"] > 1

    def test_an_impossible_design_is_refused_without_a_traceback(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A refusal is the tool working, so it reads as a message not a crash."""
        assert main(["design", "--wells", "3", "--times", "24"]) == 2
        assert "triboguard:" in capsys.readouterr().err


class TestCaseStudy:
    def test_it_reads_the_shipped_case_study(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["case-study", str(PAPER)]) == 0
        out = capsys.readouterr().out
        assert "2 supported, 5 not established" in out

    def test_a_missing_section_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "thin.json"
        path.write_text(json.dumps({"source": {}}), encoding="utf-8")
        assert main(["case-study", str(path)]) == 2
        assert "missing required section" in capsys.readouterr().err

    def test_json_output_round_trips(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["case-study", str(PAPER), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["unsupported"] == 5


class TestBoundary:
    def test_a_small_sweep_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            [
                "boundary",
                "--wells",
                "3",
                "--effects",
                "0.4",
                "--calibration",
                "12",
                "--test",
                "12",
                "--resamples",
                "20",
                "--json",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["cells"] == 1

    def test_an_impossible_sweep_is_refused(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["boundary", "--wells", "0", "--effects", "0.4"]) == 2
        assert "well counts must be positive" in capsys.readouterr().err


def test_the_console_script_is_declared() -> None:
    """The entry point is what makes any of the above reachable after install."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    assert config["project"]["scripts"]["triboguard"] == "triboguard.cli:main"
