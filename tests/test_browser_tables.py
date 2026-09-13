"""The browser's tables must match the Python that generated them -- as numbers.

The first check diffed the generated files as text and failed on GitHub's Linux
runner the first time it ran: scipy there differs from macOS in about the 15th
significant digit. That is noise between machines, not drift, so the check now
compares numbers within a relative 1e-12, the bar the browser's own parity test
holds itself to. These tests pin both halves of that: noise passes, drift fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_chi2_table as chi2  # noqa: E402
import generate_parity_fixtures as parity  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _scaled(tables: dict[str, list[float]], factor: float) -> dict[str, list[float]]:
    return {name: [value * factor for value in values] for name, values in tables.items()}


class TestTheChiSquaredTable:
    def test_the_committed_table_matches_scipy(self) -> None:
        committed = chi2.parse((ROOT / chi2.OUT).read_text(encoding="utf-8"))
        assert chi2.differences(committed, chi2.tables()) == []

    def test_rendering_and_reading_back_round_trip(self) -> None:
        fresh = chi2.tables()
        assert chi2.differences(chi2.parse(chi2.render(fresh)), fresh) == []

    def test_last_digit_noise_between_machines_is_not_drift(self) -> None:
        fresh = chi2.tables()
        assert chi2.differences(_scaled(fresh, 1 + 3e-15), fresh) == []

    def test_a_real_change_is_caught_and_located(self) -> None:
        fresh = chi2.tables()
        drifted = {name: list(values) for name, values in fresh.items()}
        drifted["C95_LO"][2] *= 1 + 1e-9
        problems = chi2.differences(drifted, fresh)
        assert len(problems) == 1
        assert problems[0].startswith("C95_LO[2]")

    def test_a_missing_array_is_caught(self) -> None:
        fresh = chi2.tables()
        partial = {name: values for name, values in fresh.items() if name != "C90_HI"}
        assert chi2.differences(partial, fresh)

    def test_a_short_array_is_caught(self) -> None:
        fresh = chi2.tables()
        short = {name: list(values) for name, values in fresh.items()}
        short["C90_LO"].pop()
        assert chi2.differences(short, fresh)


class TestTheParityFixtures:
    def _fresh(self) -> dict[str, Any]:
        return parity.fixtures()

    def test_the_committed_fixtures_match_the_package(self) -> None:
        committed = json.loads((ROOT / parity.OUT).read_text(encoding="utf-8"))
        assert parity.differences(committed, self._fresh()) == []

    def test_last_digit_noise_is_not_drift(self) -> None:
        fresh = self._fresh()
        noisy = json.loads(json.dumps(fresh))
        noisy["costs"][0]["cost"] *= 1 + 3e-15
        noisy["intervals"][2]["turnoverRatio"] *= 1 - 3e-15
        assert parity.differences(noisy, fresh) == []

    def test_a_real_change_is_caught(self) -> None:
        fresh = self._fresh()
        drifted = json.loads(json.dumps(fresh))
        drifted["intervals"][2]["turnoverRatio"] *= 1 + 1e-9
        assert parity.differences(drifted, fresh)

    def test_an_integer_must_match_exactly(self) -> None:
        fresh = self._fresh()
        drifted = json.loads(json.dumps(fresh))
        drifted["costs"][0]["wellsConsumed"] += 1
        assert parity.differences(drifted, fresh)

    def test_a_missing_value_must_stay_missing(self) -> None:
        """A well count too small to estimate is null, and must not turn into a number."""
        fresh = self._fresh()
        drifted = json.loads(json.dumps(fresh))
        assert drifted["intervals"][0]["relativeWidth"] is None
        drifted["intervals"][0]["relativeWidth"] = 0.0
        assert parity.differences(drifted, fresh)

    def test_a_missing_section_is_caught(self) -> None:
        fresh = self._fresh()
        drifted = json.loads(json.dumps(fresh))
        del drifted["costs"]
        assert parity.differences(drifted, fresh)
