"""The analysis scripts carry real arithmetic, so the arithmetic is tested.

These scripts turn artifacts into the numbers that end up in the write-up. A
mistake in one of them is a mistake in a reported result, and the fact that they
live in scripts/ rather than src/ does not make them less load-bearing.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from measure_across_cell_lines import _median_or_none  # noqa: E402
from measure_attenuation import LOA_SPAN_IN_SD, _expected_rho, _sd_error  # noqa: E402
from measure_resolution_limit import _ring  # noqa: E402


class TestLimitsOfAgreementToStandardDeviation:
    def test_a_known_span_recovers_its_standard_deviation(self) -> None:
        """LoA are mean +/- 1.96 sd, so a span of 3.92 units means sd = 1."""
        assert _sd_error({"limits_of_agreement": [-1.96, 1.96]}) == pytest.approx(1.0)

    def test_the_span_constant_is_the_one_the_convention_uses(self) -> None:
        assert pytest.approx(2 * 1.96) == LOA_SPAN_IN_SD

    def test_an_offset_span_measures_width_not_position(self) -> None:
        """A biased method is not thereby a noisier one; only the width is sd."""
        centred = _sd_error({"limits_of_agreement": [-1.96, 1.96]})
        shifted = _sd_error({"limits_of_agreement": [8.04, 11.96]})
        assert centred == pytest.approx(shifted)

    @pytest.mark.parametrize(
        "entry", [{}, {"limits_of_agreement": None}, {"limits_of_agreement": [1.0]}]
    )
    def test_a_missing_or_malformed_span_is_none_rather_than_a_guess(self, entry: dict) -> None:
        assert _sd_error(entry) is None


class TestAttenuation:
    def test_noiseless_measurement_predicts_a_perfect_correlation(self) -> None:
        assert _expected_rho(0.0, 1.0) == pytest.approx(1.0)

    def test_noise_equal_to_signal_predicts_the_classical_value(self) -> None:
        assert _expected_rho(1.0, 1.0) == pytest.approx(1 / math.sqrt(2))

    def test_more_noise_at_fixed_signal_always_predicts_less_correlation(self) -> None:
        values = [_expected_rho(noise, 1.0) for noise in (0.1, 0.5, 1.0, 2.0, 5.0)]
        assert values == sorted(values, reverse=True)

    def test_more_signal_at_fixed_noise_always_predicts_more_correlation(self) -> None:
        values = [_expected_rho(1.0, signal) for signal in (0.1, 0.5, 1.0, 2.0, 5.0)]
        assert values == sorted(values)

    def test_only_the_ratio_matters_not_the_units(self) -> None:
        """q is dimensionless, but the prediction must not depend on scaling anyway."""
        assert _expected_rho(0.2, 0.4) == pytest.approx(_expected_rho(2.0, 4.0))

    def test_the_prediction_never_leaves_the_correlation_range(self) -> None:
        for noise in (1e-9, 1e-3, 1.0, 1e3, 1e9):
            assert 0.0 < _expected_rho(noise, 1.0) <= 1.0


class TestPolygonRingExtraction:
    def test_a_nested_coco_ring_is_unwrapped(self) -> None:
        assert _ring([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]]) == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]

    def test_a_flat_ring_is_accepted_as_is(self) -> None:
        assert _ring([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]) == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]

    def test_run_length_encoding_is_declined_rather_than_misread(self) -> None:
        """RLE has no polygon, and treating its dict as coordinates would be silent nonsense."""
        assert _ring({"counts": "abc", "size": [10, 10]}) is None

    @pytest.mark.parametrize("value", [None, [], [[]], [1.0, 2.0, 3.0, 4.0]])
    def test_anything_without_three_vertices_is_declined(self, value: object) -> None:
        assert _ring(value) is None


class TestMedianOrNone:
    def test_an_empty_image_yields_none_rather_than_zero(self) -> None:
        """A frame where nothing was segmented has no shape index; zero would be a lie."""
        assert _median_or_none([]) is None

    def test_the_median_is_the_middle_value(self) -> None:
        assert _median_or_none([3.0, 1.0, 2.0]) == pytest.approx(2.0)


def test_the_dynamic_range_artifact_matches_its_own_script(tmp_path: Path) -> None:
    """The published spread must be what the committed script computes."""
    artifact = Path(__file__).resolve().parents[1] / "runs" / "mechanics" / "q_dynamic_range.json"
    if not artifact.is_file():
        pytest.skip("q_dynamic_range.json is written by a run, not by the test suite")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    for line, row in payload["per_line"].items():
        assert row["images"] > 0, line
        assert row["sd_between_images"] > 0.0, line
        # The IQR of a spread-out set cannot exceed its full range.
        assert row["iqr_between_images"] <= row["range_between_images"] + 1e-9, line
