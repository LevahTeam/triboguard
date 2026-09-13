"""The video figures are shown as evidence, so what they select and check is tested.

A figure can mislead without a single wrong pixel: by cropping to the most
dramatic corner, or by drawing a re-run that no longer matches the recorded
result. These tests pin the two rules that prevent that, and the reading of the
artifact the chart is drawn from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_video_figures import (  # noqa: E402
    agrees_with_record,
    detection_curve,
    representative_window,
)


class TestRepresentativeWindow:
    def test_it_finds_the_only_window_holding_both_kinds(self) -> None:
        centres = {1: (5.0, 5.0), 2: (8.0, 8.0), 3: (85.0, 85.0), 4: (90.0, 88.0)}
        y, x = representative_window(centres, {1, 2, 3}, {4}, shape=(100, 100), size=30)
        for cy, cx in (centres[3], centres[4]):
            assert y <= cy < y + 30 and x <= cx < x + 30

    def test_a_window_full_of_one_kind_never_beats_a_window_with_both(self) -> None:
        """The scarcer kind decides, so a crowd of healthy cells alone scores zero."""
        crowd = {cell: (5.0, 5.0 + cell) for cell in range(10)}
        centres = {**crowd, 20: (80.0, 80.0), 21: (82.0, 82.0)}
        y, x = representative_window(centres, set(crowd) | {20}, {21}, shape=(100, 100), size=30)
        assert y <= 80 < 82 < y + 30 and x <= 80 < 82 < x + 30

    def test_an_empty_image_falls_back_to_the_first_window(self) -> None:
        assert representative_window({}, set(), set(), shape=(100, 100), size=30) == (0, 0)


class TestAgreesWithRecord:
    ROW = {"intact@0.5": 146 / 158, "damaged@0.5": 1 / 151}

    def test_the_recorded_counts_agree(self) -> None:
        assert agrees_with_record(146, 158, 1, 151, self.ROW)

    @pytest.mark.parametrize("counts", [(145, 158, 1, 151), (146, 158, 2, 151)])
    def test_one_cell_off_in_either_arm_is_a_disagreement(
        self, counts: tuple[int, int, int, int]
    ) -> None:
        assert not agrees_with_record(*counts, self.ROW)


class TestDetectionCurve:
    SUMMARY = {
        f"X/{s:.1f}/cellpose/cov0.5": {
            "mean_intact_rate": 0.9,
            "mean_damaged_rate": 0.9 - s,
            "images": 20,
        }
        for s in (0.0, 0.4, 0.8)
    }

    def test_it_reads_each_severity_in_order(self) -> None:
        healthy, hurt, images = detection_curve(self.SUMMARY, "X", (0.0, 0.4, 0.8))
        assert healthy == [0.9, 0.9, 0.9]
        assert hurt == pytest.approx([0.9, 0.5, 0.1])
        assert images == [20, 20, 20]

    def test_a_missing_severity_is_an_error_not_a_gap_in_the_line(self) -> None:
        with pytest.raises(KeyError):
            detection_curve(self.SUMMARY, "X", (0.0, 0.2))
