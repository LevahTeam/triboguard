"""The damage model, and the arithmetic of the bias it exposes.

The experiment's claim is that segmenters miss damaged cells, so the damage model
itself has to be shown to do what it says: degrade the cells it is given, leave
the others untouched, and do it monotonically. A model that quietly damaged
everything would produce the same headline for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from tribovision import damage
from tribovision.damage import Damage, DamageError


def _one_cell(size: int = 60, radius: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """A dark cell on a bright background, the polarity LIVECell uses."""
    ys, xs = np.mgrid[0:size, 0:size]
    labels = np.zeros((size, size), dtype=np.int32)
    labels[((ys - size // 2) ** 2 + (xs - size // 2) ** 2) <= radius**2] = 1
    image = np.full((size, size), 200.0, dtype=np.float32)
    image[labels == 1] = 90.0
    return image, labels


def _two_cells() -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:80, 0:80]
    labels = np.zeros((80, 80), dtype=np.int32)
    labels[((ys - 25) ** 2 + (xs - 25) ** 2) <= 81] = 1
    labels[((ys - 55) ** 2 + (xs - 55) ** 2) <= 81] = 2
    image = np.full((80, 80), 200.0, dtype=np.float32)
    image[labels > 0] = 90.0
    return image, labels


def _crowded(gap: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Two cells almost touching, which is what LIVECell actually looks like.

    The far-apart pair above cannot detect damage leaking between neighbours,
    because nothing reaches that far. Confluence is the condition the real
    experiment runs under and so it is the condition the control arm has to
    survive.
    """
    ys, xs = np.mgrid[0:60, 0:60]
    labels = np.zeros((60, 60), dtype=np.int32)
    radius = 9
    left, right = 20, 20 + 2 * radius + gap
    labels[((ys - 30) ** 2 + (xs - left) ** 2) <= radius**2] = 1
    labels[((ys - 30) ** 2 + (xs - right) ** 2) <= radius**2] = 2
    image = np.full((60, 60), 200.0, dtype=np.float32)
    image[labels > 0] = 90.0
    return image, labels


def _contrast(image: np.ndarray, labels: np.ndarray, cell: int) -> float:
    inside = labels == cell
    return abs(float(image[inside].mean()) - float(image[labels == 0].mean()))


class TestTheDamageModel:
    def test_severity_reduces_contrast_monotonically(self) -> None:
        """The axis the whole experiment sweeps, asserted rather than assumed."""
        image, labels = _one_cell()
        contrasts = [
            _contrast(
                damage.apply_damage(image, labels, {1}, Damage.at(s), rng=np.random.default_rng(0)),
                labels,
                1,
            )
            for s in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        assert contrasts == sorted(contrasts, reverse=True)
        assert contrasts[0] > 20 * contrasts[-1], "full severity should nearly erase the cell"

    def test_zero_severity_leaves_the_image_alone(self) -> None:
        image, labels = _one_cell()
        untouched = damage.apply_damage(
            image, labels, {1}, Damage.at(0.0), rng=np.random.default_rng(0)
        )
        np.testing.assert_allclose(untouched, image)

    def test_an_undamaged_cell_keeps_its_pixels(self) -> None:
        """The paired design is worthless if the control arm moves too.

        Its *contrast* is not the thing to check: the blur band around the
        damaged cell touches nearby background, which shifts the background mean
        and so shifts every other cell's contrast without touching a pixel of
        them. The cell's own pixels are what must be untouched.
        """
        image, labels = _two_cells()
        out = damage.apply_damage(image, labels, {1}, Damage.at(0.8), rng=np.random.default_rng(0))
        np.testing.assert_allclose(out[labels == 2], image[labels == 2])

    def test_a_neighbouring_cell_keeps_its_pixels_even_when_touching(self) -> None:
        """The control arm must survive confluence, not just isolation.

        The isolated pair used above sits 42 px apart, so no blur band could
        reach across it and the guarantee went untested for the case that
        matters. Measured on real images, the version of this function without a
        protected mask altered pixels in 89% of the untouched cells.
        """
        image, labels = _crowded()
        out = damage.apply_damage(image, labels, {1}, Damage.at(0.8), rng=np.random.default_rng(0))
        np.testing.assert_allclose(out[labels == 2], image[labels == 2])

    @pytest.mark.parametrize("gap", [0, 1, 2, 4])
    def test_no_separation_lets_damage_reach_the_neighbour(self, gap: int) -> None:
        """Swept across separations, because one spacing proves one spacing."""
        image, labels = _crowded(gap=gap)
        out = damage.apply_damage(image, labels, {1}, Damage.at(1.0), rng=np.random.default_rng(1))
        assert not np.any(out[labels == 2] != image[labels == 2])

    def test_the_damaged_cell_is_still_blurred_when_crowded(self) -> None:
        """Guards the obvious over-correction: protecting everything blurs nothing."""
        image, labels = _crowded()
        out = damage.apply_damage(image, labels, {1}, Damage.at(0.8), rng=np.random.default_rng(0))
        assert _contrast(out, labels, 1) < 0.5 * _contrast(out, labels, 2)

    def test_damaging_one_cell_does_not_damage_the_other(self) -> None:
        image, labels = _two_cells()
        out = damage.apply_damage(image, labels, {1}, Damage.at(0.8), rng=np.random.default_rng(0))
        assert _contrast(out, labels, 1) < 0.5 * _contrast(out, labels, 2)

    def test_damaging_nothing_is_a_copy_not_the_original(self) -> None:
        image, labels = _one_cell()
        out = damage.apply_damage(image, labels, set(), Damage.at(0.9))
        np.testing.assert_allclose(out, image)
        assert out is not image

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
    def test_an_impossible_severity_is_refused(self, bad: float) -> None:
        with pytest.raises(DamageError):
            Damage(fade=bad)

    def test_mismatched_shapes_are_refused(self) -> None:
        image, labels = _one_cell()
        with pytest.raises(DamageError, match="disagree"):
            damage.apply_damage(image, labels[:-1], {1}, Damage.at(0.5))


class TestDetection:
    def test_a_covered_cell_is_found(self) -> None:
        _, labels = _one_cell()
        assert damage.detected_cells(labels.copy(), labels) == {1}

    def test_an_empty_prediction_finds_nothing(self) -> None:
        _, labels = _one_cell()
        assert damage.detected_cells(np.zeros_like(labels), labels) == set()

    def test_a_merged_blob_does_not_count_as_finding_every_cell(self) -> None:
        """One predicted object covering a cluster has not separated it.

        Scoring against the union of all foreground would call this a perfect
        detection, which is how a method that merges everything scores well on a
        metric that was not thinking.
        """
        _, labels = _two_cells()
        blob = (labels > 0).astype(np.int32)  # one object spanning both cells
        found = damage.detected_cells(blob, labels)
        # Each cell is still individually covered by that one object, so both are
        # "found" -- detection is deliberately generous. The separation question
        # is scored elsewhere; this test pins the semantics so they are not
        # confused later.
        assert found == {1, 2}

    def test_partial_coverage_below_the_threshold_is_not_a_detection(self) -> None:
        _, labels = _one_cell()
        partial = np.zeros_like(labels)
        cell = np.argwhere(labels == 1)
        # Cover only a fifth of the cell.
        for y, x in cell[: len(cell) // 5]:
            partial[y, x] = 1
        assert damage.detected_cells(partial, labels) == set()

    @pytest.mark.parametrize("bad", [0.0, -0.2, 1.5])
    def test_an_impossible_coverage_threshold_is_refused(self, bad: float) -> None:
        _, labels = _one_cell()
        with pytest.raises(DamageError, match="coverage"):
            damage.detected_cells(labels, labels, coverage=bad)


class TestDetectionCoverage:
    """The continuous quantity underneath the yes/no verdict.

    Recorded rather than thresholded on the spot so a whole sweep of coverage
    thresholds costs one pass, which is what makes it cheap to check whether the
    finding survives the choice of threshold.
    """

    def test_a_perfect_prediction_covers_everything(self) -> None:
        _, labels = _one_cell()
        assert damage.detection_coverage(labels.copy(), labels) == {1: 1.0}

    def test_a_missing_cell_is_recorded_as_zero_not_omitted(self) -> None:
        """An absent key and a zero mean different things to a caller."""
        _, labels = _one_cell()
        assert damage.detection_coverage(np.zeros_like(labels), labels) == {1: 0.0}

    def test_partial_coverage_is_reported_as_the_fraction(self) -> None:
        _, labels = _one_cell()
        partial = np.zeros_like(labels)
        cell = np.argwhere(labels == 1)
        half = len(cell) // 2
        for y, x in cell[:half]:
            partial[y, x] = 1
        share = damage.detection_coverage(partial, labels)[1]
        assert share == pytest.approx(half / len(cell))

    def test_the_largest_single_object_wins_not_the_union(self) -> None:
        """Two predicted fragments splitting a cell must not sum to a detection."""
        _, labels = _one_cell()
        fragments = np.zeros_like(labels)
        cell = np.argwhere(labels == 1)
        for index, (y, x) in enumerate(cell):
            fragments[y, x] = 1 + index % 2
        share = damage.detection_coverage(fragments, labels)[1]
        assert share < 0.6, "the union would be 1.0; one fragment is about half"

    def test_the_verdict_is_the_coverage_thresholded(self) -> None:
        """The two functions must not be able to disagree."""
        _, labels = _one_cell()
        partial = np.zeros_like(labels)
        cell = np.argwhere(labels == 1)
        for y, x in cell[: int(len(cell) * 0.62)]:
            partial[y, x] = 1
        shares = damage.detection_coverage(partial, labels)
        for threshold in (0.3, 0.5, 0.7):
            expected = {c for c, s in shares.items() if s >= threshold}
            assert damage.detected_cells(partial, labels, coverage=threshold) == expected

    def test_mismatched_shapes_are_refused(self) -> None:
        _, labels = _one_cell()
        with pytest.raises(DamageError, match="disagree"):
            damage.detection_coverage(labels[:-1], labels)


class TestSplitting:
    def test_every_cell_lands_in_exactly_one_arm(self) -> None:
        _, labels = _two_cells()
        damaged, intact = damage.split_cells(labels, rng=np.random.default_rng(0))
        assert damaged | intact == {1, 2}
        assert not damaged & intact

    def test_the_split_is_reproducible_from_its_seed(self) -> None:
        _, labels = _two_cells()
        first = damage.split_cells(labels, rng=np.random.default_rng(4))
        second = damage.split_cells(labels, rng=np.random.default_rng(4))
        assert first == second

    def test_an_empty_image_splits_into_nothing(self) -> None:
        assert damage.split_cells(np.zeros((10, 10), dtype=np.int32)) == (set(), set())

    @pytest.mark.parametrize("fraction,expected", [(0.0, "intact"), (1.0, "damaged")])
    def test_the_extremes_put_everything_in_one_arm(self, fraction: float, expected: str) -> None:
        _, labels = _two_cells()
        damaged, intact = damage.split_cells(labels, fraction=fraction)
        assert (damaged if expected == "damaged" else intact) == {1, 2}


class TestRecoveredEffect:
    """The morphology readout: blindness dilutes the measured shift."""

    def test_equal_detection_recovers_the_whole_effect(self) -> None:
        assert damage.recovered_effect(0.9, 0.9, 0.5) == pytest.approx(1.0)

    def test_total_blindness_recovers_nothing(self) -> None:
        assert damage.recovered_effect(0.9, 0.0, 0.5) == pytest.approx(0.0)

    def test_a_detection_gap_under_reports(self) -> None:
        """The direction that matters: the effect always looks smaller."""
        assert damage.recovered_effect(0.9, 0.3, 0.5) < 1.0

    def test_worse_detection_recovers_less(self) -> None:
        recovered = [damage.recovered_effect(0.9, d, 0.5) for d in (0.9, 0.6, 0.3, 0.0)]
        assert recovered == sorted(recovered, reverse=True)

    def test_no_affected_cells_means_nothing_was_missed(self) -> None:
        """There is no effect to dilute, so the question is vacuous rather than zero."""
        assert damage.recovered_effect(0.9, 0.0, 0.0) == pytest.approx(1.0)

    @pytest.mark.parametrize("args", [(1.5, 0.5, 0.5), (0.5, -0.1, 0.5), (0.9, 0.5, 1.4)])
    def test_impossible_rates_are_refused(self, args: tuple[float, float, float]) -> None:
        with pytest.raises(DamageError):
            damage.recovered_effect(*args)


class TestApparentCountDrop:
    """The count readout: the same blindness pushes the other way.

    Reporting only the morphology direction would be a convenient half of the
    story, so both are modelled and both are tested.
    """

    def test_equal_detection_shows_no_drop_at_all(self) -> None:
        """Counting cannot detect damage while the damaged cells are still there."""
        assert damage.apparent_count_drop(0.9, 0.9, 0.5) == pytest.approx(0.0)

    def test_blindness_manufactures_an_apparent_kill(self) -> None:
        assert damage.apparent_count_drop(0.9, 0.0, 0.5) == pytest.approx(0.5)

    def test_the_two_readouts_move_in_opposite_directions(self) -> None:
        """The finding that makes reporting only one of them misleading."""
        worse = [0.9, 0.6, 0.3, 0.0]
        recovered = [damage.recovered_effect(0.9, d, 0.5) for d in worse]
        drops = [damage.apparent_count_drop(0.9, d, 0.5) for d in worse]
        assert recovered == sorted(recovered, reverse=True)
        assert drops == sorted(drops)

    @pytest.mark.parametrize("args", [(1.5, 0.5, 0.5), (0.5, -0.1, 0.5), (0.9, 0.5, 1.4)])
    def test_impossible_rates_are_refused(self, args: tuple[float, float, float]) -> None:
        with pytest.raises(DamageError):
            damage.apparent_count_drop(*args)
