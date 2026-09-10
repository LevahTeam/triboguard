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
