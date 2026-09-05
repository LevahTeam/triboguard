"""Per-object measurement, instance separation, and units."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tribovision import morphology
from tribovision.morphology import Calibration


def _disc(radius: float, centre: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    return ((ys - centre[0]) ** 2 + (xs - centre[1]) ** 2) <= radius**2


def test_connected_components_merges_touching_cells() -> None:
    """The known limitation, asserted so the report cannot silently claim otherwise."""
    mask = _disc(11, (30, 22), (60, 60)) | _disc(11, (30, 38), (60, 60))
    assert morphology.connected_components(mask).max() == 1


def test_watershed_separates_touching_cells() -> None:
    mask = _disc(11, (30, 22), (60, 60)) | _disc(11, (30, 38), (60, 60))
    assert morphology.watershed_split(mask).max() == 2


def test_watershed_leaves_well_separated_cells_alone() -> None:
    mask = _disc(6, (10, 10), (60, 60)) | _disc(6, (45, 45), (60, 60))
    assert morphology.watershed_split(mask).max() == 2


def test_empty_mask_produces_no_objects() -> None:
    empty = np.zeros((20, 20), dtype=bool)
    assert morphology.watershed_split(empty).max() == 0
    assert morphology.measure(morphology.connected_components(empty)) == []
    assert morphology.summarise_image([]) == {"objects": 0}


def test_area_matches_the_pixel_count_of_a_known_square() -> None:
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:15, 5:20] = True
    row = morphology.measure(morphology.connected_components(mask))[0]
    assert row["area_pixels"] == 150
    assert row["perimeter_pixels"] == 2 * (10 + 15)
    assert row["centroid_y"] == pytest.approx(9.5)
    assert row["centroid_x"] == pytest.approx(12.0)


def test_aspect_ratio_detects_elongation_and_roundness() -> None:
    elongated = np.zeros((40, 40), dtype=bool)
    elongated[18:22, 5:35] = True
    round_shape = _disc(9, (20, 20), (40, 40))
    long_row = morphology.measure(morphology.connected_components(elongated))[0]
    round_row = morphology.measure(morphology.connected_components(round_shape))[0]
    assert long_row["aspect_ratio"] > 4.0
    assert round_row["aspect_ratio"] == pytest.approx(1.0, abs=0.1)
    assert round_row["circularity"] > long_row["circularity"]


def test_a_digitised_disc_scores_near_the_documented_reference_value() -> None:
    """Crack perimeter biases circularity; the constant tells readers what round is."""
    row = morphology.measure(morphology.connected_components(_disc(14, (20, 20), (40, 40))))[0]
    assert row["circularity"] == pytest.approx(morphology.DIGITISED_DISC_CIRCULARITY, abs=0.05)


def test_solidity_falls_for_a_concave_shape() -> None:
    solid = np.zeros((40, 40), dtype=bool)
    solid[5:25, 5:25] = True
    solid_row = morphology.measure(morphology.connected_components(solid))[0]
    assert solid_row["solidity"] == pytest.approx(1.0, abs=0.02)

    # An L: hull area is the square minus one corner triangle, so 300/350.
    l_shape = solid.copy()
    l_shape[5:15, 15:25] = False
    assert morphology.measure(morphology.connected_components(l_shape))[0][
        "solidity"
    ] == pytest.approx(300 / 350, abs=0.02)

    # A plus sign is far more ragged: its hull is much larger than the shape.
    plus = np.zeros((40, 40), dtype=bool)
    plus[16:24, 4:36] = True
    plus[4:36, 16:24] = True
    assert morphology.measure(morphology.connected_components(plus))[0]["solidity"] < 0.65


def test_intensity_statistics_are_reported_when_an_image_is_given() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    intensity = np.zeros((20, 20), dtype=np.float32)
    intensity[5:10, 5:10] = 200.0
    row = morphology.measure(morphology.connected_components(mask), intensity)[0]
    assert row["mean_intensity"] == pytest.approx(200.0)
    assert row["std_intensity"] == pytest.approx(0.0)


def test_physical_units_appear_only_when_a_calibration_is_supplied() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    labels = morphology.connected_components(mask)
    without = morphology.measure(labels)[0]
    assert "area_um2" not in without

    with_units = morphology.measure(labels, calibration=Calibration(0.5))[0]
    assert with_units["area_um2"] == pytest.approx(100 * 0.25)
    assert with_units["micrometers_per_pixel"] == 0.5


def test_calibration_returns_none_without_a_scale() -> None:
    assert Calibration().area(10) is None
    assert Calibration().length(10) is None


def test_min_area_drops_small_debris_and_renumbers() -> None:
    mask = np.zeros((30, 30), dtype=bool)
    mask[2:12, 2:12] = True
    mask[25, 25] = True
    labels = morphology.label_objects(mask, min_area=10)
    assert labels.max() == 1
    assert sorted(np.unique(labels)) == [0, 1]


def test_every_row_records_how_it_was_produced() -> None:
    mask = _disc(8, (15, 15), (30, 30))
    for method in morphology.INSTANCE_METHODS:
        rows = morphology.measure(morphology.label_objects(mask, method=method), method=method)
        assert all(row["instance_method"] == method for row in rows)


def test_unknown_instance_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        morphology.label_objects(np.zeros((4, 4), dtype=bool), method="magic")


def test_summary_reports_counts_and_dispersion() -> None:
    mask = _disc(6, (10, 10), (60, 60)) | _disc(9, (40, 40), (60, 60))
    rows = morphology.measure(morphology.connected_components(mask))
    summary = morphology.summarise_image(rows)
    assert summary["objects"] == 2
    assert summary["area_pixels_sd"] > 0
    assert summary["total_area_pixels"] == pytest.approx(sum(row["area_pixels"] for row in rows))


def test_circularity_never_exceeds_one() -> None:
    tiny = np.zeros((10, 10), dtype=bool)
    tiny[5, 5] = True
    assert morphology.measure(morphology.connected_components(tiny))[0]["circularity"] <= 1.0
    assert math.isfinite(
        morphology.measure(morphology.connected_components(tiny))[0]["circularity"]
    )
