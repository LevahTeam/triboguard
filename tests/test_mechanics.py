"""Tissue shape index: geometry, the discretisation trap, and interpretation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tribovision import mechanics
from tribovision.mechanics import (
    CIRCLE_SHAPE_INDEX,
    HEXAGON_SHAPE_INDEX,
    JAMMING_THRESHOLD,
    MechanicsError,
    corrected_pixel_shape_index,
    parse_hours,
    polygon_area_perimeter,
    polygon_shape_index,
    shape_index,
    summarise,
)


def _regular_polygon(sides: int, radius: float = 100.0) -> list[float]:
    angles = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    return [
        value
        for x, y in zip(radius * np.cos(angles), radius * np.sin(angles), strict=True)
        for value in (float(x), float(y))
    ]


def test_the_constants_match_closed_form_geometry() -> None:
    """These are physical constants, not tuning parameters."""
    assert pytest.approx(2 * math.sqrt(math.pi)) == CIRCLE_SHAPE_INDEX
    assert pytest.approx(3.5449, abs=1e-4) == CIRCLE_SHAPE_INDEX
    assert pytest.approx(3.7224, abs=1e-4) == HEXAGON_SHAPE_INDEX
    assert shape_index(1.0, 4.0) == 4.0  # unit square
    # The threshold sits between a hexagon and a square, as the theory requires.
    assert HEXAGON_SHAPE_INDEX < JAMMING_THRESHOLD < 4.0


@pytest.mark.parametrize("sides,expected", [(6, HEXAGON_SHAPE_INDEX), (4, 4.0)])
def test_regular_polygons_recover_their_analytic_shape_index(sides: int, expected: float) -> None:
    assert polygon_shape_index([_regular_polygon(sides)]) == pytest.approx(expected, rel=1e-3)


def test_a_many_sided_polygon_approaches_the_circle_from_above() -> None:
    """No shape can go below the circle; the estimator must respect that."""
    values = [polygon_shape_index([_regular_polygon(n)]) for n in (8, 16, 64, 256)]
    assert all(v is not None and v >= CIRCLE_SHAPE_INDEX - 1e-9 for v in values)
    assert values == sorted(values, reverse=True)
    assert values[-1] == pytest.approx(CIRCLE_SHAPE_INDEX, abs=1e-3)


def test_shoelace_area_and_perimeter_of_a_known_rectangle() -> None:
    area, perimeter = polygon_area_perimeter([0, 0, 40, 0, 40, 10, 0, 10])
    assert area == pytest.approx(400.0)
    assert perimeter == pytest.approx(100.0)


@pytest.mark.parametrize("bad", [[0, 0, 1, 1], [0, 0, 1, 1, 2], []])
def test_degenerate_rings_are_rejected(bad: list[float]) -> None:
    with pytest.raises(MechanicsError):
        polygon_area_perimeter(bad)
    assert polygon_shape_index([bad] if bad else []) is None


@pytest.mark.parametrize("area,perimeter", [(0, 4), (-1, 4), (1, 0), (1, -2)])
def test_non_physical_inputs_are_rejected(area: float, perimeter: float) -> None:
    with pytest.raises(MechanicsError):
        shape_index(area, perimeter)


def test_rle_segmentations_are_reported_as_unmeasurable_not_guessed() -> None:
    assert polygon_shape_index({"size": [4, 4], "counts": [5, 2, 9]}) is None
    assert polygon_shape_index(None) is None


def test_the_discretisation_correction_recovers_a_discs_true_shape_index() -> None:
    """The trap: a pixel perimeter says a smooth disc is unjammed. It is not.

    A digitised disc is the least elongated object possible, so its shape index
    must come out at the circle minimum. Measured with a raw pixel-edge perimeter
    it lands well above the q* = 3.81 rigidity transition instead, which would
    report a fluid tissue no matter what the tissue is doing.
    """
    from tribovision.morphology import crack_perimeter

    ys, xs = np.mgrid[0:400, 0:400]
    disc = ((ys - 200) ** 2 + (xs - 200) ** 2) <= 150**2
    area = float(disc.sum())
    crack = float(crack_perimeter(disc))

    naive = shape_index(area, crack)
    corrected = corrected_pixel_shape_index(area, crack)

    assert naive > JAMMING_THRESHOLD, "the trap should be reproducible"
    assert naive == pytest.approx(CIRCLE_SHAPE_INDEX * mechanics.CRACK_PERIMETER_FACTOR, rel=0.02)
    assert corrected == pytest.approx(CIRCLE_SHAPE_INDEX, rel=0.02)
    assert corrected < JAMMING_THRESHOLD


def test_the_correction_leaves_the_ratio_of_two_shapes_alone() -> None:
    """It is a scale factor, so it cannot invent a difference between conditions."""
    from tribovision.morphology import crack_perimeter

    ys, xs = np.mgrid[0:200, 0:200]
    disc = ((ys - 100) ** 2 + (xs - 100) ** 2) <= 60**2
    bar = np.zeros((200, 200), dtype=bool)
    bar[95:105, 20:180] = True
    pairs = []
    for mask in (disc, bar):
        area, crack = float(mask.sum()), float(crack_perimeter(mask))
        pairs.append((shape_index(area, crack), corrected_pixel_shape_index(area, crack)))
    naive_ratio = pairs[1][0] / pairs[0][0]
    corrected_ratio = pairs[1][1] / pairs[0][1]
    assert naive_ratio == pytest.approx(corrected_ratio, rel=1e-9)


@pytest.mark.parametrize(
    "name,hours",
    [
        ("A172_Phase_C7_2_02d12h00m_3.tif", 60.0),
        ("A172_Phase_A7_1_00d00h00m_1.tif", 0.0),
        ("A172_Phase_A7_1_01d06h30m_1.tif", 30.5),
        ("not-a-livecell-name.png", None),
    ],
)
def test_timestamps_are_recovered_from_the_file_name(name: str, hours: float | None) -> None:
    assert parse_hours(name) == hours


def test_summary_reports_the_state_and_a_bootstrap_interval() -> None:
    jammed = summarise([3.6, 3.65, 3.7, 3.72, 3.75], resamples=200)
    assert jammed["state"].startswith("jammed")
    assert jammed["fraction_unjammed"] == 0.0
    assert jammed["median_q_ci95"] is not None

    fluid = summarise([4.2, 4.5, 4.8, 5.1, 5.4], resamples=200)
    assert fluid["state"].startswith("unjammed")
    assert fluid["fraction_unjammed"] == 1.0
    assert summarise([]) == {"cells": 0}


def test_no_measured_cell_may_fall_below_the_physical_minimum() -> None:
    """A shape index under 2*sqrt(pi) is impossible and signals a broken outline."""
    summary = summarise([3.6, 4.0, 4.4])
    assert summary["fraction_below_circle"] == 0.0


def test_the_interpretation_refuses_to_overclaim() -> None:
    text = mechanics.interpret(summarise([4.4, 4.5, 4.6], resamples=100))
    assert "not a measurement of invasion" in text
    assert "3.81" in text
    assert mechanics.interpret({"cells": 0}) == "No cells measured."


def test_measuring_a_small_annotation_file_end_to_end(tmp_path: Path) -> None:
    payload = {
        "images": [
            {
                "id": index,
                "file_name": f"A172_Phase_A1_{index}_00d00h00m_1.tif",
                "width": 704,
                "height": 520,
            }
            for index in range(1, 4)
        ],
        "annotations": [
            {
                "id": index * 100 + cell,
                "image_id": index,
                "segmentation": [_regular_polygon(6, radius=40.0)],
            }
            for index in range(1, 4)
            for cell in range(15)
        ],
    }
    path = tmp_path / "test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = mechanics.measure_annotation_file(path, cell_type="A172")
    assert result["summary"]["cells"] == 45
    # Every cell is a regular hexagon, so the tissue reads as jammed.
    assert result["summary"]["median_q"] == pytest.approx(HEXAGON_SHAPE_INDEX, rel=1e-3)
    assert result["summary"]["state"].startswith("jammed")
    assert len(result["images"]) == 3
    assert all(row["hours"] == 0.0 for row in result["images"])


def test_tiny_objects_are_excluded_because_their_outline_is_all_discretisation(
    tmp_path: Path,
) -> None:
    payload = {
        "images": [
            {"id": 1, "file_name": "A172_Phase_A1_1_00d00h00m_1.tif", "width": 704, "height": 520}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "segmentation": [_regular_polygon(6, radius=40.0)]},
            {"id": 2, "image_id": 1, "segmentation": [_regular_polygon(6, radius=2.0)]},
        ],
    }
    path = tmp_path / "test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = mechanics.measure_annotation_file(path, cell_type="A172", min_area_pixels=50.0)
    assert result["summary"]["cells"] == 1
    assert result["skipped_below_min_area"] == 1


def test_the_density_relationship_states_its_hypothesis_and_can_reject_it() -> None:
    jamming = [{"cells_per_megapixel": d, "median_q": 5.0 - 0.01 * d} for d in range(50, 150, 10)]
    assert mechanics.density_relationship(jamming)["supports_hypothesis"] is True

    opposite = [{"cells_per_megapixel": d, "median_q": 3.9 + 0.01 * d} for d in range(50, 150, 10)]
    result = mechanics.density_relationship(opposite)
    assert result["evaluated"] is True
    assert result["by_cell_count"]["spearman_rho"] > 0
    assert result["supports_hypothesis"] is False

    assert mechanics.density_relationship([])["evaluated"] is False


def test_confluence_is_preferred_over_cell_count_when_available() -> None:
    """Counting cells conflates crowding with cell size; confluence does not."""
    rows = [
        {
            "cells_per_megapixel": 200 - d,
            "confluence": 0.2 + 0.005 * d,
            "median_area_pixels": 1000.0,
            "median_q": 5.0 - 0.02 * d,
        }
        for d in range(0, 100, 10)
    ]
    result = mechanics.density_relationship(rows)
    assert "by_confluence" in result
    assert "controlling_for_cell_size" in result
    # The verdict comes from confluence, not from the confounded count measure.
    assert result["supports_hypothesis"] == (
        result["by_confluence"]["spearman_rho"] < 0 and result["by_confluence"]["p_value"] < 0.05
    )
    assert result["by_cell_count"]["caveat"].startswith("Confounded")


def test_a_correlation_driven_purely_by_cell_size_does_not_survive_the_control() -> None:
    """The confound made concrete: big cells are both rarer and more elongated.

    Shape index here is a deterministic function of cell area alone, and crowding
    is a deterministic function of area too. The raw correlation is therefore
    perfect, and the partial correlation must report that nothing is left.
    """
    rows = []
    for index in range(12):
        area = 500.0 + 100.0 * index
        rows.append(
            {
                "cells_per_megapixel": 1e6 / area,
                "confluence": 0.8 - 0.03 * index,
                "median_area_pixels": area,
                "median_q": 3.9 + 0.05 * index,
            }
        )
    result = mechanics.density_relationship(rows)
    assert result["by_confluence"]["p_value"] < 0.05
    # Once cell size is removed, the residual association is gone.
    assert result["survives_size_control"] is False


# ------------------------------------------------------------- time trajectories


def _timelapse(tmp_path: Path, points: list[tuple[float, float, int]]) -> Path:
    """Build a COCO file whose cells have a prescribed shape index at each hour.

    Each timepoint is realised as `count` regular polygons; more sides means a
    lower shape index, so a trajectory can be planted exactly.
    """
    images, annotations = [], []
    for index, (hours, sides, count) in enumerate(points, start=1):
        days, rest = divmod(int(hours), 24)
        images.append(
            {
                "id": index,
                "file_name": f"A172_Phase_C7_1_{days:02d}d{rest:02d}h00m_1.tif",
                "width": 704,
                "height": 520,
            }
        )
        for cell in range(count):
            annotations.append(
                {
                    "id": index * 1000 + cell,
                    "image_id": index,
                    "segmentation": [_regular_polygon(sides, radius=40.0)],
                }
            )
    path = tmp_path / f"tl{len(points)}.json"
    path.write_text(json.dumps({"images": images, "annotations": annotations}), encoding="utf-8")
    return path


def test_a_trajectory_is_followed_within_a_well(tmp_path: Path) -> None:
    points = [(h, 6, 40) for h in range(0, 72, 6)]
    result = mechanics.time_course([_timelapse(tmp_path, points)], cell_type="A172")
    assert result["wells_measured"] == 1
    well = result["wells"]["C7"]
    assert well["timepoints"] == len(points)
    assert well["hours_range"] == [0.0, 66.0]
    assert well["q_start"] == pytest.approx(HEXAGON_SHAPE_INDEX, rel=1e-3)


def test_crossing_the_rigidity_transition_is_detected(tmp_path: Path) -> None:
    """Start as hexagons (jammed) and end as squares (fluid)."""
    points = [(h, 6, 30) for h in range(0, 36, 6)] + [(h, 4, 30) for h in range(36, 72, 6)]
    well = mechanics.time_course([_timelapse(tmp_path, points)], cell_type="A172")["wells"]["C7"]
    assert well["q_start"] < JAMMING_THRESHOLD < well["q_end"]
    assert well["crossed_transition"] is True
    assert well["q_change"] > 0


def test_a_well_that_never_crowds_is_excluded_rather_than_counted_against_jamming(
    tmp_path: Path,
) -> None:
    """The finding that made the regime split necessary.

    Post-plating spreading raises the shape index while confluence is still low.
    Pooling that with the later crowding phase cancels the jamming signature, and
    counting a sparse well as a counter-example is simply wrong.
    """
    # Twelve small cells: enough to be measured, far too few to crowd the field.
    sparse = [(h, 6 - min(2, h // 24), 12) for h in range(0, 72, 6)]
    result = mechanics.time_course([_timelapse(tmp_path, sparse)], cell_type="A172")
    well = result["wells"]["C7"]
    assert well["crowded_regime"]["reached"] is False
    assert "not evidence against jamming" in well["crowded_regime"]["note"]
    assert well["jams_as_it_crowds"] is False
    assert "never crowd that far" in result["verdict"]


def test_wells_with_too_few_timepoints_are_dropped(tmp_path: Path) -> None:
    result = mechanics.time_course(
        [_timelapse(tmp_path, [(0, 6, 30), (6, 6, 30)])], cell_type="A172"
    )
    assert result["wells_measured"] == 0
    assert result["verdict"] == "no well had enough timepoints"


def test_the_trajectory_note_warns_against_quoting_a_single_well(tmp_path: Path) -> None:
    result = mechanics.time_course(
        [_timelapse(tmp_path, [(h, 6, 30) for h in range(0, 60, 6)])], cell_type="A172"
    )
    assert "not replicated" in result["note"]
    assert result["wells_measured"] == 1
