"""Metric definitions, fast instance matching, and average precision."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tribovision import evaluation


def test_dice_and_iou_are_exact_not_smoothed() -> None:
    prediction = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    truth = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    metrics = evaluation.semantic_metrics(prediction, truth)
    # tp=1, fp=1, fn=0 -> Dice 2/3, IoU 1/2. Smoothing would have given 3/4 and 2/3.
    assert metrics["dice"] == pytest.approx(2 / 3)
    assert metrics["iou"] == pytest.approx(0.5)


def test_two_empty_masks_score_one_and_a_missed_object_scores_zero() -> None:
    empty = np.zeros((4, 4), dtype=np.uint8)
    assert evaluation.semantic_metrics(empty, empty)["dice"] == 1.0
    assert evaluation.semantic_metrics(empty, np.ones((4, 4), dtype=np.uint8))["dice"] == 0.0


def test_padding_is_excluded_from_the_metric() -> None:
    prediction = np.ones((4, 4), dtype=np.uint8)
    truth = np.zeros((4, 4), dtype=np.uint8)
    truth[:2] = 1
    valid = np.zeros((4, 4), dtype=np.uint8)
    valid[:2] = 1
    # Restricted to the valid half the prediction is perfect.
    assert evaluation.semantic_metrics(prediction, truth, valid)["dice"] == 1.0
    assert evaluation.semantic_metrics(prediction, truth)["dice"] < 1.0


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluation.semantic_metrics(np.zeros((2, 2)), np.zeros((3, 3)))


def _brute_force_iou(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    matrix = np.zeros((int(predicted.max()), int(truth.max())))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            a = predicted == row + 1
            b = truth == column + 1
            union = np.logical_or(a, b).sum()
            matrix[row, column] = np.logical_and(a, b).sum() / union if union else 0.0
    return matrix


def test_fast_iou_matrix_equals_the_brute_force_computation() -> None:
    rng = np.random.default_rng(3)
    predicted = rng.integers(0, 5, size=(40, 40)).astype(np.int64)
    truth = rng.integers(0, 4, size=(40, 40)).astype(np.int64)
    assert np.allclose(
        evaluation.instance_iou_matrix(predicted, truth), _brute_force_iou(predicted, truth)
    )


def test_matching_score_is_one_for_identical_labels_and_zero_when_disjoint() -> None:
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[1:4, 1:4] = 1
    labels[6:9, 6:9] = 2
    assert evaluation.matching_score(labels, labels)["mean"] == 1.0
    empty = np.zeros_like(labels)
    assert evaluation.matching_score(labels, empty)["mean"] == 0.0
    assert evaluation.matching_score(empty, empty)["mean"] == 1.0


def test_matching_score_halves_when_one_of_two_objects_is_missed() -> None:
    truth = np.zeros((10, 10), dtype=np.int64)
    truth[1:4, 1:4] = 1
    truth[6:9, 6:9] = 2
    predicted = np.zeros_like(truth)
    predicted[1:4, 1:4] = 1
    assert evaluation.matching_score(predicted, truth)["mean"] == pytest.approx(0.5)


def test_instance_counts_report_matches_and_errors() -> None:
    truth = np.zeros((10, 10), dtype=np.int64)
    truth[1:4, 1:4] = 1
    truth[6:9, 6:9] = 2
    predicted = np.zeros_like(truth)
    predicted[1:4, 1:4] = 1
    counts = evaluation.instance_counts(predicted, truth)
    assert counts == {"tp": 1, "fp": 0, "fn": 1, "predicted": 1, "true": 2}


def test_average_precision_uses_the_confidence_ranking() -> None:
    truth = np.zeros((12, 12), dtype=np.int64)
    truth[1:5, 1:5] = 1
    good = truth == 1
    junk = np.zeros((12, 12), dtype=bool)
    junk[8:11, 8:11] = True
    confident_correct = evaluation.average_precision([(0.9, good), (0.1, junk)], truth)["mean"]
    confident_wrong = evaluation.average_precision([(0.1, good), (0.9, junk)], truth)["mean"]
    assert confident_correct > confident_wrong
    assert evaluation.average_precision([], truth)["mean"] == 0.0


def test_matching_score_is_not_called_average_precision() -> None:
    """The old name overstated what the number was; keep it from coming back."""
    assert not hasattr(evaluation, "instance_average_precision")


def test_aggregate_reports_both_macro_and_micro_views() -> None:
    rows = [
        {"dice": 1.0, "iou": 1.0, "tp": 100, "fp": 0, "fn": 0},
        {"dice": 0.0, "iou": 0.0, "tp": 0, "fp": 1, "fn": 1},
    ]
    summary = evaluation.aggregate(rows)
    assert summary["macro_dice"] == pytest.approx(0.5)
    # Pooling pixels, one tiny bad image barely moves the score.
    assert summary["micro_dice"] > 0.98
    assert summary["images"] == 2
    assert evaluation.aggregate([]) == {"images": 0}


def test_matching_is_inclusive_at_exactly_the_threshold() -> None:
    """Guards `>=` versus `>` at the IoU cut, which no other test can see.

    Two 4x4 squares overlapping in a 2x8 strip give intersection 8, union 24,
    IoU exactly 1/3. Constructed so the boundary is exact in integer pixels.
    """
    truth = np.zeros((12, 12), dtype=np.int64)
    truth[2:6, 2:10] = 1
    predicted = np.zeros((12, 12), dtype=np.int64)
    predicted[4:8, 2:10] = 1
    ious = evaluation.instance_iou_matrix(predicted, truth)
    assert ious[0, 0] == pytest.approx(1 / 3)

    matched_at = evaluation.matching_score(predicted, truth, thresholds=(1 / 3,))
    just_above = evaluation.matching_score(predicted, truth, thresholds=(1 / 3 + 1e-9,))
    assert matched_at["mean"] == 1.0
    assert just_above["mean"] == 0.0


def test_instance_counts_use_the_same_inclusive_boundary() -> None:
    truth = np.zeros((12, 12), dtype=np.int64)
    truth[2:6, 2:10] = 1
    predicted = np.zeros((12, 12), dtype=np.int64)
    predicted[4:8, 2:10] = 1
    assert evaluation.instance_counts(predicted, truth, threshold=1 / 3)["tp"] == 1
    assert evaluation.instance_counts(predicted, truth, threshold=1 / 3 + 1e-9)["tp"] == 0


# --------------------------------------------------------- bootstrap intervals


def test_a_bootstrap_interval_brackets_the_mean() -> None:
    rng = np.random.default_rng(0)
    values = list(rng.normal(0.5, 0.1, 60))
    result = evaluation.bootstrap_interval(values, resamples=500)
    assert result["evaluated"]
    assert result["ci_low"] < result["mean"] < result["ci_high"]
    assert result["resampling_unit"] == "image"


def test_clustered_resampling_gives_a_wider_interval_than_pretending_independence() -> None:
    """The whole reason to cluster: correlated images do not carry n images of information.

    Twelve groups of ten near-identical crops. Within a group the values are
    effectively one observation repeated, so the honest sample size is 12 rather
    than 120 and the interval should be about sqrt(10) times wider. Asserting the
    theoretical ratio rather than an arbitrary factor makes this a real check.
    """
    rng = np.random.default_rng(1)
    per_group = 10
    values: list[float] = []
    groups: list[str] = []
    for group in range(12):
        centre = rng.normal(0.5, 0.15)
        for _ in range(per_group):
            values.append(centre + rng.normal(0, 0.002))
            groups.append(f"g{group}")
    naive = evaluation.bootstrap_interval(values, resamples=1500)
    clustered = evaluation.bootstrap_interval(values, groups=groups, resamples=1500)
    ratio = (clustered["ci_high"] - clustered["ci_low"]) / (naive["ci_high"] - naive["ci_low"])
    assert 0.6 * math.sqrt(per_group) < ratio < 1.6 * math.sqrt(per_group)
    assert clustered["clusters"] == 12
    assert clustered["resampling_unit"] == "acquisition group"


def test_a_bootstrap_interval_declines_on_too_little_data() -> None:
    assert evaluation.bootstrap_interval([])["evaluated"] is False
    assert evaluation.bootstrap_interval([0.1, 0.2])["evaluated"] is False
    assert (
        evaluation.bootstrap_interval([0.1] * 9, groups=["a"] * 5 + ["b"] * 4)["evaluated"] is False
    )


def test_a_paired_interval_detects_a_real_difference_and_ignores_a_shared_one() -> None:
    rng = np.random.default_rng(2)
    shared = rng.normal(0.5, 0.2, 40)
    better = [float(v) + 0.08 for v in shared]
    same = [float(v) for v in shared]

    real = evaluation.paired_bootstrap(better, list(shared), resamples=800)
    assert real["difference"] == pytest.approx(0.08, abs=0.01)
    assert real["excludes_zero"] is True

    none = evaluation.paired_bootstrap(same, list(shared), resamples=800)
    assert none["difference"] == pytest.approx(0.0, abs=1e-9)
    assert none["excludes_zero"] is False


def test_pairing_is_what_makes_a_small_consistent_difference_detectable() -> None:
    """Unpaired, a tiny consistent gap drowns in between-image variance."""
    rng = np.random.default_rng(3)
    base = rng.normal(0.5, 0.3, 50)
    better = [float(v) + 0.02 for v in base]
    paired = evaluation.paired_bootstrap(better, list(base), resamples=1000)
    assert paired["excludes_zero"] is True
    # The same 0.02 gap against the spread of the values themselves is invisible.
    spread = evaluation.bootstrap_interval(list(base), resamples=1000)
    assert (spread["ci_high"] - spread["ci_low"]) > 0.02
