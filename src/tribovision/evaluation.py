"""Segmentation metrics with honest definitions.

Two things were wrong with the metrics this project reported before.

*Smoothing.* Dice was computed as ``(2i + 1) / (s + 1)``. That +1 is a training
loss stabiliser; in a reported result it inflates good scores and, worse, awards
1.0 to an empty prediction on an empty image. Reported metrics here are exact,
with the empty/empty case handled explicitly and stated.

*"Average precision".* The previous ``instance_ap_50_95`` had no confidence
ranking and no precision-recall curve, so it was not average precision. It is
kept under an accurate name — the matching score TP/(TP+FP+FN) averaged over IoU
thresholds 0.50:0.05:0.95, the metric popularised by the 2018 Data Science Bowl
— and real COCO AP is available separately for predictors that emit scores.

Instance matching also used to compare every predicted mask against every true
mask over the whole image, which is O(P x T x H x W). Here the intersection
matrix is built in a single pass over the label images with ``np.bincount``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.05 * step, 2) for step in range(10))


def confusion(
    prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None
) -> dict[str, int]:
    """Pixel counts restricted to *valid* (letterbox padding is never counted)."""
    predicted = np.asarray(prediction).astype(bool)
    actual = np.asarray(truth).astype(bool)
    if predicted.shape != actual.shape:
        raise ValueError(f"Shape mismatch: prediction {predicted.shape} vs truth {actual.shape}.")
    if valid is not None:
        keep = np.asarray(valid).astype(bool)
        if keep.shape != predicted.shape:
            raise ValueError("Valid mask shape does not match the prediction.")
        predicted = predicted & keep
        actual = actual & keep
    true_positive = int(np.count_nonzero(predicted & actual))
    return {
        "tp": true_positive,
        "fp": int(np.count_nonzero(predicted)) - true_positive,
        "fn": int(np.count_nonzero(actual)) - true_positive,
    }


def dice_from_counts(counts: dict[str, int]) -> float:
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    # Both masks empty: the prediction is exactly right, so Dice is defined as 1.
    return 1.0 if denominator == 0 else 2 * counts["tp"] / denominator


def iou_from_counts(counts: dict[str, int]) -> float:
    denominator = counts["tp"] + counts["fp"] + counts["fn"]
    return 1.0 if denominator == 0 else counts["tp"] / denominator


def semantic_metrics(
    prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None
) -> dict[str, float]:
    """Exact, unsmoothed Dice and IoU for one image."""
    counts = confusion(prediction, truth, valid)
    return {"dice": dice_from_counts(counts), "iou": iou_from_counts(counts), **counts}


def intersection_matrix(
    predicted_labels: np.ndarray, true_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (intersections, predicted areas, true areas) for two label images.

    Labels are 1..n with 0 as background. Complexity is O(H x W) rather than
    O(instances^2 x H x W).
    """
    predicted = np.asarray(predicted_labels, dtype=np.int64)
    truth = np.asarray(true_labels, dtype=np.int64)
    if predicted.shape != truth.shape:
        raise ValueError("Label images must have the same shape.")
    n_pred = int(predicted.max())
    n_true = int(truth.max())
    flat = (predicted.ravel() * (n_true + 1)) + truth.ravel()
    table = np.bincount(flat, minlength=(n_pred + 1) * (n_true + 1)).reshape(n_pred + 1, n_true + 1)
    intersections = table[1:, 1:].astype(np.int64)
    predicted_areas = table[1:, :].sum(axis=1).astype(np.int64)
    true_areas = table[:, 1:].sum(axis=0).astype(np.int64)
    return intersections, predicted_areas, true_areas


def instance_iou_matrix(predicted_labels: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    intersections, predicted_areas, true_areas = intersection_matrix(predicted_labels, true_labels)
    if intersections.size == 0:
        return intersections.astype(np.float64)
    unions = predicted_areas[:, None] + true_areas[None, :] - intersections
    with np.errstate(divide="ignore", invalid="ignore"):
        ious = np.where(unions > 0, intersections / np.maximum(unions, 1), 0.0)
    return ious.astype(np.float64)


def matching_score(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    thresholds: tuple[float, ...] = IOU_THRESHOLDS,
) -> dict[str, float]:
    """Mean of TP/(TP+FP+FN) over IoU thresholds — the DSB2018 matching metric.

    This is *not* average precision: the classical baseline assigns no confidence
    to a component, so there is nothing to rank and no precision-recall curve to
    integrate. Reporting it as AP would overstate what was measured.
    """
    n_pred = int(np.asarray(predicted_labels).max())
    n_true = int(np.asarray(true_labels).max())
    if n_pred == 0 and n_true == 0:
        return {"mean": 1.0, **{f"at_{threshold:.2f}": 1.0 for threshold in thresholds}}
    if n_pred == 0 or n_true == 0:
        return {"mean": 0.0, **{f"at_{threshold:.2f}": 0.0 for threshold in thresholds}}
    ious = instance_iou_matrix(predicted_labels, true_labels)
    per_threshold: dict[str, float] = {}
    scores: list[float] = []
    for threshold in thresholds:
        matched = _greedy_match(ious, threshold)
        true_positive = len(matched)
        denominator = true_positive + (n_pred - true_positive) + (n_true - true_positive)
        value = true_positive / denominator if denominator else 1.0
        per_threshold[f"at_{threshold:.2f}"] = float(value)
        scores.append(value)
    return {"mean": float(np.mean(scores)), **per_threshold}


def _greedy_match(ious: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Match instances greedily by descending IoU, one-to-one, above *threshold*."""
    candidates = np.argwhere(ious >= threshold)
    if candidates.size == 0:
        return []
    order = np.argsort(-ious[candidates[:, 0], candidates[:, 1]], kind="stable")
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    matches: list[tuple[int, int]] = []
    for index in order:
        row, column = int(candidates[index, 0]), int(candidates[index, 1])
        if row in used_predictions or column in used_truths:
            continue
        used_predictions.add(row)
        used_truths.add(column)
        matches.append((row, column))
    return matches


def instance_counts(
    predicted_labels: np.ndarray, true_labels: np.ndarray, threshold: float = 0.5
) -> dict[str, int]:
    n_pred = int(np.asarray(predicted_labels).max())
    n_true = int(np.asarray(true_labels).max())
    if n_pred == 0 or n_true == 0:
        return {"tp": 0, "fp": n_pred, "fn": n_true, "predicted": n_pred, "true": n_true}
    matches = _greedy_match(instance_iou_matrix(predicted_labels, true_labels), threshold)
    true_positive = len(matches)
    return {
        "tp": true_positive,
        "fp": n_pred - true_positive,
        "fn": n_true - true_positive,
        "predicted": n_pred,
        "true": n_true,
    }


def average_precision(
    scored_instances: list[tuple[float, np.ndarray]],
    true_labels: np.ndarray,
    thresholds: tuple[float, ...] = IOU_THRESHOLDS,
) -> dict[str, float]:
    """Real COCO-style AP: rank by confidence, integrate the precision-recall curve.

    ``scored_instances`` are ``(confidence, boolean mask)`` pairs. Use this only
    for predictors that genuinely produce a confidence per instance.
    """
    n_true = int(np.asarray(true_labels).max())
    if not scored_instances and n_true == 0:
        return {"mean": 1.0, **{f"at_{t:.2f}": 1.0 for t in thresholds}}
    if not scored_instances or n_true == 0:
        return {"mean": 0.0, **{f"at_{t:.2f}": 0.0 for t in thresholds}}

    ordered = sorted(scored_instances, key=lambda item: -item[0])
    predicted_labels = np.zeros_like(np.asarray(true_labels), dtype=np.int64)
    for index, (_, mask) in enumerate(ordered, start=1):
        predicted_labels[np.asarray(mask).astype(bool)] = index
    ious = instance_iou_matrix(predicted_labels, true_labels)

    results: dict[str, float] = {}
    values: list[float] = []
    for threshold in thresholds:
        matched_truths: set[int] = set()
        hits = np.zeros(len(ordered), dtype=np.float64)
        for row in range(len(ordered)):
            best_column, best_iou = -1, threshold
            for column in range(n_true):
                if column in matched_truths or ious[row, column] < best_iou:
                    continue
                best_column, best_iou = column, ious[row, column]
            if best_column >= 0:
                matched_truths.add(best_column)
                hits[row] = 1.0
        cumulative_tp = np.cumsum(hits)
        cumulative_fp = np.cumsum(1.0 - hits)
        recalls = cumulative_tp / n_true
        precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
        # 101-point interpolated precision, as in the COCO evaluator.
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]
        grid = np.linspace(0.0, 1.0, 101)
        indices = np.searchsorted(recalls, grid, side="left")
        sampled = np.where(
            indices < len(precisions), precisions[np.minimum(indices, len(precisions) - 1)], 0.0
        )
        sampled[indices >= len(precisions)] = 0.0
        value = float(sampled.mean())
        results[f"at_{threshold:.2f}"] = value
        values.append(value)
    return {"mean": float(np.mean(values)), **results}


def aggregate(per_image: list[dict[str, Any]]) -> dict[str, float]:
    """Combine per-image results without letting a short final batch dominate.

    Both views are reported because they answer different questions: ``macro`` is
    the mean over images, ``micro`` pools pixels across the whole split.
    """
    if not per_image:
        return {"images": 0}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for row in per_image:
        for key in totals:
            totals[key] += int(row.get(key, 0))
    return {
        "images": len(per_image),
        "macro_dice": float(np.mean([float(row["dice"]) for row in per_image])),
        "macro_iou": float(np.mean([float(row["iou"]) for row in per_image])),
        "micro_dice": dice_from_counts(totals),
        "micro_iou": iou_from_counts(totals),
        **{f"total_{key}": value for key, value in totals.items()},
    }
