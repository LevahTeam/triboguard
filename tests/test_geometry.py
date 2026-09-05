"""Letterboxing: aspect preservation, padding bookkeeping, and the inverse map."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tribovision.geometry import (
    letterbox_image,
    letterbox_mask,
    normalize_intensity,
    plan_letterbox,
    unletterbox_mask,
)


@pytest.mark.parametrize("size", [(704, 520), (40, 24), (100, 100), (13, 57)])
@pytest.mark.parametrize("target", [64, 256, 512])
def test_aspect_ratio_is_preserved(size: tuple[int, int], target: int) -> None:
    width, height = size
    transform = plan_letterbox(width, height, target)
    assert max(transform.resized_width, transform.resized_height) == target
    # Both axes were scaled by the same factor, up to one pixel of rounding.
    assert abs(transform.resized_width / transform.resized_height - width / height) < 0.02
    assert transform.pad_left >= 0 and transform.pad_top >= 0
    assert transform.pad_right >= 0 and transform.pad_bottom >= 0


def test_a_stretched_resize_would_have_changed_the_shape() -> None:
    """Guards the specific defect: 704x520 squashed into a square distorts area."""
    transform = plan_letterbox(704, 520, 512)
    stretched_aspect = 512 / 512
    assert abs(transform.resized_width / transform.resized_height - 704 / 520) < 0.01
    assert abs(stretched_aspect - 704 / 520) > 0.3


def test_padding_is_marked_invalid_and_excluded_from_the_image() -> None:
    image = Image.new("L", (40, 24), 200)
    boxed, transform = letterbox_image(image, 64)
    valid = transform.valid_mask()
    assert boxed.size == (64, 64)
    assert valid.sum() == transform.resized_width * transform.resized_height
    assert np.asarray(boxed)[valid == 0].max() == 0


def test_mask_round_trip_recovers_the_original_region() -> None:
    mask = np.zeros((24, 40), dtype=np.uint8)
    mask[6:18, 10:30] = 1
    transform = plan_letterbox(40, 24, 128)
    restored = unletterbox_mask(letterbox_mask(mask, transform), transform)
    assert restored.shape == mask.shape
    intersection = np.logical_and(restored, mask).sum()
    union = np.logical_or(restored, mask).sum()
    assert intersection / union > 0.95


def test_unletterbox_rejects_a_wrong_sized_mask() -> None:
    transform = plan_letterbox(40, 24, 64)
    with pytest.raises(ValueError, match="Expected a 64x64 mask"):
        unletterbox_mask(np.zeros((32, 32), dtype=np.uint8), transform)


@pytest.mark.parametrize("width,height,target", [(0, 5, 32), (5, 0, 32), (5, 5, 0)])
def test_invalid_geometry_is_rejected(width: int, height: int, target: int) -> None:
    with pytest.raises(ValueError):
        plan_letterbox(width, height, target)


def test_intensity_normalisation_is_zero_mean_unit_variance() -> None:
    values = np.random.default_rng(0).normal(120, 30, size=1000).astype(np.float32)
    normalized = normalize_intensity(values)
    assert abs(float(normalized.mean())) < 1e-5
    assert abs(float(normalized.std()) - 1.0) < 1e-5


def test_constant_image_normalises_to_zero_rather_than_dividing_by_zero() -> None:
    assert np.all(normalize_intensity(np.full((8, 8), 42.0)) == 0.0)
