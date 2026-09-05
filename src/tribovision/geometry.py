"""Aspect-preserving resize ("letterboxing") and its exact inverse.

LIVECell frames are 704x520. Stretching them into a square changes width and
height by different factors, so every downstream quantity TriboVision intends to
report — area, perimeter, roundness, swelling, rounding — is distorted, and the
distortion is anisotropic, so it cannot be divided out afterwards.

Letterboxing scales both axes by one factor and pads the remainder. The padding
is tracked explicitly so that (a) loss and metrics ignore padded pixels and
(b) predictions can be mapped back to native pixel coordinates before any
morphology measurement is taken.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LetterboxTransform:
    """Records how a native ``(width, height)`` image was fitted into a square."""

    original_width: int
    original_height: int
    target_size: int
    scale: float
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int

    @property
    def pad_right(self) -> int:
        return self.target_size - self.resized_width - self.pad_left

    @property
    def pad_bottom(self) -> int:
        return self.target_size - self.resized_height - self.pad_top

    def valid_mask(self) -> np.ndarray:
        """A float mask that is 1 on real image pixels and 0 on padding."""
        mask = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        mask[
            self.pad_top : self.pad_top + self.resized_height,
            self.pad_left : self.pad_left + self.resized_width,
        ] = 1.0
        return mask

    def as_dict(self) -> dict[str, float | int]:
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "target_size": self.target_size,
            "scale": self.scale,
            "resized_width": self.resized_width,
            "resized_height": self.resized_height,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
        }


def plan_letterbox(width: int, height: int, target_size: int) -> LetterboxTransform:
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}.")
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}.")
    scale = min(target_size / width, target_size / height)
    resized_width = max(1, min(target_size, int(round(width * scale))))
    resized_height = max(1, min(target_size, int(round(height * scale))))
    return LetterboxTransform(
        original_width=width,
        original_height=height,
        target_size=target_size,
        scale=scale,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=(target_size - resized_width) // 2,
        pad_top=(target_size - resized_height) // 2,
    )


def letterbox_image(image: Image.Image, target_size: int) -> tuple[Image.Image, LetterboxTransform]:
    """Fit *image* into a square canvas without changing its aspect ratio."""
    transform = plan_letterbox(image.width, image.height, target_size)
    resized = image.resize(
        (transform.resized_width, transform.resized_height), Image.Resampling.BILINEAR
    )
    canvas = Image.new(image.mode, (target_size, target_size), 0)
    canvas.paste(resized, (transform.pad_left, transform.pad_top))
    return canvas, transform


def letterbox_mask(mask: np.ndarray, transform: LetterboxTransform) -> np.ndarray:
    """Apply the same transform to a binary mask using nearest-neighbour sampling."""
    source = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
    resized = source.resize(
        (transform.resized_width, transform.resized_height), Image.Resampling.NEAREST
    )
    canvas = Image.new("L", (transform.target_size, transform.target_size), 0)
    canvas.paste(resized, (transform.pad_left, transform.pad_top))
    return (np.asarray(canvas) > 0).astype(np.uint8)


def unletterbox_mask(mask: np.ndarray, transform: LetterboxTransform) -> np.ndarray:
    """Map a square prediction back onto the native image grid.

    Morphology must be measured here, not on the padded square, so that areas and
    perimeters are in real pixels of the original acquisition.
    """
    array = np.asarray(mask)
    if array.shape != (transform.target_size, transform.target_size):
        raise ValueError(
            f"Expected a {transform.target_size}x{transform.target_size} mask, got {array.shape}."
        )
    cropped = array[
        transform.pad_top : transform.pad_top + transform.resized_height,
        transform.pad_left : transform.pad_left + transform.resized_width,
    ]
    image = Image.fromarray((cropped > 0).astype(np.uint8) * 255)
    restored = image.resize(
        (transform.original_width, transform.original_height), Image.Resampling.NEAREST
    )
    return (np.asarray(restored) > 0).astype(np.uint8)


def normalize_intensity(array: np.ndarray) -> np.ndarray:
    """Standardise one image to zero mean and unit variance.

    Phase-contrast illumination drifts between wells and imaging days. Dividing by
    255 leaves that drift in the input, so a model trained on two wells sees a
    shifted distribution on a third. Per-image standardisation removes the
    offending nuisance variable without discarding cell/background contrast.
    """
    values = np.asarray(array, dtype=np.float32)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)
