"""Optional third-party segmenters, used as benchmarks rather than as the model.

TriboVision's own U-Net is semantic: it predicts a binary foreground mask, and
`tribovision compare` measures the ceiling that representation imposes. Reaching
past that ceiling needs a model that predicts instances directly, and the honest
way to find out how far short the in-house model falls is to run a strong
off-the-shelf one on the same held-out images.

Cellpose is imported lazily and is not a required dependency. Nothing in the core
pipeline depends on it; if it is absent, the benchmark reports that and the rest
of the project is unaffected.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np

#: Model weights are large and are fetched on first use into ~/.cellpose.
DEFAULT_MODEL = "cpsam"


class ExternalModelError(RuntimeError):
    """Raised when an optional third-party segmenter cannot be used."""


def cellpose_available() -> bool:
    try:
        import cellpose  # noqa: F401
    except ImportError:
        return False
    return True


def cellpose_version() -> str | None:
    try:
        import cellpose
    except ImportError:
        return None
    return str(getattr(cellpose, "version", None) or getattr(cellpose, "__version__", "unknown"))


_MODEL_CACHE: dict[tuple[str, bool], Any] = {}


def _load(model_name: str, gpu: bool) -> Any:
    """Load and cache a Cellpose model; construction dominates per-image cost."""
    key = (model_name, gpu)
    if key not in _MODEL_CACHE:
        try:
            from cellpose import models
        except ImportError as exc:
            raise ExternalModelError(
                "Cellpose is not installed. Install the optional extra with "
                "`pip install -e '.[cellpose]'` to run this benchmark."
            ) from exc
        try:
            # Weight download writes progress bars to stderr; keep reports clean.
            with contextlib.redirect_stderr(io.StringIO()):
                _MODEL_CACHE[key] = models.CellposeModel(gpu=gpu, pretrained_model=model_name)
        except Exception as exc:  # cellpose raises several unrelated types
            raise ExternalModelError(
                f"Could not load Cellpose model {model_name!r}: {exc}"
            ) from exc
    return _MODEL_CACHE[key]


def cellpose_instances(
    image: np.ndarray,
    *,
    diameter: float | None = None,
    model_name: str = DEFAULT_MODEL,
    gpu: bool = True,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
) -> np.ndarray:
    """Segment one greyscale image into an instance label map.

    ``diameter`` is the expected cell diameter in pixels. It is the one knob that
    matters, and it must be chosen on training images — never on the split the
    result is reported from.
    """
    model = _load(model_name, gpu)
    array = np.asarray(image)
    if array.ndim != 2:
        raise ExternalModelError(f"Expected a 2D greyscale image, got shape {array.shape}.")
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            masks, _, _ = model.eval(
                array,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
            )
    except Exception as exc:
        raise ExternalModelError(f"Cellpose inference failed: {exc}") from exc
    return np.asarray(masks, dtype=np.int64)
