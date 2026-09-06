"""The optional Cellpose benchmark, and its absence."""

from __future__ import annotations

import numpy as np
import pytest

from tribovision import external

pytestmark = pytest.mark.filterwarnings("ignore")


def test_availability_is_reported_not_assumed() -> None:
    assert isinstance(external.cellpose_available(), bool)
    version = external.cellpose_version()
    assert version is None or isinstance(version, str)
    assert (version is None) == (not external.cellpose_available())


def test_a_missing_dependency_gives_an_actionable_message(monkeypatch) -> None:
    """Nothing in the core pipeline may depend on this being installed."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def blocked(name, *args, **kwargs):
        if name.startswith("cellpose"):
            raise ImportError("blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    external._MODEL_CACHE.clear()
    assert external.cellpose_available() is False
    with pytest.raises(external.ExternalModelError, match="pip install"):
        external.cellpose_instances(np.zeros((16, 16), dtype=np.uint8))


@pytest.mark.skipif(not external.cellpose_available(), reason="cellpose not installed")
def test_cellpose_returns_an_instance_label_map() -> None:
    ys, xs = np.mgrid[0:96, 0:96]
    image = np.full((96, 96), 30, dtype=np.uint8)
    for cy, cx in ((28, 28), (28, 68), (68, 28), (68, 68)):
        image[((ys - cy) ** 2 + (xs - cx) ** 2) <= 100] = 220
    labels = external.cellpose_instances(image, diameter=20)
    assert labels.shape == (96, 96)
    assert labels.dtype == np.int64
    # Four well-separated discs should come back as separate objects.
    assert labels.max() >= 3


@pytest.mark.skipif(not external.cellpose_available(), reason="cellpose not installed")
def test_a_non_2d_image_is_rejected() -> None:
    with pytest.raises(external.ExternalModelError, match="2D greyscale"):
        external.cellpose_instances(np.zeros((4, 16, 16), dtype=np.uint8))
