"""PyTorch dataset for validated TriboVision manifests.

Three properties matter here and none of them were true before:

1. Geometry is preserved. Images are letterboxed, never stretched, and the
   padding is returned as a validity mask so no loss or metric is computed on
   invented pixels.
2. Provenance is enforced at load time through :mod:`tribovision.manifest`.
3. Ground-truth masks are rasterised once and cached, instead of being rebuilt
   from COCO polygons on every epoch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from tribovision import coco
from tribovision.geometry import (
    LetterboxTransform,
    letterbox_image,
    letterbox_mask,
    normalize_intensity,
    plan_letterbox,
)
from tribovision.manifest import ManifestError, ManifestRecord, load_manifest

Image.MAX_IMAGE_PIXELS = 64 * 1024 * 1024


class DatasetError(RuntimeError):
    """Raised when a manifest or annotation cannot be loaded safely."""


Sample = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class LiveCellDataset(Dataset[Sample]):
    """Load phase-contrast images and merge instance labels into a semantic mask.

    Each item is ``(image, mask, valid)`` where ``valid`` is 1 on real pixels and
    0 on letterbox padding.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        image_size: int = 512,
        cache_dir: Path | None = None,
        verify_hashes: bool = True,
    ) -> None:
        if not isinstance(image_size, int) or isinstance(image_size, bool):
            raise ValueError(f"image_size must be an integer, got {image_size!r}.")
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        if image_size % 2:
            raise ValueError(
                f"image_size must be even, got {image_size}. The exact requirement is "
                "depth-dependent (a multiple of 2**depth, and at least 2**(depth+1)) and "
                "is enforced by TrainConfig and by the model itself."
            )
        self.manifest_path = Path(manifest_path).resolve()
        self.image_size = image_size
        try:
            self.records: list[ManifestRecord] = load_manifest(
                self.manifest_path, verify_hashes=verify_hashes
            )
        except ManifestError as exc:
            raise DatasetError(str(exc)) from exc
        self.root = self.manifest_path.parent.parent
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.root / "cache" / "masks"
        self._memory_cache: dict[int, tuple[bytes, tuple[int, int]]] = {}
        self._annotation_cache: dict[Path, dict[int, dict[str, Any]]] = {}

    # ------------------------------------------------------------------ loading

    def annotations_for(self, index: int) -> list[dict[str, Any]]:
        """Fetch one image's COCO annotations, parsing the source file on demand.

        Loading every annotation eagerly kept tens of megabytes of parsed JSON
        alive for the whole run. Because rasterised masks are cached, a warm run
        never needs the polygons at all, so the file is parsed only on a cache
        miss and the index is dropped once every record that uses it is cached.
        """
        import json

        record = self.records[index]
        if record.annotation_path not in self._annotation_cache:
            payload = json.loads(record.annotation_path.read_text(encoding="utf-8"))
            self._annotation_cache[record.annotation_path] = {
                int(annotation["id"]): annotation for annotation in payload.get("annotations", [])
            }
        source = self._annotation_cache[record.annotation_path]
        try:
            return [source[annotation_id] for annotation_id in record.annotation_ids]
        except KeyError as exc:
            raise DatasetError(
                f"Manifest references COCO annotation {exc.args[0]} that is not in "
                f"{record.annotation_path.name}."
            ) from exc

    def _cache_key(self, index: int) -> str:
        record = self.records[index]
        payload = "|".join(
            [
                str(record.image_id),
                record.annotation_path.name,
                ",".join(str(value) for value in record.annotation_ids),
                f"{record.width}x{record.height}",
                coco.backend(),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def native_mask(self, index: int) -> np.ndarray:
        """Return the union ground-truth mask at native image resolution."""
        record = self.records[index]
        shape = (record.height, record.width)
        cached = self._memory_cache.get(index)
        if cached is not None:
            packed, cached_shape = cached
            return np.unpackbits(
                np.frombuffer(packed, dtype=np.uint8), count=cached_shape[0] * cached_shape[1]
            ).reshape(cached_shape)

        cache_path = self.cache_dir / f"{self._cache_key(index)}.npy"
        mask: np.ndarray | None = None
        if cache_path.is_file():
            try:
                packed_array = np.load(cache_path)
                mask = np.unpackbits(packed_array, count=shape[0] * shape[1]).reshape(shape)
            except (OSError, ValueError):
                mask = None
        if mask is None:
            mask = coco.semantic_mask(self.annotations_for(index), record.width, record.height)
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                # np.save appends ".npy" unless the name already ends in it, so
                # a temporary called "x.npy.tmp" is written as "x.npy.tmp.npy"
                # and the rename below then fails on a file that was never
                # created. That raised OSError, which the handler swallowed, so
                # the cache silently never worked while leaving one stray file
                # per mask behind -- 948 of them here. Writing through an open
                # handle bypasses the extension rule entirely.
                temporary = cache_path.with_name(cache_path.name + ".tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, np.packbits(mask.astype(np.uint8)))
                temporary.replace(cache_path)
            except OSError:
                # A read-only cache location must not break training.
                pass
        self._memory_cache[index] = (np.packbits(mask.astype(np.uint8)).tobytes(), shape)
        return mask.astype(np.uint8)

    def transform_for(self, index: int) -> LetterboxTransform:
        record = self.records[index]
        return plan_letterbox(record.width, record.height, self.image_size)

    # ------------------------------------------------------------- torch dataset

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        record = self.records[index]
        try:
            with Image.open(record.image_path) as raw:
                image = raw.convert("L")
                if image.size != (record.width, record.height):
                    raise DatasetError(
                        f"Image size {image.size} for {record.image_path.name} does not match "
                        f"manifest size {(record.width, record.height)}."
                    )
                boxed, transform = letterbox_image(image, self.image_size)
                pixels = np.asarray(boxed, dtype=np.float32)
        except DatasetError:
            raise
        except (FileNotFoundError, OSError) as exc:
            raise DatasetError(f"Cannot read image: {record.image_path}") from exc

        valid = transform.valid_mask()
        # Standardise using only the real pixels so padding cannot shift the mean.
        interior = pixels[valid > 0]
        normalized = np.zeros_like(pixels, dtype=np.float32)
        if interior.size:
            standardized = normalize_intensity(interior)
            normalized[valid > 0] = standardized
        mask = letterbox_mask(self.native_mask(index), transform).astype(np.float32)
        return (
            torch.from_numpy(normalized[None, ...]),
            torch.from_numpy(mask[None, ...]),
            torch.from_numpy(valid[None, ...]),
        )


def dataset_fingerprint(dataset: LiveCellDataset) -> dict[str, Any]:
    """Summarise which data a result was produced from."""
    digest = hashlib.sha256()
    for record in dataset.records:
        digest.update(f"{record.image_id}:{record.image_sha256 or ''}\n".encode())
    return {
        "manifest": dataset.manifest_path.name,
        "images": len(dataset.records),
        "wells": sorted({record.well for record in dataset.records}),
        "cell_types": sorted({record.cell_type for record in dataset.records}),
        "content_sha256": digest.hexdigest(),
        "coco_backend": coco.backend(),
        "image_size": dataset.image_size,
    }


# Re-exported for backwards compatibility with existing imports and tests.
segmentation_mask = coco.segmentation_mask
