"""Shared fixtures.

The tiny dataset is deliberately *not* square (40x24) so that any regression back
to aspect-distorting resizing shows up immediately, and it is split by well so
the leakage guards are exercised on every test that loads it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

NATIVE_WIDTH = 40
NATIVE_HEIGHT = 24


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_image_record(
    image_id: int,
    *,
    cell_type: str = "A172",
    well: str = "A1",
    location: int = 1,
    timestamp: str = "01d00h00m",
    crop: int = 1,
    width: int = NATIVE_WIDTH,
    height: int = NATIVE_HEIGHT,
) -> dict[str, Any]:
    return {
        "id": image_id,
        "file_name": f"{cell_type}_Phase_{well}_{location}_{timestamp}_{crop}.tif",
        "width": width,
        "height": height,
    }


def make_annotation(
    annotation_id: int, image_id: int, *, x: int = 4, y: int = 4, size: int = 8
) -> dict[str, Any]:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": [[x, y, x + size, y, x + size, y + size, x, y + size]],
        "area": size * size,
        "bbox": [x, y, size, size],
        "iscrowd": 0,
    }


def _draw(path: Path, boxes: list[tuple[int, int, int]]) -> None:
    pixels = np.full((NATIVE_HEIGHT, NATIVE_WIDTH), 30, dtype=np.uint8)
    for x, y, size in boxes:
        pixels[y : y + size, x : x + size] = 220
    Image.fromarray(pixels).save(path)


@pytest.fixture
def coco_payload() -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = 1
    for cell_type in ("A172", "BV2"):
        for well_number in range(1, 7):
            well = f"A{well_number}" if cell_type == "A172" else f"B{well_number}"
            for crop in (1, 2):
                images.append(
                    make_image_record(image_id, cell_type=cell_type, well=well, crop=crop)
                )
                annotations.append(make_annotation(1000 + image_id, image_id))
                image_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "cell"}],
    }


def build_dataset(
    root: Path,
    *,
    images_per_split: int = 2,
    wells: dict[str, str] | None = None,
) -> Path:
    """Create a complete, provenance-consistent tiny dataset under *root*."""
    wells = wells or {"train": "A1", "val": "B1", "test": "C1"}
    # Distinct fields of view per split, so file names never collide even when the
    # well deliberately does; that is what makes well-level leakage detectable only
    # by grouping, not by comparing names.
    location_offset = {"train": 0, "val": 10, "test": 20}
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "images" / "A172").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)

    annotations: list[dict[str, Any]] = []
    image_id = 0
    for split in ("train", "val", "test"):
        lines: list[str] = []
        for index in range(images_per_split):
            image_id += 1
            location = location_offset[split] + index + 1
            file_name = f"A172_Phase_{wells[split]}_{location}_01d00h00m_1.tif"
            image_path = root / "images" / "A172" / file_name
            _draw(image_path, [(4, 4, 8), (24, 10, 6)])
            for offset, (x, y, size) in enumerate(((4, 4, 8), (24, 10, 6))):
                annotations.append(
                    make_annotation(image_id * 10 + offset, image_id, x=x, y=y, size=size)
                )
            lines.append(
                json.dumps(
                    {
                        "image_id": image_id,
                        "image_path": str(image_path.relative_to(root)),
                        "annotation_path": "annotations/source.json",
                        "annotation_ids": [image_id * 10, image_id * 10 + 1],
                        "width": NATIVE_WIDTH,
                        "height": NATIVE_HEIGHT,
                        "cell_type": "A172",
                        "well": wells[split],
                        "split": split,
                        "official_split": split,
                        "source_file_name": file_name,
                        "image_sha256": sha256_file(image_path),
                        "micrometers_per_pixel": None,
                    },
                    sort_keys=True,
                )
            )
        (root / "manifests" / f"{split}.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    (root / "annotations" / "source.json").write_text(
        json.dumps({"images": [], "annotations": annotations}), encoding="utf-8"
    )
    return root


@pytest.fixture
def tiny_training_data(tmp_path: Path) -> Path:
    return build_dataset(tmp_path / "livecell")


@pytest.fixture
def leaking_training_data(tmp_path: Path) -> Path:
    """Train and validation drawn from the same well — must be rejected."""
    return build_dataset(tmp_path / "leaky", wells={"train": "A1", "val": "A1", "test": "C1"})


def rewrite_manifest(root: Path, split: str, mutate) -> Path:
    path = root / "manifests" / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = mutate(rows)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def build_synthetic_experiment(
    root: Path,
    *,
    days: tuple[str, ...] = ("2026-03-01", "2026-03-08"),
    concentrations: tuple[float, ...] = (0.0, 25.0, 50.0, 100.0, 200.0),
    wells_per_condition: int = 2,
    fields: int = 2,
    effect: bool = True,
    seed: int = 0,
) -> Path:
    """Write a clearly synthetic treatment experiment used only to test the analysis.

    Cells shrink and round as concentration rises, and viability falls, so a
    correct pipeline must detect the trend. With ``effect=False`` the morphology
    is pure noise, so a correct pipeline must *not* claim a trend.
    """
    import csv as _csv

    rng = np.random.default_rng(seed)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    well_index = 0
    for day in days:
        for concentration in concentrations:
            for replicate in range(wells_per_condition):
                well_index += 1
                well = f"{chr(ord('A') + well_index % 8)}{well_index % 12 + 1:02d}"
                strength = concentration / max(max(concentrations), 1.0)
                radius = 9.0 - (4.0 * strength if effect else 0.0) + rng.normal(0, 0.25)
                viability = float(np.clip(1.0 - 0.85 * strength + rng.normal(0, 0.03), 0.0, 1.2))
                for field in range(1, fields + 1):
                    canvas = np.full((96, 96), 25, dtype=np.uint8)
                    for cy, cx in ((28, 28), (28, 68), (68, 28), (68, 68)):
                        ys, xs = np.mgrid[0:96, 0:96]
                        jitter = rng.normal(0, 0.2)
                        canvas[((ys - cy) ** 2 + (xs - cx) ** 2) <= (radius + jitter) ** 2] = 225
                    name = f"{day}_{well}_f{field}.png"
                    Image.fromarray(canvas).save(images_dir / name)
                    rows.append(
                        {
                            "image_path": f"images/{name}",
                            "experiment_day": day,
                            "plate_id": f"plate-{day}",
                            "well_id": well,
                            "field": str(field),
                            "cell_line": "SYNTHETIC-not-real-cells",
                            "treatment": "synthetic extract",
                            "concentration_ug_per_ml": concentration,
                            "exposure_hours": 24,
                            "control_type": "vehicle" if concentration == 0 else "treated",
                            "replicate": str(replicate + 1),
                            "viability_fraction": round(viability, 4),
                            "micrometers_per_pixel": 0.5,
                            "notes": "SYNTHETIC DATA - generated by the test suite",
                        }
                    )
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


CROWDED_WIDTH = 64
CROWDED_HEIGHT = 48


def _crowded_frame(
    rng: np.random.Generator, cells: list[tuple[int, int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """A textured, high-confluence frame: the case where a fixed rule struggles.

    A flat background with bright squares is separable by a single threshold, so
    the classical baseline scores a perfect 1.0 on it and no learned model can
    ever clear the acceptance gate. That made the gate's pass branch untestable.
    These frames add an illumination gradient, background texture, and cells whose
    interiors are close to the background in intensity — so local contrast finds
    edges rather than cells, exactly as it does on real phase-contrast images.
    """
    ys, xs = np.mgrid[0:CROWDED_HEIGHT, 0:CROWDED_WIDTH]
    background = 90 + 40 * (xs / CROWDED_WIDTH) + 12 * np.sin(ys / 3.0)
    background = background + rng.normal(0, 4, background.shape)
    mask = np.zeros((CROWDED_HEIGHT, CROWDED_WIDTH), dtype=np.uint8)
    image = background.copy()
    for cy, cx, radius in cells:
        blob = ((ys - cy) ** 2 + (xs - cx) ** 2) <= radius**2
        rim = (((ys - cy) ** 2 + (xs - cx) ** 2) <= radius**2) & (
            ((ys - cy) ** 2 + (xs - cx) ** 2) >= (radius - 1.5) ** 2
        )
        mask |= blob
        # Interior barely differs from background; only the rim is high contrast.
        image[blob] = background[blob] + 10
        image[rim] = background[rim] + 70
    return np.clip(image, 0, 255).astype(np.uint8), mask


def build_crowded_dataset(root: Path, *, images_per_split: int = 3) -> Path:
    """A dataset where all-foreground beats the classical rule, as on real data."""
    wells = {"train": "A1", "val": "B1", "test": "C1"}
    location_offset = {"train": 0, "val": 10, "test": 20}
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "images" / "A172").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260905)
    annotations: list[dict[str, Any]] = []
    image_id = 0
    for split in ("train", "val", "test"):
        lines: list[str] = []
        for index in range(images_per_split):
            image_id += 1
            cells = [
                (12, 14, 9),
                (12, 40, 10),
                (34, 20, 10),
                (34, 48, 9),
                (24, 32, 8),
            ]
            pixels, mask = _crowded_frame(rng, cells)
            location = location_offset[split] + index + 1
            file_name = f"A172_Phase_{wells[split]}_{location}_01d00h00m_1.tif"
            image_path = root / "images" / "A172" / file_name
            Image.fromarray(pixels).save(image_path)

            # One RLE annotation carrying the exact union mask.
            from pycocotools import mask as mask_utils

            encoded = mask_utils.encode(np.asfortranarray(mask))
            annotation_id = image_id * 10
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": {
                        "size": [CROWDED_HEIGHT, CROWDED_WIDTH],
                        "counts": encoded["counts"].decode("ascii"),
                    },
                    "area": int(mask.sum()),
                    "bbox": [0, 0, CROWDED_WIDTH, CROWDED_HEIGHT],
                    "iscrowd": 0,
                }
            )
            lines.append(
                json.dumps(
                    {
                        "image_id": image_id,
                        "image_path": str(image_path.relative_to(root)),
                        "annotation_path": "annotations/source.json",
                        "annotation_ids": [annotation_id],
                        "width": CROWDED_WIDTH,
                        "height": CROWDED_HEIGHT,
                        "cell_type": "A172",
                        "well": wells[split],
                        "split": split,
                        "official_split": split,
                        "source_file_name": file_name,
                        "image_sha256": sha256_file(image_path),
                        "micrometers_per_pixel": None,
                    },
                    sort_keys=True,
                )
            )
        (root / "manifests" / f"{split}.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    (root / "annotations" / "source.json").write_text(
        json.dumps({"images": [], "annotations": annotations}), encoding="utf-8"
    )
    return root


@pytest.fixture
def crowded_training_data(tmp_path: Path) -> Path:
    return build_crowded_dataset(tmp_path / "crowded")
