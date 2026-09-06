"""A laptop-scale instance segmenter, and the representation that makes it possible.

The ceiling analysis in :mod:`tribovision.mechanics` and ``tribovision compare``
established the problem precisely. Predicting a binary foreground mask caps
instance matching at about 0.13-0.17 on these images no matter how good the model
is, because touching cells merge into one region and no post-hoc split recovers
them. A perfect binary mask scores worse than an imperfect instance model.

Changing what the network predicts moves the ceiling. Asking for three classes -
background, cell interior, and the boundary between touching cells - raises the
measured ceiling to about 0.96, because interiors are separated by construction
and each one seeds exactly one cell.

The model is the same 2M-parameter U-Net with a three-channel head. What changes
is the target and the decoder, not the capacity, which is the point being tested:
representation rather than model size.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tribovision import coco, provenance
from tribovision.data import LiveCellDataset
from tribovision.geometry import letterbox_mask
from tribovision.model import TriboUNet

BACKGROUND, INTERIOR, BOUNDARY = 0, 1, 2


def three_class_target(labels: np.ndarray) -> np.ndarray:
    """Split an instance label image into background / interior / boundary.

    A foreground pixel is boundary when any neighbour carries a different label -
    which covers both the outer edge of a cell and, crucially, the seam where two
    cells touch. Interiors are therefore separated even when the cells are not,
    and that separation is what the decoder turns back into instances.
    """
    from scipy import ndimage

    array = np.asarray(labels)
    foreground = array > 0
    highest = ndimage.maximum_filter(array, size=3)
    lowest = ndimage.minimum_filter(array, size=3)
    boundary = foreground & (highest != lowest)
    target = np.zeros(array.shape, dtype=np.uint8)
    target[foreground] = INTERIOR
    target[boundary] = BOUNDARY
    return target


def decode_instances(
    interior_probability: np.ndarray,
    foreground_probability: np.ndarray,
    *,
    interior_threshold: float = 0.5,
    foreground_threshold: float = 0.5,
    min_area: int = 20,
) -> np.ndarray:
    """Seed from confident interiors, then flood out to the cell edge."""
    from scipy import ndimage

    from tribovision.morphology import _flood, relabel

    seeds = np.asarray(interior_probability) >= interior_threshold
    cells = np.asarray(foreground_probability) >= foreground_threshold
    markers, _ = ndimage.label(seeds, structure=np.ones((3, 3), dtype=bool))
    if markers.max() == 0:
        return np.zeros(seeds.shape, dtype=np.int32)
    grown = _flood(markers.astype(np.int32), cells)
    if min_area > 0 and grown.max() > 0:
        areas = np.bincount(grown.ravel())
        drop = np.where(areas < min_area)[0]
        drop = drop[drop > 0]
        if drop.size:
            grown = np.where(np.isin(grown, drop), 0, grown)
    return relabel(grown)


class ThreeClassDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """The same images and geometry as ``LiveCellDataset``, with a 3-class target."""

    def __init__(self, manifest_path: Path, *, image_size: int = 512, **kwargs: Any) -> None:
        self.base = LiveCellDataset(manifest_path, image_size=image_size, **kwargs)
        self.records = self.base.records
        self.image_size = image_size
        self._targets: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.base)

    def _native_target(self, index: int) -> np.ndarray:
        if index in self._targets:
            return self._targets[index]
        record = self.records[index]
        cache = self.base.cache_dir.parent / "three_class"
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"{self.base._cache_key(index)}.npy"
        if path.is_file():
            try:
                target = np.load(path)
            except (OSError, ValueError):
                target = None
        else:
            target = None
        if target is None:
            payload = json.loads(record.annotation_path.read_text(encoding="utf-8"))
            index_by_id = {int(a["id"]): a for a in payload.get("annotations", [])}
            annotations = [index_by_id[i] for i in record.annotation_ids]
            labels = coco.label_image(annotations, record.width, record.height)
            target = three_class_target(labels)
            with contextlib.suppress(OSError):
                # A read-only cache location must not break training.
                np.save(path, target)
        self._targets[index] = target
        return target

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, _, valid = self.base[index]
        transform = self.base.transform_for(index)
        native = self._native_target(index)
        # Nearest-neighbour each class separately so a boundary is never averaged away.
        boxed = np.zeros((self.image_size, self.image_size), dtype=np.int64)
        for value in (INTERIOR, BOUNDARY):
            resized = letterbox_mask((native == value).astype(np.uint8), transform)
            boxed[resized > 0] = value
        return image, torch.from_numpy(boxed), valid


@dataclass(frozen=True)
class InstanceConfig:
    data_dir: Path
    output_dir: Path
    epochs: int = 40
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    image_size: int = 512
    base_channels: int = 32
    depth: int = 3
    seed: int = 42
    device: str = "auto"
    boundary_weight: float = 3.0
    patience: int = 12


def _class_weights(boundary_weight: float, device: torch.device) -> torch.Tensor:
    """Boundaries are a thin minority class and carry all the separating power."""
    return torch.tensor([1.0, 1.0, boundary_weight], dtype=torch.float32, device=device)


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    import math

    from tribovision.training import DivergenceError

    training = optimizer is not None
    model.train(training)
    total, samples = 0.0, 0
    correct = {name: 0 for name in ("interior", "boundary")}
    counts = dict(correct)
    for images, target, valid in loader:
        images, target, valid = images.to(device), target.to(device), valid.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            keep = valid[:, 0] > 0.5
            loss = nn.functional.cross_entropy(logits, target, weight=weights, reduction="none")
            loss = (loss * keep).sum() / keep.sum().clamp_min(1.0)
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise DivergenceError(
                f"Loss became {value}; lower --learning-rate or use --device cpu."
            )
        batch = images.shape[0]
        total += value * batch
        samples += batch
        with torch.no_grad():
            predicted = logits.argmax(dim=1)
            for name, klass in (("interior", INTERIOR), ("boundary", BOUNDARY)):
                actual = (target == klass) & keep
                hit = (predicted == klass) & actual
                correct[name] += int(hit.sum().item())
                counts[name] += int(actual.sum().item())
    if samples == 0:
        raise ValueError("Dataset split is empty.")
    return {
        "loss": total / samples,
        "samples": samples,
        **{
            f"{name}_recall": (correct[name] / counts[name]) if counts[name] else 0.0
            for name in correct
        },
    }


def train_instance_model(config: InstanceConfig, *, progress: bool = True) -> dict[str, Any]:
    """Train the three-class model. Same network, different question."""
    from tribovision.training import resolve_device, seed_everything

    seed_everything(config.seed)
    device = resolve_device(config.device)
    manifests = Path(config.data_dir).resolve() / "manifests"
    datasets = {
        split: ThreeClassDataset(manifests / f"{split}.jsonl", image_size=config.image_size)
        for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=config.batch_size, shuffle=True, generator=generator
        ),
        **{
            split: DataLoader(datasets[split], batch_size=config.batch_size)
            for split in ("val", "test")
        },
    }
    model = TriboUNet(base_channels=config.base_channels, depth=config.depth, out_channels=3).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    weights = _class_weights(config.boundary_weight, device)

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    best, best_epoch, stale = -1.0, 0, 0
    history: list[dict[str, Any]] = []
    for number in range(1, config.epochs + 1):
        train_metrics = _epoch(model, loaders["train"], device, weights, optimizer)
        with torch.no_grad():
            val_metrics = _epoch(model, loaders["val"], device, weights)
        scheduler.step()
        history.append({"epoch": number, "train": train_metrics, "val": val_metrics})
        # Boundary recall is what separates cells, so it selects the checkpoint.
        score = val_metrics["boundary_recall"]
        if progress:
            print(
                f"epoch {number:>3}/{config.epochs}  loss {val_metrics['loss']:.4f}  "
                f"interior {val_metrics['interior_recall']:.3f}  boundary {score:.3f}",
                flush=True,
            )
        if score > best:
            best, best_epoch, stale = score, number, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": {
                        "class": "TriboUNet",
                        "base_channels": config.base_channels,
                        "depth": config.depth,
                        "out_channels": 3,
                        "normalization": "GroupNorm",
                    },
                    "preprocessing": {"letterbox": True, "image_size": config.image_size},
                    "target": "three_class",
                    "epoch": number,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if config.patience and stale >= config.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    with torch.no_grad():
        test_metrics = _epoch(model, loaders["test"], device, weights)
    result = {
        "config": {
            **{k: v for k, v in vars(config).items()},
            "data_dir": provenance.relative_to_repo(Path(config.data_dir)),
            "output_dir": provenance.relative_to_repo(output_dir),
        },
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation": checkpoint["val_metrics"],
        "test": test_metrics,
        "history": history,
        "checkpoint": provenance.relative_to_repo(checkpoint_path),
        "environment": provenance.environment(),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return result


@torch.no_grad()
def predict_instances(
    model: TriboUNet,
    image: Any,
    *,
    image_size: int,
    device: torch.device,
    interior_threshold: float = 0.5,
    min_area: int = 20,
) -> np.ndarray:
    """Run the three-class model on one image and decode instances natively."""
    from tribovision.geometry import letterbox_image, normalize_intensity, unletterbox_mask

    grey = image.convert("L")
    boxed, transform = letterbox_image(grey, image_size)
    valid = transform.valid_mask()
    pixels = np.asarray(boxed, dtype=np.float32)
    normalized = np.zeros_like(pixels)
    interior_pixels = pixels[valid > 0]
    if interior_pixels.size:
        normalized[valid > 0] = normalize_intensity(interior_pixels)
    tensor = torch.from_numpy(normalized[None, None, ...]).to(device)
    probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy() * valid

    # Decode on the padded square, then map the labels back to the native grid.
    labels = decode_instances(
        probabilities[INTERIOR],
        probabilities[INTERIOR] + probabilities[BOUNDARY],
        interior_threshold=interior_threshold,
        min_area=min_area,
    )
    native = np.zeros((transform.original_height, transform.original_width), dtype=np.int32)
    for index in range(1, int(labels.max()) + 1):
        restored = unletterbox_mask((labels == index).astype(np.uint8), transform)
        native[restored > 0] = index
    from tribovision.morphology import relabel

    return relabel(native)


def load_instance_checkpoint(path: Any, device: torch.device) -> TriboUNet:
    """Load a three-class checkpoint, refusing a binary one."""
    from tribovision.predict import PredictionError

    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("target") != "three_class":
        raise PredictionError(
            f"{Path(path).name} is not a three-class checkpoint; it was trained for "
            f"{payload.get('target', 'semantic segmentation')}."
        )
    architecture = payload.get("architecture") or {}
    model = TriboUNet(
        base_channels=int(architecture.get("base_channels", 32)),
        depth=int(architecture.get("depth", 3)),
        out_channels=3,
    )
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model
