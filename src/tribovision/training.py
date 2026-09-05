"""Training and evaluation for the TriboVision semantic segmenter.

Design decisions that follow directly from the audit:

* Every configuration value is validated before a directory is created or a file
  is written, so an impossible run fails immediately with a readable message
  instead of crashing inside cuDNN or loading a checkpoint that was never saved.
* Training refuses to start when the splits share an acquisition group.
* Metrics are accumulated per sample, not per batch, so an uneven final batch
  cannot skew an epoch. Padding pixels are excluded everywhere.
* ``metrics.json`` records the code revision, dependency versions, seeds, device,
  dataset fingerprints and configuration, with repository-relative paths only.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tribovision import evaluation, provenance
from tribovision.data import LiveCellDataset, dataset_fingerprint
from tribovision.manifest import ManifestError, require_no_group_leakage
from tribovision.model import TriboUNet

VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


class ConfigError(ValueError):
    """Raised when a training configuration cannot produce a meaningful run."""


class DivergenceError(RuntimeError):
    """Raised when the loss stops being a finite number.

    A diverged run does not look broken from the outside: the loop keeps going,
    ``best_model.pt`` still holds whatever was saved before the blow-up, and
    ``metrics.json`` reports a plausible score from that stale checkpoint. That is
    the same "looks trained, is broken" shape as the BatchNorm bug this project
    already had once, so it is made loud instead of silent.
    """


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    output_dir: Path
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-3
    image_size: int = 512
    seed: int = 42
    workers: int = 0
    device: str = "auto"
    base_channels: int = 32
    depth: int = 3
    augment: bool = True
    photometric_augment: bool = True
    weight_decay: float = 1e-4
    patience: int = 0
    group_by: str = "well"
    verify_hashes: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool) or self.epochs < 1:
            raise ConfigError(f"epochs must be an integer >= 1, got {self.epochs!r}.")
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
            or self.batch_size < 1
        ):
            raise ConfigError(f"batch_size must be an integer >= 1, got {self.batch_size!r}.")
        if not isinstance(self.learning_rate, int | float) or isinstance(self.learning_rate, bool):
            raise ConfigError(f"learning_rate must be a number, got {self.learning_rate!r}.")
        if not 0 < float(self.learning_rate) <= 1:
            raise ConfigError(f"learning_rate must be in (0, 1], got {self.learning_rate}.")
        if float(self.weight_decay) < 0:
            raise ConfigError(f"weight_decay must be >= 0, got {self.weight_decay}.")
        if not isinstance(self.workers, int) or isinstance(self.workers, bool) or self.workers < 0:
            raise ConfigError(f"workers must be an integer >= 0, got {self.workers!r}.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ConfigError(f"seed must be a non-negative integer, got {self.seed!r}.")
        if not isinstance(self.patience, int) or self.patience < 0:
            raise ConfigError(f"patience must be an integer >= 0, got {self.patience!r}.")
        if not 1 <= self.depth <= 5:
            raise ConfigError(f"depth must be between 1 and 5, got {self.depth}.")
        if self.base_channels < 1:
            raise ConfigError(f"base_channels must be >= 1, got {self.base_channels}.")
        multiple = 2**self.depth
        minimum = 2 ** (self.depth + 1)
        if (
            not isinstance(self.image_size, int)
            or isinstance(self.image_size, bool)
            or self.image_size < minimum
        ):
            raise ConfigError(
                f"image_size must be an integer >= {minimum} for a depth-{self.depth} "
                f"U-Net, got {self.image_size!r}. Below that the bottleneck collapses to "
                "one value per normalisation group."
            )
        if self.image_size % multiple:
            raise ConfigError(
                f"image_size must be a multiple of {multiple} for a depth-{self.depth} "
                f"U-Net, got {self.image_size}. Try "
                f"{max(multiple, round(self.image_size / multiple) * multiple)}."
            )
        if self.device not in VALID_DEVICES:
            raise ConfigError(f"device must be one of {VALID_DEVICES}, got {self.device!r}.")
        if self.group_by not in ("well", "acquisition"):
            raise ConfigError(f"group_by must be 'well' or 'acquisition', got {self.group_by!r}.")


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def worker_init(worker_id: int) -> None:
    """Give every DataLoader worker a distinct but reproducible seed."""
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigError("CUDA was requested but is not available on this machine.")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise ConfigError("MPS was requested but is not available on this machine.")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denominator = valid.sum().clamp_min(1.0)
    return (losses * valid).sum() / denominator


def masked_soft_dice(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits) * valid
    targets = targets * valid
    dims = tuple(range(1, targets.ndim))
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    return 1 - ((2 * intersection + 1) / (denominator + 1)).mean()


def _augment(
    images: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
    generator: torch.Generator,
    *,
    photometric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Augment each sample independently.

    Two things were wrong with the first version. It drew one transform per
    *batch*, so four images received the same flip and the effective augmentation
    diversity was a quarter of what the code appeared to provide. And it was
    geometry-only, which left the model brittle to exactly the nuisance variables
    that change between microscopes: a measured stress test showed it collapsing
    to labelling every pixel foreground under moderate sensor noise, scoring the
    trivial predictor's Dice while being completely uninformative.

    Geometric transforms are restricted to flips and quarter turns, which cannot
    change a cell's area, perimeter or shape. Photometric transforms perturb
    contrast, offset, noise and blur — the things a different camera changes —
    and are applied to the image only, never to the mask.
    """
    augmented_images = []
    augmented_masks = []
    augmented_valid = []
    for index in range(images.shape[0]):
        image, mask, keep = images[index], masks[index], valid[index]
        if torch.rand((), generator=generator).item() < 0.5:
            image, mask, keep = (t.flip(-1) for t in (image, mask, keep))
        if torch.rand((), generator=generator).item() < 0.5:
            image, mask, keep = (t.flip(-2) for t in (image, mask, keep))
        turns = int(torch.randint(0, 4, (), generator=generator).item())
        if turns:
            image, mask, keep = (torch.rot90(t, turns, (-2, -1)) for t in (image, mask, keep))
        # Flips and quarter turns return non-contiguous views. Arithmetic on two
        # non-contiguous operands returns garbage on Apple's MPS backend — values
        # around 1e34, then NaN one step later — which silently corrupts training
        # on the default device for anyone on Apple silicon. Materialising the
        # views costs a copy of a single image and removes the failure entirely.
        image, mask, keep = (t.contiguous() for t in (image, mask, keep))
        if photometric:
            image = _photometric(image, keep, generator)
        augmented_images.append(image)
        augmented_masks.append(mask)
        augmented_valid.append(keep)
    return (
        torch.stack(augmented_images),
        torch.stack(augmented_masks),
        torch.stack(augmented_valid),
    )


def _photometric(
    image: torch.Tensor, valid: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Perturb contrast, offset, noise and sharpness on one already-standardised image."""
    scale = 0.7 + 0.6 * torch.rand((), generator=generator).item()
    offset = (torch.rand((), generator=generator).item() - 0.5) * 0.6
    image = image * scale + offset
    if torch.rand((), generator=generator).item() < 0.5:
        sigma = 0.05 + 0.25 * torch.rand((), generator=generator).item()
        noise = torch.randn(image.shape, generator=generator) * sigma
        image = image + noise
    if torch.rand((), generator=generator).item() < 0.3:
        # A 3x3 box blur stands in for a softer objective or a defocused frame.
        blurred = torch.nn.functional.avg_pool2d(
            image.unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze(0)
        weight = torch.rand((), generator=generator).item()
        image = (1 - weight) * image + weight * blurred
    # Padding must stay exactly zero, or the model learns to read the border.
    return image * valid


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    generator: torch.Generator | None = None,
    augment: bool = False,
    photometric: bool = True,
) -> dict[str, float]:
    """One pass over *loader*, accumulating per-sample rather than per-batch."""
    is_training = optimizer is not None
    model.train(is_training)
    loss_total = 0.0
    samples = 0
    totals = {"tp": 0, "fp": 0, "fn": 0}
    dice_sum = 0.0
    iou_sum = 0.0

    for images, masks, valid in loader:
        # Augment on the CPU, then transfer. Doing it on an accelerator was not
        # merely slower: flips and quarter turns produce strided views, and Apple's
        # MPS backend returned corrupt data for them — masks arriving as NaN while
        # the logits were still finite. Transforming before the transfer sidesteps
        # every backend view bug and moves the same number of bytes.
        if is_training and augment and generator is not None:
            images, masks, valid = _augment(
                images, masks, valid, generator, photometric=photometric
            )
        # A blocking copy on purpose. ``non_blocking=True`` only helps from pinned
        # host memory, which this loader does not use; from ordinary memory it lets
        # the transfer race the freeing of these temporaries, and the tensor that
        # arrives on the device is garbage. It surfaced as masks full of NaN while
        # the logits computed from the same batch were still finite.
        images = images.to(device)
        masks = masks.to(device)
        valid = valid.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = masked_bce(logits, masks, valid) + masked_soft_dice(logits, masks, valid)
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        batch = images.shape[0]
        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise DivergenceError(
                f"Loss became {loss_value} during "
                f"{'training' if is_training else 'evaluation'}. The run is diverging; "
                "lower --learning-rate, or use --device cpu if this only happens on an "
                "accelerator."
            )
        loss_total += loss_value * batch
        samples += batch
        with torch.no_grad():
            predicted = ((torch.sigmoid(logits.detach()) >= 0.5) & (valid > 0.5)).float()
            truth = (masks >= 0.5).float() * valid
            dims = tuple(range(1, truth.ndim))
            tp = (predicted * truth).sum(dim=dims)
            fp = (predicted * (1 - truth)).sum(dim=dims)
            fn = ((1 - predicted) * truth).sum(dim=dims)
            totals["tp"] += int(tp.sum().item())
            totals["fp"] += int(fp.sum().item())
            totals["fn"] += int(fn.sum().item())
            dice_denominator = 2 * tp + fp + fn
            iou_denominator = tp + fp + fn
            dice_sum += float(
                torch.where(
                    dice_denominator > 0,
                    2 * tp / dice_denominator.clamp_min(1e-9),
                    torch.ones_like(tp),
                )
                .sum()
                .item()
            )
            iou_sum += float(
                torch.where(
                    iou_denominator > 0, tp / iou_denominator.clamp_min(1e-9), torch.ones_like(tp)
                )
                .sum()
                .item()
            )

    if samples == 0:
        raise ConfigError("Dataset split is empty; prepare more images or select more cell types.")
    return {
        "loss": loss_total / samples,
        "macro_dice": dice_sum / samples,
        "macro_iou": iou_sum / samples,
        "micro_dice": evaluation.dice_from_counts(totals),
        "micro_iou": evaluation.iou_from_counts(totals),
        "samples": samples,
        **{f"total_{key}": value for key, value in totals.items()},
    }


def build_datasets(config: TrainConfig) -> dict[str, LiveCellDataset]:
    manifests = Path(config.data_dir).resolve() / "manifests"
    datasets = {
        split: LiveCellDataset(
            manifests / f"{split}.jsonl",
            image_size=config.image_size,
            verify_hashes=config.verify_hashes,
        )
        for split in ("train", "val", "test")
    }
    try:
        require_no_group_leakage(
            {split: dataset.records for split, dataset in datasets.items()},
            group_by=config.group_by,
        )
    except ManifestError as exc:
        raise ConfigError(str(exc)) from exc
    return datasets


def train(config: TrainConfig, *, progress: bool = True) -> dict[str, Any]:
    """Train the segmenter and evaluate the untouched test split."""
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    datasets = build_datasets(config)

    loader_generator = torch.Generator().manual_seed(config.seed)
    augment_generator = torch.Generator().manual_seed(config.seed + 1)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            generator=loader_generator,
            worker_init_fn=worker_init if config.workers else None,
            drop_last=False,
        ),
        **{
            split: DataLoader(
                datasets[split],
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.workers,
                worker_init_fn=worker_init if config.workers else None,
            )
            for split in ("val", "test")
        },
    }

    model = TriboUNet(base_channels=config.base_channels, depth=config.depth).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"

    serialisable_config = {
        **asdict(config),
        "data_dir": provenance.relative_to_repo(Path(config.data_dir)),
        "output_dir": provenance.relative_to_repo(output_dir),
    }
    best_dice = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0

    for epoch_number in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            loaders["train"],
            device,
            optimizer,
            generator=augment_generator,
            augment=config.augment,
            photometric=config.photometric_augment,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(model, loaders["val"], device)
        scheduler.step()
        row = {
            "epoch": epoch_number,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        if progress:
            print(
                f"epoch {epoch_number:>3}/{config.epochs}  "
                f"train_dice {train_metrics['macro_dice']:.4f}  "
                f"val_dice {val_metrics['macro_dice']:.4f}  "
                f"val_loss {val_metrics['loss']:.4f}",
                flush=True,
            )
        if val_metrics["macro_dice"] > best_dice:
            best_dice = val_metrics["macro_dice"]
            best_epoch = epoch_number
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": serialisable_config,
                    "architecture": {
                        "class": "TriboUNet",
                        "base_channels": config.base_channels,
                        "depth": config.depth,
                        "normalization": "GroupNorm",
                    },
                    "preprocessing": {
                        "letterbox": True,
                        "image_size": config.image_size,
                        "intensity": "per-image zero-mean unit-variance over valid pixels",
                    },
                    "epoch": epoch_number,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if config.patience and epochs_without_improvement >= config.patience:
                if progress:
                    print(
                        f"early stop: no validation improvement for {config.patience} epochs",
                        flush=True,
                    )
                break

    if not checkpoint_path.is_file():  # pragma: no cover - guarded by epochs >= 1
        raise ConfigError("Training finished without saving a checkpoint.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    with torch.no_grad():
        test_metrics = _run_epoch(model, loaders["test"], device)
        val_metrics_final = _run_epoch(model, loaders["val"], device)

    result: dict[str, Any] = {
        "config": serialisable_config,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation": checkpoint["val_metrics"],
        "validation": val_metrics_final,
        "test": test_metrics,
        "history": history,
        "checkpoint": provenance.relative_to_repo(checkpoint_path),
        "datasets": {split: dataset_fingerprint(dataset) for split, dataset in datasets.items()},
        "split_check": require_no_group_leakage(
            {split: dataset.records for split, dataset in datasets.items()},
            group_by=config.group_by,
        ),
        "environment": provenance.environment(),
        "metric_definitions": {
            "macro_dice": "mean over images of exact 2TP/(2TP+FP+FN), padding excluded",
            "micro_dice": "pixels pooled across the whole split before the ratio is taken",
            "note": "No epsilon smoothing is applied to reported metrics.",
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def with_overrides(config: TrainConfig, **overrides: Any) -> TrainConfig:
    return replace(config, **overrides)
