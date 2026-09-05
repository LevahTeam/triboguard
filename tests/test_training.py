"""Training configuration, metric aggregation, learning, and determinism."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from conftest import build_dataset

from tribovision.training import (
    ConfigError,
    TrainConfig,
    _run_epoch,
    build_datasets,
    resolve_device,
    train,
)

SMALL = {"image_size": 32, "base_channels": 4, "depth": 2, "batch_size": 2, "device": "cpu"}


def config(tmp_path: Path, data: Path, **overrides: Any) -> TrainConfig:
    return TrainConfig(data_dir=data, output_dir=tmp_path / "run", **{**SMALL, **overrides})


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"epochs": 0}, "epochs"),
        ({"epochs": -3}, "epochs"),
        ({"epochs": 1.5}, "epochs"),
        ({"epochs": True}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": -1}, "batch_size"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": -0.1}, "learning_rate"),
        ({"learning_rate": 5.0}, "learning_rate"),
        ({"learning_rate": "fast"}, "learning_rate"),
        ({"weight_decay": -1.0}, "weight_decay"),
        ({"workers": -1}, "workers"),
        ({"seed": -1}, "seed"),
        ({"patience": -2}, "patience"),
        ({"depth": 0}, "depth"),
        ({"depth": 7}, "depth"),
        ({"base_channels": 0}, "base_channels"),
        ({"image_size": 0}, "image_size"),
        ({"image_size": 17}, "image_size"),
        ({"image_size": -32}, "image_size"),
        ({"image_size": 2}, "image_size"),
        ({"device": "tpu"}, "device"),
        ({"group_by": "plate"}, "group_by"),
    ],
)
def test_invalid_configuration_is_refused_before_anything_is_written(
    tmp_path: Path, tiny_training_data: Path, overrides: dict, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        config(tmp_path, tiny_training_data, **overrides)
    assert not (tmp_path / "run").exists()


def test_a_17_pixel_image_size_is_rejected_with_a_usable_suggestion(
    tmp_path: Path, tiny_training_data: Path
) -> None:
    with pytest.raises(ConfigError, match="Try 16"):
        config(tmp_path, tiny_training_data, image_size=17)


def test_zero_epochs_never_reaches_a_missing_checkpoint(
    tmp_path: Path, tiny_training_data: Path
) -> None:
    """It used to build the run directory, train nothing, then fail on torch.load."""
    with pytest.raises(ConfigError, match="epochs"):
        config(tmp_path, tiny_training_data, epochs=0)


def test_training_refuses_leaking_splits(tmp_path: Path, leaking_training_data: Path) -> None:
    with pytest.raises(ConfigError, match="Split leakage detected"):
        build_datasets(config(tmp_path, leaking_training_data, epochs=1))


def test_clean_splits_build_successfully(tmp_path: Path, tiny_training_data: Path) -> None:
    datasets = build_datasets(config(tmp_path, tiny_training_data, epochs=1))
    assert set(datasets) == {"train", "val", "test"}


class _Loader:
    """A loader whose final batch is deliberately smaller than the rest."""

    def __init__(self, batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


def _batch(n: int, correct: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masks = torch.zeros(n, 1, 8, 8)
    masks[:, :, :4] = 1.0
    valid = torch.ones(n, 1, 8, 8)
    return torch.zeros(n, 1, 8, 8), masks, valid


def test_epoch_metrics_weight_by_sample_not_by_batch() -> None:
    """A 4+1 split must not give the single trailing sample 50% of the weight."""

    class Perfect(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.full_like(x, -20.0)
            logits[:, :, :4] = 20.0
            return logits

    class Wrong(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x, -20.0)

    class Mixed(torch.nn.Module):
        """Right for batches of four, wrong for the trailing single sample."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] == 1:
                return Wrong().forward(x)
            return Perfect().forward(x)

    loader = _Loader([_batch(4, True), _batch(1, False)])
    metrics = _run_epoch(Mixed(), loader, torch.device("cpu"))
    assert metrics["samples"] == 5
    # Per sample: 4 correct (Dice 1) and 1 empty (Dice 0) -> 0.8.
    assert metrics["macro_dice"] == pytest.approx(0.8, abs=1e-6)
    # A per-batch mean would have produced 0.5.
    assert metrics["macro_dice"] != pytest.approx(0.5, abs=1e-6)


def test_an_empty_split_is_reported_clearly() -> None:
    with pytest.raises(ConfigError, match="Dataset split is empty"):
        _run_epoch(torch.nn.Identity(), _Loader([]), torch.device("cpu"))


def test_the_pipeline_can_actually_learn_a_tiny_dataset(
    tmp_path: Path, tiny_training_data: Path
) -> None:
    """End-to-end proof that gradients reach the output, not just that it runs."""
    result = train(
        config(tmp_path, tiny_training_data, epochs=40, augment=False, learning_rate=5e-3),
        progress=False,
    )
    assert result["history"][-1]["train"]["macro_dice"] > 0.9
    assert result["history"][-1]["train"]["loss"] < result["history"][0]["train"]["loss"]


def test_training_writes_portable_reproducible_metadata(
    tmp_path: Path, tiny_training_data: Path
) -> None:
    output = tmp_path / "run"
    result = train(config(tmp_path, tiny_training_data, epochs=2), progress=False)
    persisted = json.loads((output / "metrics.json").read_text())

    assert persisted["checkpoint"] == result["checkpoint"]
    # No absolute paths: an artifact must not leak the author's home directory.
    text = json.dumps(persisted)
    assert "/Users/" not in text and "/home/" not in text
    assert persisted["environment"]["packages"]["torch"]
    assert persisted["environment"]["git"] is not None
    assert persisted["split_check"]["clean"] is True
    assert persisted["datasets"]["train"]["content_sha256"]
    assert persisted["config"]["seed"] == 42
    for section in ("best_validation", "test", "validation"):
        assert 0.0 <= persisted[section]["macro_dice"] <= 1.0
        assert 0.0 <= persisted[section]["micro_iou"] <= 1.0


def test_the_same_seed_reproduces_the_same_numbers(
    tmp_path: Path, tiny_training_data: Path
) -> None:
    first = train(
        TrainConfig(data_dir=tiny_training_data, output_dir=tmp_path / "a", epochs=3, **SMALL),
        progress=False,
    )
    second = train(
        TrainConfig(data_dir=tiny_training_data, output_dir=tmp_path / "b", epochs=3, **SMALL),
        progress=False,
    )
    assert first["test"]["macro_dice"] == pytest.approx(second["test"]["macro_dice"], abs=1e-6)
    assert first["history"][-1]["train"]["loss"] == pytest.approx(
        second["history"][-1]["train"]["loss"], abs=1e-6
    )


def test_a_different_seed_changes_the_trajectory(tmp_path: Path, tiny_training_data: Path) -> None:
    baseline = train(
        TrainConfig(data_dir=tiny_training_data, output_dir=tmp_path / "a", epochs=2, **SMALL),
        progress=False,
    )
    other = train(
        TrainConfig(
            data_dir=tiny_training_data, output_dir=tmp_path / "b", epochs=2, seed=7, **SMALL
        ),
        progress=False,
    )
    assert baseline["history"][0]["train"]["loss"] != other["history"][0]["train"]["loss"]


def test_dataloader_workers_produce_the_same_result_as_the_main_process(
    tmp_path: Path, tiny_training_data: Path
) -> None:
    single = train(
        TrainConfig(data_dir=tiny_training_data, output_dir=tmp_path / "a", epochs=2, **SMALL),
        progress=False,
    )
    workers = train(
        TrainConfig(
            data_dir=tiny_training_data, output_dir=tmp_path / "b", epochs=2, workers=2, **SMALL
        ),
        progress=False,
    )
    assert single["test"]["macro_dice"] == pytest.approx(workers["test"]["macro_dice"], abs=1e-4)


def test_early_stopping_can_end_a_run_before_the_last_epoch(
    tmp_path: Path, tiny_training_data: Path
) -> None:
    result = train(
        config(tmp_path, tiny_training_data, epochs=30, patience=1, learning_rate=1e-6),
        progress=False,
    )
    assert len(result["history"]) < 30


def test_requesting_an_unavailable_accelerator_fails_clearly() -> None:
    if not torch.cuda.is_available():
        with pytest.raises(ConfigError, match="CUDA was requested"):
            resolve_device("cuda")
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_more_wells_than_the_fixture_still_split_cleanly(tmp_path: Path) -> None:
    data = build_dataset(
        tmp_path / "wide", images_per_split=3, wells={"train": "A7", "val": "B7", "test": "C7"}
    )
    datasets = build_datasets(config(tmp_path, data, epochs=1))
    assert len(datasets["train"].records) == 3
