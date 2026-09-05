"""Model shape contracts, size guards, and the BatchNorm regression."""

from __future__ import annotations

import pytest
import torch

from tribovision.model import TriboUNet


@pytest.mark.parametrize("size", [16, 32, 64, 128])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_output_matches_input_resolution(size: int, depth: int) -> None:
    model = TriboUNet(base_channels=4, depth=depth)
    assert tuple(model(torch.zeros((2, 1, size, size))).shape) == (2, 1, size, size)


def test_non_square_inputs_are_supported() -> None:
    model = TriboUNet(base_channels=4, depth=2)
    assert tuple(model(torch.zeros((1, 1, 32, 48))).shape) == (1, 1, 32, 48)


@pytest.mark.parametrize("size", [17, 15, 31, 7])
def test_incompatible_sizes_fail_with_an_actionable_message(size: int) -> None:
    """The 17x17 crash used to happen inside the skip concatenation."""
    model = TriboUNet(base_channels=4, depth=3)
    with pytest.raises(ValueError, match="nearest valid size|smaller than"):
        model(torch.zeros((1, 1, size, size)))


def test_wrong_rank_input_is_rejected() -> None:
    model = TriboUNet(base_channels=4, depth=2)
    with pytest.raises(ValueError, match="4D"):
        model(torch.zeros((1, 32, 32)))


@pytest.mark.parametrize("kwargs", [{"depth": 0}, {"depth": 6}, {"base_channels": 0}])
def test_invalid_architecture_arguments_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        TriboUNet(**kwargs)


def test_train_and_eval_modes_agree_exactly() -> None:
    """Regression guard for the failure that made the first model look dead.

    With BatchNorm at batch size 2-4 the same weights scored 0.739 Dice in train
    mode and 0.001 in eval mode. GroupNorm makes the two identical by
    construction, so any drift here means normalisation was changed back.
    """
    torch.manual_seed(0)
    model = TriboUNet(base_channels=8, depth=2)
    batch = torch.randn(4, 1, 32, 32)
    model.eval()
    with torch.no_grad():
        evaluated = model(batch)
    model.train()
    with torch.no_grad():
        trained = model(batch)
    assert torch.equal(evaluated, trained)


def test_prediction_does_not_depend_on_the_rest_of_the_batch() -> None:
    """A batch-size-dependent normaliser would break single-image inference."""
    torch.manual_seed(0)
    model = TriboUNet(base_channels=8, depth=2).eval()
    single = torch.randn(1, 1, 32, 32)
    padding = torch.randn(3, 1, 32, 32)
    with torch.no_grad():
        alone = model(single)
        together = model(torch.cat([single, padding]))[:1]
    assert torch.allclose(alone, together, atol=1e-6)


def test_size_multiple_reflects_depth() -> None:
    assert TriboUNet(depth=1).size_multiple == 2
    assert TriboUNet(depth=4).size_multiple == 16
