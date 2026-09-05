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


@pytest.mark.parametrize(
    "size,suggestion", [(17, "16x16"), (31, "32x32"), (63, "64x64"), (25, "24x24")]
)
def test_incompatible_sizes_fail_with_an_actionable_message(size: int, suggestion: str) -> None:
    """The 17x17 crash used to happen inside the skip concatenation.

    The message must name a size that actually works, so the suggestion is
    asserted rather than accepting any error text.
    """
    model = TriboUNet(base_channels=4, depth=3)
    with pytest.raises(ValueError, match=f"nearest valid size is {suggestion}"):
        model(torch.zeros((1, 1, size, size)))
    # And the suggested size must genuinely work.
    side = int(suggestion.split("x")[0])
    assert model(torch.zeros((1, 1, side, side))).shape[-1] == side


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_the_minimum_size_is_a_pooling_step_above_mere_divisibility(depth: int) -> None:
    """Divisibility is not sufficient, and the failure below it used to be opaque.

    At exactly ``2**depth`` the bottleneck is 1x1, and ``torch.group_norm`` raises
    "Expected more than 1 value per channel" from inside functional.py — a message
    that names neither the image size nor this model.
    """
    model = TriboUNet(base_channels=2, depth=depth)
    assert model.minimum_size == 2 * model.size_multiple
    with pytest.raises(ValueError, match="too small for a depth"):
        model(torch.zeros((1, 1, model.size_multiple, model.size_multiple)))
    side = model.minimum_size
    assert model(torch.zeros((1, 1, side, side))).shape[-2:] == (side, side)


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
    # assert_close with zero tolerance reports the actual drift on failure, which
    # is the one number you want when this guard fires.
    torch.testing.assert_close(evaluated, trained, rtol=0, atol=0)


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
