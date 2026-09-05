"""A compact U-Net sized for laptops and small LIVECell subsets.

The first version of this model appeared to fail completely: 0.68 Dice while
training and 0.001 on validation. It had in fact learned. The blocks used
``BatchNorm2d``, and at batch size 2-4 for one epoch the running mean and
variance never approached the batch statistics the network was actually trained
with, so switching to evaluation mode collapsed every prediction to background.
Loading that same checkpoint with batch statistics recovers 0.739 validation
Dice, which is what identified the cause.

``GroupNorm`` removes the failure mode structurally rather than papering over it:
its statistics are computed per sample, so training and evaluation are identical
and the result does not depend on batch size.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    # 8 groups is the usual default; fall back for channel counts below 8.
    groups = 8
    while groups > 1 and channels % groups:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
        )


class TriboUNet(nn.Module):
    """U-Net with ``depth`` pooling stages and batch-size-independent normalisation."""

    def __init__(self, base_channels: int = 32, depth: int = 3, in_channels: int = 1) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError(f"base_channels must be at least 1, got {base_channels}.")
        if not 1 <= depth <= 5:
            raise ValueError(f"depth must be between 1 and 5, got {depth}.")
        self.depth = depth
        self.base_channels = base_channels
        widths = [base_channels * 2**level for level in range(depth)]

        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.encoders.append(DoubleConv(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(2)
        self.bridge = DoubleConv(previous, previous * 2)
        previous *= 2

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(previous, width, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(width * 2, width))
            previous = width
        self.output = nn.Conv2d(previous, 1, kernel_size=1)

    @property
    def size_multiple(self) -> int:
        """Input height and width must both be a multiple of this value."""
        return 2**self.depth

    @property
    def minimum_size(self) -> int:
        """Smallest input side this network can process.

        Divisibility alone is not sufficient. At exactly ``2**depth`` the bridge
        sees a 1x1 feature map, and normalising a single value per group fails
        deep inside ``torch.group_norm`` with a message that mentions neither the
        image size nor this model. One more halving of headroom avoids it.
        """
        return 2 ** (self.depth + 1)

    def check_input_size(self, height: int, width: int) -> None:
        multiple = self.size_multiple
        if min(height, width) < self.minimum_size:
            raise ValueError(
                f"Input {height}x{width} is too small for a depth-{self.depth} U-Net: "
                f"both sides must be at least {self.minimum_size} pixels, otherwise the "
                "bottleneck collapses to a single value per normalisation group. Use "
                f"--depth {max(1, self.depth - 1)} for smaller inputs."
            )
        if height % multiple or width % multiple:
            raise ValueError(
                f"Input {height}x{width} is not compatible with a depth-{self.depth} U-Net. "
                f"Both dimensions must be multiples of {multiple}; the nearest valid size is "
                f"{max(multiple, round(height / multiple) * multiple)}x"
                f"{max(multiple, round(width / multiple) * multiple)}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D (N, C, H, W) tensor, got shape {tuple(x.shape)}.")
        self.check_input_size(x.shape[-2], x.shape[-1])
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bridge(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips), strict=True):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                # Defensive: keeps concatenation valid rather than crashing deep in cuDNN.
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = decoder(torch.cat((x, skip), dim=1))
        return self.output(x)


# Historical name kept so existing checkpoints and imports keep working.
TinyUNet = TriboUNet
