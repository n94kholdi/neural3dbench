from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


def _activation(name: str) -> nn.Module:
    activations: dict[str, Callable[[], nn.Module]] = {
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(0.01, inplace=True),
        "elu": lambda: nn.ELU(inplace=True),
        "gelu": nn.GELU,
        "silu": lambda: nn.SiLU(inplace=True),
    }
    try:
        return activations[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unsupported activation '{name}'. Choose from {sorted(activations)}."
        ) from exc


def _normalization(name: str, channels: int) -> nn.Module:
    if name == "batch":
        return nn.BatchNorm3d(channels)
    if name == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if name == "group":
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if name in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(
        "Unsupported normalization "
        f"'{name}'. Choose from ['batch', 'group', 'instance', 'none']."
    )


class ConvBlock3D(nn.Module):
    """Two same-resolution 3D convolutions with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        normalization: str = "batch",
        activation: str = "relu",
    ):
        super().__init__()
        padding = kernel_size // 2
        use_bias = normalization in {"none", "identity"}
        self.layers = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=use_bias,
            ),
            _normalization(normalization, out_channels),
            _activation(activation),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=use_bias,
            ),
            _normalization(normalization, out_channels),
            _activation(activation),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class EncoderBlock3D(nn.Module):
    """A feature block followed by optional resolution reduction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        normalization: str = "batch",
        activation: str = "relu",
        downsampling: str | None = "max_pool",
    ):
        super().__init__()
        self.features = ConvBlock3D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            normalization=normalization,
            activation=activation,
        )
        if downsampling == "max_pool":
            self.downsample: nn.Module | None = nn.MaxPool3d(2)
        elif downsampling == "strided_conv":
            self.downsample = nn.Conv3d(out_channels, out_channels, 2, stride=2)
        elif downsampling is None:
            self.downsample = None
        else:
            raise ValueError("downsampling must be 'max_pool' or 'strided_conv'.")

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.features(inputs)
        downsampled = skip if self.downsample is None else self.downsample(skip)
        return skip, downsampled


class DecoderBlock3D(nn.Module):
    """Upsample, join the corresponding encoder feature, and refine it."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        *,
        kernel_size: int = 3,
        normalization: str = "batch",
        activation: str = "relu",
        upsampling: str = "transpose_conv",
    ):
        super().__init__()
        self.upsampling = upsampling
        if upsampling == "transpose_conv":
            self.upsample: nn.Module = nn.ConvTranspose3d(
                in_channels, skip_channels, kernel_size=2, stride=2
            )
        elif upsampling == "interpolate":
            self.upsample = nn.Conv3d(in_channels, skip_channels, kernel_size=1)
        else:
            raise ValueError("upsampling must be 'transpose_conv' or 'interpolate'.")
        self.features = ConvBlock3D(
            skip_channels * 2,
            skip_channels,
            kernel_size=kernel_size,
            normalization=normalization,
            activation=activation,
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.upsampling == "interpolate":
            inputs = F.interpolate(
                inputs, size=skip.shape[-3:], mode="trilinear", align_corners=False
            )
            inputs = self.upsample(inputs)
        else:
            inputs = self.upsample(inputs)
            # Pooling odd-sized volumes can leave a one-voxel discrepancy. Resize
            # decoder features to the skip tensor; the user volume is never resized.
            if inputs.shape[-3:] != skip.shape[-3:]:
                inputs = F.interpolate(
                    inputs, size=skip.shape[-3:], mode="trilinear", align_corners=False
                )
        return self.features(torch.cat((skip, inputs), dim=1))


class VolumetricOutputHead(nn.Module):
    """A task-neutral 1x1 projection returning raw dense predictions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.projection = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)
