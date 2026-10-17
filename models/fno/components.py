from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from .spectral import SpectralConv3d


def normalized_coordinate_grid(x: torch.Tensor) -> torch.Tensor:
    """Return non-trainable [D, H, W] coordinates in [0, 1]."""
    if x.ndim != 5:
        raise ValueError("Coordinate-grid input must have shape [B, C, D, H, W].")
    axes = [
        torch.linspace(0.0, 1.0, size, device=x.device, dtype=x.dtype)
        for size in x.shape[-3:]
    ]
    depth, height, width = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack((depth, height, width), dim=0)
    return grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    activations: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "gelu": torch.nn.functional.gelu,
        "relu": torch.nn.functional.relu,
        "silu": torch.nn.functional.silu,
    }
    try:
        return activations[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation '{name}'.") from exc


class FourierOperatorBlock3d(nn.Module):
    """A spectral branch plus a pointwise local branch."""

    def __init__(
        self,
        width: int,
        modes: Sequence[int],
        *,
        normalization: str = "none",
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes)
        self.local = nn.Conv3d(width, width, kernel_size=1)
        if normalization == "none":
            self.normalization: nn.Module = nn.Identity()
        elif normalization == "batch":
            self.normalization = nn.BatchNorm3d(width)
        elif normalization == "instance":
            self.normalization = nn.InstanceNorm3d(width, affine=True)
        else:
            raise ValueError("normalization must be none, batch, or instance.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalization(self.spectral(x) + self.local(x))


class ProjectionHead3d(nn.Module):
    def __init__(
        self,
        width: int,
        projection_channels: int,
        output_channels: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden = nn.Conv3d(width, projection_channels, kernel_size=1)
        self.output = nn.Conv3d(projection_channels, output_channels, kernel_size=1)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout3d(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(self.activation(self.hidden(x))))
