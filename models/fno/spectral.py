from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _validate_modes(modes: Sequence[int]) -> tuple[int, int, int]:
    if len(modes) != 3:
        raise ValueError("modes must contain exactly three values for [D, H, W].")
    parsed = tuple(int(mode) for mode in modes)
    if any(mode <= 0 for mode in parsed):
        raise ValueError("Every Fourier mode count must be positive.")
    return parsed  # type: ignore[return-value]


class SpectralConv3d(nn.Module):
    """Truncated 3D Fourier convolution using the four canonical rFFT corners.

    Inputs and outputs are channel-first. Each complex weight has shape
    ``[in_channels, out_channels, modes_d, modes_h, modes_w]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = _validate_modes(modes)
        weight_shape = (self.in_channels, self.out_channels, *self.modes)
        scale = 1.0 / (self.in_channels * self.out_channels)
        # Match the authors' direct complex parameterization and scaling.
        self.weights1 = nn.Parameter(scale * torch.rand(weight_shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(weight_shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(weight_shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(weight_shape, dtype=torch.cfloat))

    @property
    def spectral_weights(self) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter]:
        return self.weights1, self.weights2, self.weights3, self.weights4

    def _validate_spatial_shape(self, spatial_shape: Sequence[int]) -> None:
        depth, height, width = (int(size) for size in spatial_shape)
        modes_d, modes_h, modes_w = self.modes
        if modes_d > depth // 2:
            raise ValueError(
                f"modes[0]={modes_d} exceeds the non-overlapping positive/negative "
                f"frequency capacity D//2={depth // 2} for D={depth}."
            )
        if modes_h > height // 2:
            raise ValueError(
                f"modes[1]={modes_h} exceeds the non-overlapping positive/negative "
                f"frequency capacity H//2={height // 2} for H={height}."
            )
        if modes_w > width // 2 + 1:
            raise ValueError(
                f"modes[2]={modes_w} exceeds the rFFT capacity W//2+1="
                f"{width // 2 + 1} for W={width}."
            )

    @staticmethod
    def _contract(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # [B, Cin, D, H, Wf] x [Cin, Cout, D, H, Wf]
        # -> [B, Cout, D, H, Wf]
        return torch.einsum("bidhw,iodhw->bodhw", x, weight.to(dtype=x.dtype))

    def multiply_modes(self, x_ft: torch.Tensor) -> torch.Tensor:
        """Apply learned weights and return the sparse Fourier output tensor."""
        if x_ft.ndim != 5:
            raise ValueError("x_ft must have shape [B, C, D, H, W_rfft].")
        if x_ft.shape[1] != self.in_channels:
            raise ValueError(
                f"SpectralConv3d expects {self.in_channels} input channels, "
                f"got {x_ft.shape[1]}."
            )
        if not x_ft.is_complex():
            raise TypeError("x_ft must have a complex dtype.")

        modes_d, modes_h, modes_w = self.modes
        # W_rfft alone cannot reveal whether the original W was odd or even;
        # forward validates against the exact physical shape before this call.
        if modes_d > x_ft.shape[-3] // 2 or modes_h > x_ft.shape[-2] // 2:
            raise ValueError("Configured full-axis modes overlap for this Fourier tensor.")
        if modes_w > x_ft.shape[-1]:
            raise ValueError("Configured final-axis modes exceed the rFFT tensor size.")

        out_ft = x_ft.new_zeros(
            (x_ft.shape[0], self.out_channels, *x_ft.shape[-3:])
        )
        regions = (
            (slice(0, modes_d), slice(0, modes_h), self.weights1),
            (slice(-modes_d, None), slice(0, modes_h), self.weights2),
            (slice(0, modes_d), slice(-modes_h, None), self.weights3),
            (slice(-modes_d, None), slice(-modes_h, None), self.weights4),
        )
        for depth_slice, height_slice, weight in regions:
            source = x_ft[
                :, :, depth_slice, height_slice, :modes_w
            ]
            out_ft[
                :, :, depth_slice, height_slice, :modes_w
            ] = self._contract(source, weight)
        return out_ft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("SpectralConv3d input must have shape [B, C, D, H, W].")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"SpectralConv3d expects {self.in_channels} input channels, "
                f"got {x.shape[1]}."
            )
        if not x.is_floating_point():
            raise TypeError("SpectralConv3d input must use a floating-point dtype.")
        spatial_shape = tuple(x.shape[-3:])
        self._validate_spatial_shape(spatial_shape)
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = self.multiply_modes(x_ft)
        # Passing s is required to reconstruct odd final-axis sizes exactly.
        return torch.fft.irfftn(out_ft, s=spatial_shape, dim=(-3, -2, -1))
