from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.types import ModelCapabilities, ModelInput, ModelOutput

from .components import (
    FourierOperatorBlock3d,
    ProjectionHead3d,
    get_activation,
    normalized_coordinate_grid,
)
from .spectral import _validate_modes


def _parse_padding(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        parsed = (value, value, value)
    else:
        if len(value) != 3:
            raise ValueError("padding must be an integer or three values for [D, H, W].")
        parsed = tuple(int(item) for item in value)
    if any(item < 0 for item in parsed):
        raise ValueError("padding values must be non-negative.")
    return parsed  # type: ignore[return-value]


@dataclass
class FNO3DConfig(BaseModelConfig):
    input_channels: int = 1
    output_channels: int = 1
    width: int = 32
    num_layers: int = 4
    modes: tuple[int, int, int] = (12, 12, 12)
    projection_channels: int = 128
    activation: str = "gelu"
    append_coordinates: bool = True
    dropout: float = 0.0
    normalization: str = "none"
    padding: tuple[int, int, int] | int = (0, 0, 0)
    representation: str = "GRID"

    def __post_init__(self) -> None:
        self.name = "fno3d"
        self.model_type = "fno3d"
        self.representation = "GRID"
        self.modes = _validate_modes(self.modes)
        self.padding = _parse_padding(self.padding)
        self.activation = self.activation.lower()
        self.normalization = self.normalization.lower()
        for field_name in (
            "input_channels",
            "output_channels",
            "width",
            "num_layers",
            "projection_channels",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if self.activation not in {"gelu", "relu", "silu"}:
            raise ValueError("activation must be gelu, relu, or silu.")
        if self.normalization not in {"none", "batch", "instance"}:
            raise ValueError("normalization must be none, batch, or instance.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class FNO3D(BaseNetwork):
    """Fourier Neural Operator for dense fields on regular 3D grids."""

    config_class = FNO3DConfig

    def __init__(self, config: FNO3DConfig | None = None) -> None:
        super().__init__(config=config or FNO3DConfig())
        lifted_channels = self.config.input_channels + (
            3 if self.config.append_coordinates else 0
        )
        self.lifting = nn.Conv3d(lifted_channels, self.config.width, kernel_size=1)
        self.blocks = nn.ModuleList(
            FourierOperatorBlock3d(
                self.config.width,
                self.config.modes,
                normalization=self.config.normalization,
            )
            for _ in range(self.config.num_layers)
        )
        self.activation = get_activation(self.config.activation)
        self.projection = ProjectionHead3d(
            self.config.width,
            self.config.projection_channels,
            self.config.output_channels,
            self.config.activation,
            self.config.dropout,
        )

    @property
    def required_inputs(self) -> set[str]:
        return {"voxel_fields"}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            representation="GRID",
            requires={"voxel_fields"},
            supports={"boundary_mask", "material_properties"},
            variable_geometry=False,
            variable_node_count=False,
        )

    def validate_inputs(self, inputs: ModelInput) -> bool:
        super().validate_inputs(inputs)
        volume = self._ensure_tensor(inputs.voxel_fields, name="voxel_fields")
        if volume.ndim != 5:
            raise ValueError(
                "FNO3D voxel_fields must have shape [B, C, D, H, W], "
                f"got {tuple(volume.shape)}."
            )
        if volume.shape[1] != self.config.input_channels:
            raise ValueError(
                f"FNO3D expects {self.config.input_channels} input channels, "
                f"got {volume.shape[1]}."
            )
        if not volume.is_floating_point():
            raise TypeError("voxel_fields must use a floating-point dtype.")
        # Validate against the user-supplied grid, not an artificially enlarged
        # padded grid, so incompatible data fails early and predictably.
        self.blocks[0].spectral._validate_spatial_shape(volume.shape[-3:])
        return True

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        parameter = next(self.parameters())
        x = torch.as_tensor(
            inputs.voxel_fields, device=parameter.device, dtype=parameter.dtype
        )
        original_shape = tuple(x.shape[-3:])
        if self.config.append_coordinates:
            x = torch.cat((x, normalized_coordinate_grid(x)), dim=1)
        x = self.lifting(x)

        pad_d, pad_h, pad_w = self.config.padding
        if any(self.config.padding):
            # F.pad order is W, H, D. Padding is one-sided at the upper end.
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index < len(self.blocks) - 1:
                x = self.activation(x)
        if any(self.config.padding):
            depth, height, width = original_shape
            x = x[..., :depth, :height, :width]

        predictions = self.projection(x)
        return ModelOutput(
            predictions=predictions,
            latent=x,
            metadata={
                "spatial_shape": tuple(predictions.shape[-3:]),
                "modes": self.config.modes,
            },
        )
