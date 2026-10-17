from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.types import ModelCapabilities, ModelInput, ModelOutput

from .layers import SineLayer


@dataclass
class SIRENConfig(BaseModelConfig):
    spatial_dim: int = 3
    input_dim: int = 0
    output_dim: int = 1
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    first_omega_0: float = 30.0
    hidden_omega_0: float = 30.0
    outermost_linear: bool = True
    representation: str = "POINT_CLOUD"

    def __post_init__(self) -> None:
        self.name = "siren"
        self.model_type = "siren"
        self.representation = "POINT_CLOUD"
        for field_name in ("spatial_dim", "output_dim", "hidden_dim"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if self.input_dim < 0:
            raise ValueError("input_dim cannot be negative.")
        if self.num_hidden_layers < 0:
            raise ValueError("num_hidden_layers cannot be negative.")
        for field_name in ("first_omega_0", "hidden_omega_0"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be a positive finite value.")
        if not isinstance(self.outermost_linear, bool):
            raise TypeError("outermost_linear must be boolean.")


class SIREN(BaseNetwork):
    """Pointwise sinusoidal representation network for continuous fields."""

    config_class = SIRENConfig

    def __init__(self, config: SIRENConfig | None = None) -> None:
        super().__init__(config=config or SIRENConfig())
        total_input_dim = self.config.spatial_dim + self.config.input_dim
        self.first_layer = SineLayer(
            total_input_dim,
            self.config.hidden_dim,
            omega_0=self.config.first_omega_0,
            is_first=True,
        )
        self.hidden_layers = nn.ModuleList(
            SineLayer(
                self.config.hidden_dim,
                self.config.hidden_dim,
                omega_0=self.config.hidden_omega_0,
            )
            for _ in range(self.config.num_hidden_layers)
        )
        self.output_layer = nn.Linear(self.config.hidden_dim, self.config.output_dim)
        output_bound = (
            math.sqrt(6.0 / self.config.hidden_dim) / self.config.hidden_omega_0
        )
        with torch.no_grad():
            self.output_layer.weight.uniform_(-output_bound, output_bound)

    @property
    def required_inputs(self) -> set[str]:
        return {"coordinates"}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            representation="POINT_CLOUD",
            requires={"coordinates"},
            supports={"point_features", "point_mask"},
            variable_geometry=True,
            variable_node_count=True,
        )

    def validate_inputs(self, inputs: ModelInput) -> bool:
        super().validate_inputs(inputs)
        coordinates = self._ensure_tensor(inputs.coordinates, name="coordinates")
        if coordinates.ndim != 3 or coordinates.shape[-1] != self.config.spatial_dim:
            raise ValueError(
                f"coordinates must have shape [B, N, {self.config.spatial_dim}], "
                f"got {tuple(coordinates.shape)}."
            )
        if not coordinates.is_floating_point():
            raise TypeError("coordinates must use a floating-point dtype.")
        if coordinates.shape[0] < 1 or coordinates.shape[1] < 1:
            raise ValueError("Every sample must contain at least one point.")

        if self.config.input_dim > 0:
            if inputs.point_features is None:
                raise ValueError(f"SIREN expects {self.config.input_dim} point features.")
            features = self._ensure_tensor(inputs.point_features, name="point_features")
            expected = (*coordinates.shape[:2], self.config.input_dim)
            if tuple(features.shape) != expected:
                raise ValueError(
                    f"point_features must have shape {expected}, got {tuple(features.shape)}."
                )
            if not features.is_floating_point():
                raise TypeError("point_features must use a floating-point dtype.")
        elif inputs.point_features is not None:
            features = self._ensure_tensor(inputs.point_features, name="point_features")
            expected = (*coordinates.shape[:2], 0)
            if tuple(features.shape) != expected:
                raise ValueError("Set input_dim to use point_features.")
            if not features.is_floating_point():
                raise TypeError("point_features must use a floating-point dtype.")

        if inputs.point_mask is not None:
            mask = torch.as_tensor(inputs.point_mask)
            expected_mask = tuple(coordinates.shape[:2])
            if tuple(mask.shape) != expected_mask:
                raise ValueError(
                    f"point_mask must have shape {expected_mask}, got {tuple(mask.shape)}."
                )
            if mask.dtype != torch.bool:
                raise TypeError("point_mask must use a boolean dtype.")
            if not torch.all(mask.any(dim=1)):
                raise ValueError("Every sample must contain at least one valid point.")
        return True

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        parameter = next(self.parameters())
        coordinates = torch.as_tensor(
            inputs.coordinates, device=parameter.device, dtype=parameter.dtype
        )
        x = coordinates
        if self.config.input_dim > 0:
            features = torch.as_tensor(
                inputs.point_features, device=parameter.device, dtype=parameter.dtype
            )
            x = torch.cat((coordinates, features), dim=-1)

        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        predictions = self.output_layer(x)
        if not self.config.outermost_linear:
            predictions = torch.sin(self.config.hidden_omega_0 * predictions)

        mask = None
        if inputs.point_mask is not None:
            mask = torch.as_tensor(inputs.point_mask, device=parameter.device, dtype=torch.bool)
            predictions = predictions.masked_fill(~mask.unsqueeze(-1), 0.0)
        return ModelOutput(
            predictions=predictions,
            latent=x,
            metadata={"point_mask": mask} if mask is not None else {},
        )
