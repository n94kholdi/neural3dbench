from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.types import ModelCapabilities, ModelInput, ModelOutput


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


@dataclass
class MLPConfig(BaseModelConfig):
    """Configuration for a pointwise coordinate MLP."""

    spatial_dim: int = 3
    input_dim: int = 0
    output_dim: int = 1
    widths: tuple[int, ...] = (256, 256, 256)
    activation: str = "tanh"
    residual: bool = False
    representation: str = "POINT_CLOUD"

    def __post_init__(self) -> None:
        self.name = "mlp"
        self.model_type = "mlp"
        self.representation = "POINT_CLOUD"
        for field_name in ("spatial_dim", "output_dim"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
            setattr(self, field_name, int(value))
        if (
            isinstance(self.input_dim, bool)
            or not isinstance(self.input_dim, Integral)
            or self.input_dim < 0
        ):
            raise ValueError("input_dim must be a non-negative integer.")
        self.input_dim = int(self.input_dim)
        try:
            self.widths = tuple(self.widths)
        except TypeError as exc:
            raise ValueError("widths must contain positive integers.") from exc
        if not self.widths:
            raise ValueError("widths must contain at least one hidden width.")
        if any(isinstance(width, bool) or not isinstance(width, Integral) or width <= 0
               for width in self.widths):
            raise ValueError("widths must contain positive integers.")
        self.widths = tuple(int(width) for width in self.widths)
        if not isinstance(self.activation, str):
            raise TypeError("activation must be a string.")
        self.activation = self.activation.lower()
        if self.activation not in _ACTIVATIONS:
            raise ValueError("activation must be tanh, silu, relu, or gelu.")
        if not isinstance(self.residual, bool):
            raise TypeError("residual must be boolean.")


class _HiddenLayer(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, activation: str, residual: bool
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = _ACTIVATIONS[activation]()
        self.residual = residual and input_dim == output_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.activation(self.linear(inputs))
        return outputs + inputs if self.residual else outputs


class MLP(BaseNetwork):
    """Pointwise multilayer perceptron for continuous coordinate fields."""

    config_class = MLPConfig

    def __init__(self, config: MLPConfig | None = None) -> None:
        super().__init__(config=config or MLPConfig())
        dimensions = (
            self.config.spatial_dim + self.config.input_dim,
            *self.config.widths,
        )
        self.hidden_layers = nn.ModuleList(
            _HiddenLayer(
                input_dim, output_dim, self.config.activation, self.config.residual
            )
            for input_dim, output_dim in zip(dimensions, dimensions[1:])
        )
        self.output_layer = nn.Linear(self.config.widths[-1], self.config.output_dim)

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
                raise ValueError(f"MLP expects {self.config.input_dim} point features.")
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
        hidden = coordinates
        if self.config.input_dim > 0:
            features = torch.as_tensor(
                inputs.point_features, device=parameter.device, dtype=parameter.dtype
            )
            hidden = torch.cat((coordinates, features), dim=-1)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        predictions = self.output_layer(hidden)

        mask = None
        if inputs.point_mask is not None:
            mask = torch.as_tensor(inputs.point_mask, device=parameter.device, dtype=torch.bool)
            predictions = predictions.masked_fill(~mask.unsqueeze(-1), 0.0)
            hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        return ModelOutput(
            predictions=predictions,
            latent=hidden,
            metadata={"point_mask": mask} if mask is not None else {},
        )
