from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.types import ModelCapabilities, ModelInput, ModelOutput

from .block import TransolverBlock
from .components import InputEmbedding, OutputHead


@dataclass
class TransolverConfig(BaseModelConfig):
    spatial_dim: int = 3
    input_dim: int = 0
    output_dim: int = 1
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    num_slices: int = 32
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    activation: str = "gelu"
    return_attention_weights: bool = False
    representation: str = "POINT_CLOUD"

    def __post_init__(self) -> None:
        self.name = "transolver"
        self.model_type = "transolver"
        self.representation = "POINT_CLOUD"
        for field_name in (
            "spatial_dim", "output_dim", "hidden_dim", "num_layers", "num_heads", "num_slices"
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if self.input_dim < 0:
            raise ValueError("input_dim cannot be negative.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.activation.lower() not in {"gelu", "relu", "silu", "tanh"}:
            raise ValueError("activation must be gelu, relu, silu, or tanh.")


class Transolver(BaseNetwork):
    """Transolver for point-wise physical-field prediction on irregular domains."""

    config_class = TransolverConfig

    def __init__(self, config: TransolverConfig | None = None) -> None:
        super().__init__(config=config or TransolverConfig())
        self.input_embedding = InputEmbedding(
            self.config.spatial_dim,
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.activation,
        )
        self.placeholder = nn.Parameter(
            torch.rand(self.config.hidden_dim) / self.config.hidden_dim
        )
        self.blocks = nn.ModuleList(
            TransolverBlock(
                self.config.hidden_dim,
                num_heads=self.config.num_heads,
                num_slices=self.config.num_slices,
                mlp_ratio=self.config.mlp_ratio,
                dropout=self.config.dropout,
                activation=self.config.activation,
            )
            for _ in range(self.config.num_layers)
        )
        self.output_head = OutputHead(self.config.hidden_dim, self.config.output_dim)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

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
        if coordinates.shape[1] < 1:
            raise ValueError("Each sample must contain at least one point.")
        if self.config.input_dim:
            if inputs.point_features is None:
                raise ValueError(f"Transolver expects {self.config.input_dim} point features.")
            features = self._ensure_tensor(inputs.point_features, name="point_features")
            expected = (*coordinates.shape[:2], self.config.input_dim)
            if tuple(features.shape) != expected:
                raise ValueError(f"point_features must have shape {expected}.")
            if not features.is_floating_point():
                raise TypeError("point_features must use a floating-point dtype.")
        elif inputs.point_features is not None:
            features = self._ensure_tensor(inputs.point_features, name="point_features")
            if features.shape[-1] != 0:
                raise ValueError("Set input_dim to use point_features.")
        if inputs.point_mask is not None:
            mask = torch.as_tensor(inputs.point_mask)
            if tuple(mask.shape) != tuple(coordinates.shape[:2]) or mask.dtype != torch.bool:
                raise ValueError("point_mask must be boolean with shape [B, N].")
            if not torch.all(mask.any(dim=1)):
                raise ValueError("Every sample must contain at least one valid point.")
        return True

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        parameter = next(self.parameters())
        coordinates = torch.as_tensor(
            inputs.coordinates, device=parameter.device, dtype=parameter.dtype
        )
        features = None if inputs.point_features is None else torch.as_tensor(
            inputs.point_features, device=parameter.device, dtype=parameter.dtype
        )
        mask = None if inputs.point_mask is None else torch.as_tensor(
            inputs.point_mask, device=parameter.device, dtype=torch.bool
        )
        hidden = self.input_embedding(coordinates, features) + self.placeholder
        if mask is not None:
            hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)

        attention_data = []
        for block in self.blocks:
            if self.config.return_attention_weights:
                hidden, block_data = block(hidden, mask, return_attention=True)
                attention_data.append(block_data)
            else:
                hidden = block(hidden, mask)
        predictions = self.output_head(hidden)
        if mask is not None:
            predictions = predictions.masked_fill(~mask.unsqueeze(-1), 0.0)
        return ModelOutput(
            predictions=predictions,
            latent=hidden,
            auxiliary={"physics_attention": attention_data} if attention_data else {},
            metadata={"point_mask": mask} if mask is not None else {},
        )
