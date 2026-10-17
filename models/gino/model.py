from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.fno.components import FourierOperatorBlock3d, get_activation
from models.types import ModelCapabilities, ModelInput, ModelOutput

from .components import KernelIntegralTransform


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _parse_resolution(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, Integral) and not isinstance(value, bool):
        resolution = (int(value),) * 3
    else:
        try:
            raw = tuple(value)
        except TypeError as exc:
            raise ValueError(
                "latent_resolution must be an integer or three positive integers."
            ) from exc
        if len(raw) != 3:
            raise ValueError("latent_resolution must contain three values for [D, H, W].")
        resolution = tuple(
            _positive_int(size, "latent_resolution values") for size in raw
        )
    return tuple(_positive_int(size, "latent_resolution values") for size in resolution)


def _parse_modes(value: Sequence[int]) -> tuple[int, int, int]:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError("modes must contain three positive integers for [D, H, W].") from exc
    if len(raw) != 3:
        raise ValueError("modes must contain three positive integers for [D, H, W].")
    return tuple(_positive_int(mode, "Fourier modes") for mode in raw)


@dataclass
class GINOConfig(BaseModelConfig):
    """Configuration for the geometry-informed neural operator."""

    input_dim: int = 1
    output_dim: int = 1
    latent_resolution: tuple[int, int, int] | int = (16, 16, 16)
    width: int = 32
    modes: tuple[int, int, int] = (8, 8, 8)
    num_layers: int = 4
    kernel_radius: float = 0.2
    kernel_hidden_dim: int = 32
    kernel_chunk_size: int = 256
    kernel_max_pairs: int = 262144
    activation: str = "gelu"
    representation: str = "POINT_CLOUD"

    def __post_init__(self) -> None:
        self.name = "gino"
        self.model_type = "gino"
        self.representation = "POINT_CLOUD"
        self.latent_resolution = _parse_resolution(self.latent_resolution)
        self.modes = _parse_modes(self.modes)
        for field_name in ("input_dim", "output_dim", "width", "num_layers", "kernel_hidden_dim", "kernel_chunk_size", "kernel_max_pairs"):
            setattr(self, field_name, _positive_int(getattr(self, field_name), field_name))
        self.activation = self.activation.lower()
        if self.activation not in {"gelu", "relu", "silu"}:
            raise ValueError("activation must be gelu, relu, or silu.")
        if not isinstance(self.kernel_radius, (int, float)) or not 0.0 < self.kernel_radius <= 3**0.5:
            raise ValueError("kernel_radius must be positive and no larger than sqrt(3).")
        depth, height, width = self.latent_resolution
        modes_d, modes_h, modes_w = self.modes
        if modes_d > depth // 2:
            raise ValueError(f"modes[0]={modes_d} exceeds latent D//2={depth // 2}.")
        if modes_h > height // 2:
            raise ValueError(f"modes[1]={modes_h} exceeds latent H//2={height // 2}.")
        if modes_w > width // 2 + 1:
            raise ValueError(f"modes[2]={modes_w} exceeds latent W//2+1={width // 2 + 1}.")


class GINO(BaseNetwork):
    """Geometry-aware point-to-grid, Fourier, and grid-to-point operator."""

    config_class = GINOConfig

    def __init__(self, config: GINOConfig | None = None) -> None:
        super().__init__(config=config or GINOConfig())
        self.encoder = KernelIntegralTransform(
            self.config.input_dim,
            self.config.width,
            self.config.kernel_hidden_dim,
            self.config.kernel_radius,
            self.config.kernel_chunk_size,
            self.config.kernel_max_pairs,
        )
        self.blocks = nn.ModuleList(
            FourierOperatorBlock3d(self.config.width, self.config.modes)
            for _ in range(self.config.num_layers)
        )
        self.activation = get_activation(self.config.activation)
        self.decoder = KernelIntegralTransform(
            self.config.width,
            self.config.width,
            self.config.kernel_hidden_dim,
            self.config.kernel_radius,
            self.config.kernel_chunk_size,
            self.config.kernel_max_pairs,
        )
        self.output_projection = nn.Linear(self.config.width, self.config.output_dim)

    @property
    def required_inputs(self) -> set[str]:
        return {"coordinates", "point_features"}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            representation="POINT_CLOUD",
            requires=set(self.required_inputs),
            supports={"point_mask", "query_coordinates"},
            variable_geometry=True,
            variable_node_count=True,
        )

    def validate_inputs(self, inputs: ModelInput) -> bool:
        super().validate_inputs(inputs)
        coordinates = self._ensure_tensor(inputs.coordinates, name="coordinates")
        features = self._ensure_tensor(inputs.point_features, name="point_features")
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError(
                "GINO coordinates must have shape [B, N, 3], "
                f"got {tuple(coordinates.shape)}."
            )
        if coordinates.shape[1] == 0:
            raise ValueError("Each source point cloud must contain at least one point.")
        if not coordinates.is_floating_point():
            raise TypeError("coordinates must use a floating-point dtype.")
        expected_features = (*coordinates.shape[:2], self.config.input_dim)
        if tuple(features.shape) != expected_features:
            raise ValueError(
                f"point_features must have shape {expected_features}, got {tuple(features.shape)}."
            )
        if not features.is_floating_point():
            raise TypeError("point_features must use a floating-point dtype.")

        if inputs.point_mask is None:
            source_mask = torch.ones(coordinates.shape[:2], dtype=torch.bool, device=coordinates.device)
        else:
            source_mask = torch.as_tensor(inputs.point_mask, device=coordinates.device)
            if tuple(source_mask.shape) != tuple(coordinates.shape[:2]):
                raise ValueError(
                    f"point_mask must have shape {tuple(coordinates.shape[:2])}, "
                    f"got {tuple(source_mask.shape)}."
                )
            if source_mask.dtype != torch.bool:
                raise TypeError("point_mask must use a boolean dtype.")
        if not torch.all(source_mask.any(dim=1)):
            raise ValueError("Every point cloud must contain at least one valid source point.")
        if not torch.isfinite(coordinates[source_mask]).all():
            raise ValueError("Valid source coordinates must be finite.")
        if not torch.isfinite(features[source_mask]).all():
            raise ValueError("Valid point_features must be finite.")
        valid_source_coordinates = coordinates[source_mask]
        if torch.any((valid_source_coordinates < 0.0) | (valid_source_coordinates > 1.0)):
            raise ValueError("Valid source coordinates must be normalized to [0, 1].")

        if inputs.query_coordinates is not None:
            query_coordinates = self._ensure_tensor(
                inputs.query_coordinates, name="query_coordinates"
            )
            if query_coordinates.ndim != 3 or query_coordinates.shape[-1] != 3:
                raise ValueError(
                    "query_coordinates must have shape [B, Q, 3], "
                    f"got {tuple(query_coordinates.shape)}."
                )
            if query_coordinates.shape[0] != coordinates.shape[0]:
                raise ValueError("query_coordinates batch size must match coordinates.")
            if query_coordinates.shape[1] == 0:
                raise ValueError("Each query set must contain at least one point.")
            if not query_coordinates.is_floating_point():
                raise TypeError("query_coordinates must use a floating-point dtype.")
            if not torch.isfinite(query_coordinates).all():
                raise ValueError("query_coordinates must be finite.")
            if torch.any((query_coordinates < 0.0) | (query_coordinates > 1.0)):
                raise ValueError("query_coordinates must be normalized to [0, 1].")
        return True

    def _latent_coordinates(self, reference: torch.Tensor) -> torch.Tensor:
        axes = [
            torch.linspace(0.0, 1.0, size, device=reference.device, dtype=reference.dtype)
            for size in self.config.latent_resolution
        ]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return grid.reshape(-1, 3)

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        parameter = next(self.parameters())
        coordinates = torch.as_tensor(
            inputs.coordinates, device=parameter.device, dtype=parameter.dtype
        )
        features = torch.as_tensor(
            inputs.point_features, device=parameter.device, dtype=parameter.dtype
        )
        source_mask = (
            torch.ones(coordinates.shape[:2], dtype=torch.bool, device=parameter.device)
            if inputs.point_mask is None
            else torch.as_tensor(inputs.point_mask, device=parameter.device, dtype=torch.bool)
        )
        default_queries = inputs.query_coordinates is None
        queries = coordinates if default_queries else torch.as_tensor(
            inputs.query_coordinates, device=parameter.device, dtype=parameter.dtype
        )
        latent_coordinates = self._latent_coordinates(coordinates)
        latent_batches = []

        for batch_index in range(coordinates.shape[0]):
            valid = source_mask[batch_index]
            latent = self.encoder(
                coordinates[batch_index, valid],
                latent_coordinates,
                features[batch_index, valid],
            )
            latent = latent.reshape(*self.config.latent_resolution, self.config.width)
            latent_batches.append(latent.permute(3, 0, 1, 2))

        latent_grid = torch.stack(latent_batches, dim=0)
        for index, block in enumerate(self.blocks):
            latent_grid = block(latent_grid)
            if index < len(self.blocks) - 1:
                latent_grid = self.activation(latent_grid)

        flat_latent = latent_grid.permute(0, 2, 3, 4, 1).reshape(
            coordinates.shape[0], -1, self.config.width
        )
        decoded_batches = []
        for batch_index in range(coordinates.shape[0]):
            query_mask = source_mask[batch_index] if default_queries else None
            valid_queries = queries[batch_index] if query_mask is None else queries[batch_index, query_mask]
            decoded = self.decoder(
                latent_coordinates,
                valid_queries,
                flat_latent[batch_index],
            )
            projected = self.output_projection(decoded)
            if query_mask is not None:
                batch_predictions = projected.new_zeros(
                    (queries.shape[1], self.config.output_dim)
                )
                batch_predictions[query_mask] = projected
                decoded_batches.append(batch_predictions)
            else:
                decoded_batches.append(projected)

        predictions = torch.stack(decoded_batches, dim=0)
        return ModelOutput(
            predictions=predictions,
            latent=latent_grid,
            metadata={
                "query_coordinates": queries,
                "query_is_source": default_queries,
                "point_mask": source_mask,
                "latent_resolution": self.config.latent_resolution,
            },
        )