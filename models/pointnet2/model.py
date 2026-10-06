from __future__ import annotations

from dataclasses import dataclass

import torch

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.pointnet.components import PointNetClassificationHead
from models.types import ModelCapabilities, ModelInput, ModelOutput

from .components import PointNet2Encoder, PointNet2SegmentationHead


@dataclass
class PointNet2Config(BaseModelConfig):
    """Configuration for single-scale PointNet++ classification or segmentation."""

    input_dim: int = 3
    output_dim: int = 1
    task: str = "segmentation"
    sample_counts: tuple[int, int] = (512, 128)
    radii: tuple[float, float] = (0.2, 0.4)
    neighbor_counts: tuple[int, int] = (32, 64)
    abstraction_channels: tuple[tuple[int, ...], tuple[int, ...]] = (
        (64, 64, 128),
        (128, 128, 256),
    )
    global_dim: int = 1024
    dropout: float = 0.3
    representation: str = "POINT_CLOUD"

    def __post_init__(self) -> None:
        self.name = "pointnet2"
        self.model_type = "pointnet2"
        self.representation = "POINT_CLOUD"
        self.task = self.task.lower()
        self.sample_counts = tuple(self.sample_counts)
        self.radii = tuple(self.radii)
        self.neighbor_counts = tuple(self.neighbor_counts)
        self.abstraction_channels = tuple(
            tuple(channels) for channels in self.abstraction_channels
        )
        if self.input_dim < 3:
            raise ValueError("input_dim must include at least XYZ coordinates (3 channels).")
        if self.output_dim <= 0 or self.global_dim <= 0:
            raise ValueError("output_dim and global_dim must be positive.")
        if self.task not in {"classification", "segmentation"}:
            raise ValueError("task must be 'classification' or 'segmentation'.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        for name, values in (
            ("sample_counts", self.sample_counts),
            ("radii", self.radii),
            ("neighbor_counts", self.neighbor_counts),
            ("abstraction_channels", self.abstraction_channels),
        ):
            if len(values) != 2:
                raise ValueError(f"{name} must contain exactly two hierarchy levels.")
        if any(value < 1 for value in self.sample_counts + self.neighbor_counts):
            raise ValueError("sample_counts and neighbor_counts must be positive.")
        if any(value <= 0 for value in self.radii):
            raise ValueError("radii must be positive.")
        if any(
            not channels or any(value < 1 for value in channels)
            for channels in self.abstraction_channels
        ):
            raise ValueError("Every abstraction MLP must contain positive channel sizes.")

    @property
    def feature_dim(self) -> int:
        return self.input_dim - 3


class PointNet2(BaseNetwork):
    """PointNet++ with hierarchical local abstraction and feature propagation."""

    config_class = PointNet2Config

    def __init__(self, config: PointNet2Config | None = None):
        super().__init__(config=config or PointNet2Config())
        self.encoder = PointNet2Encoder(
            self.config.feature_dim,
            self.config.sample_counts,
            self.config.radii,
            self.config.neighbor_counts,
            self.config.abstraction_channels,
            self.config.global_dim,
        )
        if self.config.task == "classification":
            self.head = PointNetClassificationHead(
                self.config.global_dim, self.config.output_dim, self.config.dropout
            )
        else:
            self.head = PointNet2SegmentationHead(
                self.config.input_dim,
                self.config.abstraction_channels[0][-1],
                self.config.abstraction_channels[1][-1],
                self.config.global_dim,
                self.config.output_dim,
                self.config.dropout,
            )

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
        points = self._ensure_tensor(inputs.coordinates, name="coordinates")
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                "PointNet++ coordinates must have shape [B, N, 3], "
                f"got {tuple(points.shape)}."
            )
        if not points.is_floating_point():
            raise TypeError("coordinates must use a floating-point dtype.")
        if points.shape[1] < 1:
            raise ValueError("Each point cloud must contain at least one point.")

        if inputs.point_features is None:
            if self.config.feature_dim:
                raise ValueError(
                    f"PointNet++ expects {self.config.feature_dim} additional point features."
                )
        else:
            features = self._ensure_tensor(inputs.point_features, name="point_features")
            expected = (*points.shape[:2], self.config.feature_dim)
            if tuple(features.shape) != expected:
                raise ValueError(
                    f"point_features must have shape {expected}, got {tuple(features.shape)}."
                )
            if not features.is_floating_point():
                raise TypeError("point_features must use a floating-point dtype.")

        if inputs.point_mask is not None:
            mask = torch.as_tensor(inputs.point_mask)
            if tuple(mask.shape) != tuple(points.shape[:2]):
                raise ValueError(
                    f"point_mask must have shape {tuple(points.shape[:2])}, "
                    f"got {tuple(mask.shape)}."
                )
            if mask.dtype != torch.bool:
                raise TypeError("point_mask must use a boolean dtype.")
            if not torch.all(mask.any(dim=1)):
                raise ValueError("Every point cloud must contain at least one valid point.")
        return True

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        parameter = next(self.parameters())
        points = torch.as_tensor(
            inputs.coordinates, device=parameter.device, dtype=parameter.dtype
        )
        features = None
        if inputs.point_features is not None:
            features = torch.as_tensor(
                inputs.point_features, device=parameter.device, dtype=parameter.dtype
            )
        mask = (
            torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
            if inputs.point_mask is None
            else torch.as_tensor(inputs.point_mask, device=points.device, dtype=torch.bool)
        )

        global_feature, level_points, level_features, level_masks = self.encoder(
            points, features, mask
        )
        if self.config.task == "classification":
            predictions = self.head(global_feature)
        else:
            predictions = self.head(
                points,
                features,
                mask,
                level_points,
                level_features,
                level_masks,
            )
        return ModelOutput(
            predictions=predictions,
            latent=global_feature,
            auxiliary={
                "sampled_coordinates": level_points,
                "sampled_masks": level_masks,
            },
            metadata={"point_mask": mask},
        )
