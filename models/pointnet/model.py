from __future__ import annotations

from dataclasses import dataclass

import torch

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.types import ModelCapabilities, ModelInput, ModelOutput

from .components import (
    PointNetClassificationHead,
    PointNetEncoder,
    PointNetSegmentationHead,
)


@dataclass
class PointNetConfig(BaseModelConfig):
    """Configuration for classification or point-wise PointNet prediction."""

    input_dim: int = 3
    output_dim: int = 1
    task: str = "segmentation"
    global_dim: int = 1024
    dropout: float = 0.3
    input_transform: bool = True
    feature_transform: bool = True
    representation: str = "POINT_CLOUD"

    def __post_init__(self) -> None:
        self.name = "pointnet"
        self.model_type = "pointnet"
        self.representation = "POINT_CLOUD"
        self.task = self.task.lower()
        if self.input_dim < 3:
            raise ValueError("input_dim must include at least XYZ coordinates (3 channels).")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if self.global_dim <= 0:
            raise ValueError("global_dim must be positive.")
        if self.task not in {"classification", "segmentation"}:
            raise ValueError("task must be 'classification' or 'segmentation'.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

    @property
    def feature_dim(self) -> int:
        return self.input_dim - 3


class PointNet(BaseNetwork):
    """PointNet for global classification and point-wise prediction."""

    config_class = PointNetConfig

    def __init__(self, config: PointNetConfig | None = None):
        super().__init__(config=config or PointNetConfig())
        self.encoder = PointNetEncoder(
            self.config.input_dim,
            self.config.global_dim,
            input_transform=self.config.input_transform,
            feature_transform=self.config.feature_transform,
        )
        if self.config.task == "classification":
            self.head = PointNetClassificationHead(
                self.config.global_dim, self.config.output_dim, self.config.dropout
            )
        else:
            self.head = PointNetSegmentationHead(
                self.config.global_dim, self.config.output_dim, self.config.dropout
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
                "PointNet coordinates must have shape [B, N, 3], "
                f"got {tuple(points.shape)}."
            )
        if not points.is_floating_point():
            raise TypeError("coordinates must use a floating-point dtype.")
        if points.shape[1] < 1:
            raise ValueError("Each point cloud must contain at least one point.")

        expected_features = self.config.feature_dim
        if inputs.point_features is None:
            if expected_features:
                raise ValueError(
                    f"PointNet expects {expected_features} additional point features."
                )
        else:
            features = self._ensure_tensor(inputs.point_features, name="point_features")
            expected_shape = (*points.shape[:2], expected_features)
            if tuple(features.shape) != expected_shape:
                raise ValueError(
                    f"point_features must have shape {expected_shape}, "
                    f"got {tuple(features.shape)}."
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
        mask = None
        if inputs.point_mask is not None:
            mask = torch.as_tensor(inputs.point_mask, device=parameter.device, dtype=torch.bool)

        local, global_feature, transforms = self.encoder(points, features, mask)
        if self.config.task == "classification":
            predictions = self.head(global_feature)
        else:
            predictions = self.head(local, global_feature)
            if mask is not None:
                predictions = predictions.masked_fill(~mask.unsqueeze(-1), 0.0)

        return ModelOutput(
            predictions=predictions,
            latent=global_feature,
            auxiliary=transforms,
            metadata={"point_mask": mask} if mask is not None else {},
        )
