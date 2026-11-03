from __future__ import annotations

from dataclasses import dataclass

import torch

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.point_inputs import prepare_point_cloud_inputs, validate_point_cloud_inputs
from models.pointnet.components import PointNetClassificationHead, masked_max
from models.types import ModelCapabilities, ModelInput, ModelOutput

from .components import PointNeXtEncoder, PointNeXtSegmentationHead


@dataclass
class PointNeXtConfig(BaseModelConfig):
    """Configuration for PointNeXt; defaults follow PointNeXt-S (no InvResMLP blocks)."""

    input_dim: int = 3
    output_dim: int = 1
    task: str = "classification"
    width: int = 32
    sample_counts: tuple[int, ...] = (512, 128, 32, 8)
    blocks: tuple[int, ...] = (1, 1, 1, 1, 1)
    radius: float = 0.15
    radius_scaling: float = 1.5
    neighbor_count: int = 32
    expansion: int = 4
    dropout: float = 0.5
    normalization: str = "batch"
    representation: str = "POINT_CLOUD"

    def __post_init__(self) -> None:
        self.name = "pointnext"
        self.model_type = "pointnext"
        self.representation = "POINT_CLOUD"
        self.task = self.task.lower()
        self.normalization = self.normalization.lower()
        self.sample_counts = tuple(self.sample_counts)
        self.blocks = tuple(self.blocks)
        if self.input_dim < 3:
            raise ValueError("input_dim must include at least XYZ coordinates (3 channels).")
        if min(self.output_dim, self.width, self.neighbor_count, self.expansion) < 1:
            raise ValueError(
                "output_dim, width, neighbor_count, and expansion must be positive."
            )
        if self.task not in {"classification", "segmentation"}:
            raise ValueError("task must be 'classification' or 'segmentation'.")
        if self.normalization not in {"batch", "none"}:
            raise ValueError("normalization must be batch or none.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if len(self.blocks) != len(self.sample_counts) + 1:
            raise ValueError("blocks must have one entry per stage: len(sample_counts) + 1.")
        if any(value < 1 for value in self.sample_counts + self.blocks):
            raise ValueError("sample_counts and blocks must be positive.")
        if self.radius <= 0 or self.radius_scaling <= 0:
            raise ValueError("radius and radius_scaling must be positive.")

    @property
    def feature_dim(self) -> int:
        return self.input_dim - 3

    @property
    def stage_channels(self) -> tuple[int, ...]:
        return tuple(self.width * 2**index for index in range(len(self.blocks)))

    @property
    def radii(self) -> tuple[float, ...]:
        return tuple(
            self.radius * self.radius_scaling**index for index in range(len(self.blocks))
        )


class PointNeXt(BaseNetwork):
    """PointNeXt: PointNet++ with radius-normalized grouping and InvResMLP stages."""

    config_class = PointNeXtConfig

    def __init__(self, config: PointNeXtConfig | None = None):
        super().__init__(config=config or PointNeXtConfig())
        cfg = self.config
        self.encoder = PointNeXtEncoder(
            cfg.input_dim,
            cfg.stage_channels,
            cfg.sample_counts,
            cfg.radii,
            cfg.neighbor_count,
            cfg.blocks,
            cfg.expansion,
            normalization=cfg.normalization,
        )
        if cfg.task == "classification":
            self.head = PointNetClassificationHead(
                cfg.stage_channels[-1],
                cfg.output_dim,
                cfg.dropout,
                normalization=cfg.normalization,
            )
        else:
            self.head = PointNeXtSegmentationHead(
                cfg.stage_channels,
                cfg.output_dim,
                cfg.dropout,
                normalization=cfg.normalization,
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
        validate_point_cloud_inputs(inputs, self.config.feature_dim, "PointNeXt")
        return True

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        points, features, mask = prepare_point_cloud_inputs(inputs, next(self.parameters()))
        stem_input = points if features is None else torch.cat((points, features), dim=-1)

        level_points, level_features, level_masks = self.encoder(points, stem_input, mask)
        global_feature = masked_max(level_features[-1].transpose(1, 2), level_masks[-1])
        if self.config.task == "classification":
            predictions = self.head(global_feature)
        else:
            predictions = self.head(level_points, level_features, level_masks)
        return ModelOutput(
            predictions=predictions,
            latent=global_feature,
            auxiliary={
                "sampled_coordinates": tuple(level_points[1:]),
                "sampled_masks": tuple(level_masks[1:]),
            },
            metadata={"point_mask": mask},
        )
