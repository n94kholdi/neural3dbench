from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.point_inputs import prepare_point_cloud_inputs, validate_point_cloud_inputs
from models.pointnet.components import (
    PointNetClassificationHead,
    SharedPointMLP,
    masked_max,
)
from models.types import ModelCapabilities, ModelInput, ModelOutput

from .components import EdgeConv


@dataclass
class DGCNNConfig(BaseModelConfig):
    """Configuration for Dynamic Graph CNN classification or segmentation."""

    input_dim: int = 3
    output_dim: int = 1
    task: str = "classification"
    neighbor_count: int = 20
    edge_channels: tuple[int, ...] = (64, 64, 128, 256)
    embedding_dim: int = 1024
    dropout: float = 0.5
    normalization: str = "batch"
    representation: str = "POINT_CLOUD"

    def __post_init__(self) -> None:
        self.name = "dgcnn"
        self.model_type = "dgcnn"
        self.representation = "POINT_CLOUD"
        self.task = self.task.lower()
        self.normalization = self.normalization.lower()
        self.edge_channels = tuple(self.edge_channels)
        if self.input_dim < 3:
            raise ValueError("input_dim must include at least XYZ coordinates (3 channels).")
        if self.output_dim <= 0 or self.embedding_dim <= 0:
            raise ValueError("output_dim and embedding_dim must be positive.")
        if self.task not in {"classification", "segmentation"}:
            raise ValueError("task must be 'classification' or 'segmentation'.")
        if self.normalization not in {"batch", "none"}:
            raise ValueError("normalization must be batch or none.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.neighbor_count < 1:
            raise ValueError("neighbor_count must be positive.")
        if not self.edge_channels or any(value < 1 for value in self.edge_channels):
            raise ValueError("edge_channels must contain positive channel sizes.")

    @property
    def feature_dim(self) -> int:
        return self.input_dim - 3


class DGCNN(BaseNetwork):
    """DGCNN with EdgeConv layers over k-NN graphs recomputed in feature space."""

    config_class = DGCNNConfig

    def __init__(self, config: DGCNNConfig | None = None):
        super().__init__(config=config or DGCNNConfig())
        cfg = self.config
        channels = (cfg.input_dim, *cfg.edge_channels)
        self.edge_convs = nn.ModuleList(
            EdgeConv(
                input_channels,
                output_channels,
                cfg.neighbor_count,
                normalization=cfg.normalization,
            )
            for input_channels, output_channels in zip(channels[:-1], channels[1:])
        )
        local_dim = sum(cfg.edge_channels)
        embedding: list[nn.Module] = [
            nn.Conv1d(local_dim, cfg.embedding_dim, 1, bias=cfg.normalization == "none")
        ]
        if cfg.normalization == "batch":
            embedding.append(nn.BatchNorm1d(cfg.embedding_dim))
        embedding.append(nn.LeakyReLU(0.2, inplace=True))
        self.embedding = nn.Sequential(*embedding)

        global_dim = 2 * cfg.embedding_dim
        if cfg.task == "classification":
            self.head = PointNetClassificationHead(
                global_dim, cfg.output_dim, cfg.dropout, normalization=cfg.normalization
            )
        else:
            self.head = nn.Sequential(
                SharedPointMLP(
                    (local_dim + global_dim, 256, 256, 128),
                    normalization=cfg.normalization,
                ),
                nn.Dropout(cfg.dropout),
                nn.Conv1d(128, cfg.output_dim, 1),
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
        validate_point_cloud_inputs(inputs, self.config.feature_dim, "DGCNN")
        return True

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        points, features, mask = prepare_point_cloud_inputs(inputs, next(self.parameters()))
        x = points if features is None else torch.cat((points, features), dim=-1)

        local_features = []
        for edge_conv in self.edge_convs:
            x = edge_conv(x, mask)
            local_features.append(x)
        local = torch.cat(local_features, dim=-1).transpose(1, 2)
        embedded = self.embedding(local)

        valid = mask.unsqueeze(1)
        mean = (embedded * valid).sum(dim=2) / valid.sum(dim=2)
        global_feature = torch.cat((masked_max(embedded, mask), mean), dim=1)

        if self.config.task == "classification":
            predictions = self.head(global_feature)
        else:
            repeated = global_feature.unsqueeze(-1).expand(-1, -1, local.shape[-1])
            predictions = self.head(torch.cat((local, repeated), dim=1)).transpose(1, 2)
            predictions = predictions.masked_fill(~mask.unsqueeze(-1), 0.0)
        return ModelOutput(
            predictions=predictions,
            latent=global_feature,
            auxiliary={},
            metadata={"point_mask": mask},
        )
