from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from models.pointnet.components import SharedPointMLP
from models.pointnet2.components import (
    FeaturePropagation,
    SharedPointMLP2d,
    farthest_point_sample,
    index_points,
    query_ball_point,
)


def group_and_pool(
    mlp: nn.Module,
    radius: float,
    neighbor_count: int,
    points: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
    centroids: torch.Tensor,
    centroid_mask: torch.Tensor,
) -> torch.Tensor:
    """Ball-query neighborhoods, encode radius-normalized offsets plus features, max-pool."""

    indices, grouped_mask = query_ball_point(
        radius,
        neighbor_count,
        points,
        centroids,
        point_mask=mask,
        centroid_mask=centroid_mask,
    )
    offsets = (index_points(points, indices) - centroids.unsqueeze(2)) / radius
    grouped = torch.cat((offsets, index_points(features, indices)), dim=-1)
    encoded = mlp(grouped.permute(0, 3, 1, 2))
    encoded = encoded.masked_fill(
        ~grouped_mask.unsqueeze(1), torch.finfo(encoded.dtype).min
    )
    pooled = encoded.max(dim=-1).values.transpose(1, 2)
    return pooled.masked_fill(~centroid_mask.unsqueeze(-1), 0.0)


class PointNeXtSetAbstraction(nn.Module):
    """Farthest-point downsampling followed by local grouping and max pooling."""

    def __init__(
        self,
        sample_count: int,
        radius: float,
        neighbor_count: int,
        input_channels: int,
        output_channels: int,
        *,
        normalization: str = "batch",
    ):
        super().__init__()
        self.sample_count = sample_count
        self.radius = radius
        self.neighbor_count = neighbor_count
        self.mlp = SharedPointMLP2d(
            (input_channels + 3, output_channels), normalization=normalization
        )

    def forward(
        self, points: torch.Tensor, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices, centroid_mask = farthest_point_sample(
            points, min(self.sample_count, points.shape[1]), mask
        )
        centroids = index_points(points, indices)
        pooled = group_and_pool(
            self.mlp,
            self.radius,
            self.neighbor_count,
            points,
            features,
            mask,
            centroids,
            centroid_mask,
        )
        return centroids, pooled, centroid_mask


class InvResMLP(nn.Module):
    """Inverted residual block: local aggregation, then an expanded pointwise MLP."""

    def __init__(
        self,
        channels: int,
        radius: float,
        neighbor_count: int,
        expansion: int,
        *,
        normalization: str = "batch",
    ):
        super().__init__()
        self.radius = radius
        self.neighbor_count = neighbor_count
        self.local = SharedPointMLP2d((channels + 3, channels), normalization=normalization)
        self.pointwise = SharedPointMLP(
            (channels, channels * expansion, channels),
            final_activation=False,
            normalization=normalization,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self, points: torch.Tensor, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        local = group_and_pool(
            self.local,
            self.radius,
            self.neighbor_count,
            points,
            features,
            mask,
            points,
            mask,
        )
        updated = self.pointwise(local.transpose(1, 2)).transpose(1, 2)
        updated = self.activation(updated + features)
        return updated.masked_fill(~mask.unsqueeze(-1), 0.0)


class PointNeXtEncoder(nn.Module):
    """Stem MLP plus downsampling stages, each with optional InvResMLP blocks."""

    def __init__(
        self,
        input_dim: int,
        stage_channels: Sequence[int],
        sample_counts: Sequence[int],
        radii: Sequence[float],
        neighbor_count: int,
        blocks: Sequence[int],
        expansion: int,
        *,
        normalization: str = "batch",
    ):
        super().__init__()
        self.stem = SharedPointMLP(
            (input_dim, stage_channels[0]), normalization=normalization
        )
        self.downsamples = nn.ModuleList(
            PointNeXtSetAbstraction(
                sample_counts[index - 1],
                radii[index],
                neighbor_count,
                stage_channels[index - 1],
                stage_channels[index],
                normalization=normalization,
            )
            for index in range(1, len(stage_channels))
        )
        # The first block of each stage is the stem/downsampling layer itself.
        self.stage_blocks = nn.ModuleList(
            nn.ModuleList(
                InvResMLP(
                    channels,
                    radius,
                    neighbor_count,
                    expansion,
                    normalization=normalization,
                )
                for _ in range(block_count - 1)
            )
            for channels, radius, block_count in zip(stage_channels, radii, blocks)
        )

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        features = self.stem(features.transpose(1, 2)).transpose(1, 2)
        features = features.masked_fill(~mask.unsqueeze(-1), 0.0)
        level_points, level_features, level_masks = [], [], []
        for index, blocks in enumerate(self.stage_blocks):
            if index > 0:
                points, features, mask = self.downsamples[index - 1](
                    points, features, mask
                )
            for block in blocks:
                features = block(points, features, mask)
            level_points.append(points)
            level_features.append(features)
            level_masks.append(mask)
        return level_points, level_features, level_masks


class PointNeXtSegmentationHead(nn.Module):
    """Feature-propagation decoder from the coarsest stage back to every input point."""

    def __init__(
        self,
        stage_channels: Sequence[int],
        output_dim: int,
        dropout: float,
        *,
        normalization: str = "batch",
    ):
        super().__init__()
        self.propagations = nn.ModuleList(
            FeaturePropagation(
                stage_channels[index] + stage_channels[index + 1],
                (stage_channels[index], stage_channels[index]),
                normalization=normalization,
            )
            for index in range(len(stage_channels) - 1)
        )
        width = stage_channels[0]
        self.output = nn.Sequential(
            SharedPointMLP((width, width), normalization=normalization),
            nn.Dropout(dropout),
            nn.Conv1d(width, output_dim, 1),
        )

    def forward(
        self,
        level_points: Sequence[torch.Tensor],
        level_features: Sequence[torch.Tensor],
        level_masks: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        decoded = level_features[-1]
        for index in reversed(range(len(self.propagations))):
            decoded = self.propagations[index](
                level_points[index],
                level_points[index + 1],
                level_features[index],
                decoded,
                level_masks[index],
                level_masks[index + 1],
            )
        prediction = self.output(decoded.transpose(1, 2)).transpose(1, 2)
        return prediction.masked_fill(~level_masks[0].unsqueeze(-1), 0.0)
