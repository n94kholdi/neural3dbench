from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def square_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance between two batched point sets."""

    return (
        source.square().sum(dim=-1, keepdim=True)
        + target.square().sum(dim=-1).unsqueeze(1)
        - 2.0 * torch.bmm(source, target.transpose(1, 2))
    ).clamp_min(0.0)


def index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Batch-aware indexing for ``[B, N, C]`` point or feature tensors."""

    batch_shape = (points.shape[0],) + (1,) * (indices.ndim - 1)
    batch_indices = torch.arange(points.shape[0], device=points.device).view(batch_shape)
    return points[batch_indices.expand_as(indices), indices]


def farthest_point_sample(
    points: torch.Tensor,
    sample_count: int,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic masked farthest-point sampling.

    The first centroid is the valid point furthest from the cloud mean, which
    avoids an order-dependent random seed and makes evaluation reproducible.
    """

    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    batch_size, point_count, _ = points.shape
    if mask is None:
        mask = torch.ones(
            (batch_size, point_count), dtype=torch.bool, device=points.device
        )
    valid_counts = mask.sum(dim=1)
    if torch.any(valid_counts == 0):
        raise ValueError("Every point cloud must contain at least one valid point.")

    centroid = (points * mask.unsqueeze(-1)).sum(dim=1) / valid_counts.unsqueeze(-1)
    initial_distance = (points - centroid.unsqueeze(1)).square().sum(dim=-1)
    farthest = initial_distance.masked_fill(~mask, -1.0).argmax(dim=1)

    indices = torch.empty(
        (batch_size, sample_count), dtype=torch.long, device=points.device
    )
    minimum_distance = torch.full(
        (batch_size, point_count), float("inf"), device=points.device
    )
    batch_indices = torch.arange(batch_size, device=points.device)
    for sample_index in range(sample_count):
        indices[:, sample_index] = farthest
        selected = points[batch_indices, farthest]
        distance = (points - selected.unsqueeze(1)).square().sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        farthest = minimum_distance.masked_fill(~mask, -1.0).argmax(dim=1)

    sampled_mask = (
        torch.arange(sample_count, device=points.device).unsqueeze(0)
        < valid_counts.unsqueeze(1).clamp_max(sample_count)
    )
    return indices, sampled_mask


def query_ball_point(
    radius: float,
    neighbor_count: int,
    points: torch.Tensor,
    centroids: torch.Tensor,
    *,
    point_mask: torch.Tensor | None = None,
    centroid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find nearest points inside each radius, returning indices and a mask."""

    if radius <= 0.0 or neighbor_count < 1:
        raise ValueError("radius and neighbor_count must be positive.")
    batch_size, point_count, _ = points.shape
    centroid_count = centroids.shape[1]
    if point_mask is None:
        point_mask = torch.ones(
            (batch_size, point_count), dtype=torch.bool, device=points.device
        )
    if centroid_mask is None:
        centroid_mask = torch.ones(
            (batch_size, centroid_count), dtype=torch.bool, device=points.device
        )

    distances = square_distance(centroids, points)
    valid = (
        point_mask.unsqueeze(1)
        & centroid_mask.unsqueeze(-1)
        & (distances <= radius * radius)
    )
    distances = distances.masked_fill(~valid, float("inf"))
    selected_count = min(neighbor_count, point_count)
    selected_distances, indices = distances.topk(
        selected_count, dim=-1, largest=False, sorted=True
    )
    selected_mask = torch.isfinite(selected_distances)

    # Invalid slots repeat the closest valid point; their mask still excludes
    # them from local max pooling.
    fallback = indices[..., :1]
    indices = torch.where(selected_mask, indices, fallback.expand_as(indices))
    if selected_count < neighbor_count:
        padding = neighbor_count - selected_count
        indices = torch.cat((indices, fallback.expand(-1, -1, padding)), dim=-1)
        selected_mask = torch.cat(
            (
                selected_mask,
                torch.zeros(
                    (batch_size, centroid_count, padding),
                    dtype=torch.bool,
                    device=points.device,
                ),
            ),
            dim=-1,
        )
    return indices, selected_mask


class SharedPointMLP2d(nn.Module):
    """Shared MLP over every point in every local neighborhood."""

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            layers.extend(
                (
                    nn.Conv2d(input_channels, output_channels, 1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class SetAbstraction(nn.Module):
    """PointNet++ sampling, local grouping, PointNet, and max aggregation."""

    def __init__(
        self,
        sample_count: int | None,
        radius: float | None,
        neighbor_count: int | None,
        feature_channels: int,
        mlp_channels: Sequence[int],
    ):
        super().__init__()
        self.sample_count = sample_count
        self.radius = radius
        self.neighbor_count = neighbor_count
        self.group_all = sample_count is None
        self.mlp = SharedPointMLP2d((feature_channels + 3, *mlp_channels))

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.group_all:
            centroids = torch.zeros(
                (points.shape[0], 1, 3), dtype=points.dtype, device=points.device
            )
            grouped_points = points.unsqueeze(1)
            grouped_mask = mask.unsqueeze(1)
            centroid_mask = torch.ones(
                (points.shape[0], 1), dtype=torch.bool, device=points.device
            )
        else:
            assert self.sample_count is not None
            assert self.radius is not None
            assert self.neighbor_count is not None
            effective_count = min(self.sample_count, points.shape[1])
            centroid_indices, centroid_mask = farthest_point_sample(
                points, effective_count, mask
            )
            centroids = index_points(points, centroid_indices)
            group_indices, grouped_mask = query_ball_point(
                self.radius,
                self.neighbor_count,
                points,
                centroids,
                point_mask=mask,
                centroid_mask=centroid_mask,
            )
            grouped_points = index_points(points, group_indices)

        relative_points = grouped_points - centroids.unsqueeze(2)
        grouped = relative_points
        if features is not None:
            if self.group_all:
                grouped_features = features.unsqueeze(1)
            else:
                grouped_features = index_points(features, group_indices)
            grouped = torch.cat((relative_points, grouped_features), dim=-1)

        encoded = self.mlp(grouped.permute(0, 3, 2, 1))
        encoded = encoded.masked_fill(
            ~grouped_mask.permute(0, 2, 1).unsqueeze(1),
            torch.finfo(encoded.dtype).min,
        )
        aggregated = encoded.max(dim=2).values.transpose(1, 2)
        aggregated = aggregated.masked_fill(~centroid_mask.unsqueeze(-1), 0.0)
        return centroids, aggregated, centroid_mask


class FeaturePropagation(nn.Module):
    """Three-neighbor interpolation followed by a shared point MLP."""

    def __init__(self, input_channels: int, mlp_channels: Sequence[int]):
        super().__init__()
        from models.pointnet.components import SharedPointMLP

        self.mlp = SharedPointMLP((input_channels, *mlp_channels))

    def forward(
        self,
        target_points: torch.Tensor,
        source_points: torch.Tensor,
        target_features: torch.Tensor | None,
        source_features: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        if source_points.shape[1] == 1:
            interpolated = source_features.expand(-1, target_points.shape[1], -1)
        else:
            distances = square_distance(target_points, source_points)
            distances = distances.masked_fill(~source_mask.unsqueeze(1), float("inf"))
            count = min(3, source_points.shape[1])
            distances, indices = distances.topk(count, dim=-1, largest=False)
            weights = 1.0 / distances.clamp_min(1e-10)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            neighbors = index_points(source_features, indices)
            interpolated = (neighbors * weights.unsqueeze(-1)).sum(dim=2)

        combined = interpolated
        if target_features is not None:
            combined = torch.cat((target_features, interpolated), dim=-1)
        propagated = self.mlp(combined.transpose(1, 2)).transpose(1, 2)
        return propagated.masked_fill(~target_mask.unsqueeze(-1), 0.0)


class PointNet2Encoder(nn.Module):
    """Single-scale PointNet++ hierarchy with two local and one global level."""

    def __init__(
        self,
        feature_dim: int,
        sample_counts: Sequence[int],
        radii: Sequence[float],
        neighbor_counts: Sequence[int],
        abstraction_channels: Sequence[Sequence[int]],
        global_dim: int,
    ):
        super().__init__()
        first_channels, second_channels = abstraction_channels
        self.sa1 = SetAbstraction(
            sample_counts[0], radii[0], neighbor_counts[0], feature_dim, first_channels
        )
        self.sa2 = SetAbstraction(
            sample_counts[1],
            radii[1],
            neighbor_counts[1],
            first_channels[-1],
            second_channels,
        )
        self.sa3 = SetAbstraction(
            None, None, None, second_channels[-1], (256, 512, global_dim)
        )

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        level1_points, level1_features, level1_mask = self.sa1(points, features, mask)
        level2_points, level2_features, level2_mask = self.sa2(
            level1_points, level1_features, level1_mask
        )
        level3_points, level3_features, level3_mask = self.sa3(
            level2_points, level2_features, level2_mask
        )
        global_feature = level3_features[:, 0]
        return (
            global_feature,
            (level1_points, level2_points, level3_points),
            (level1_features, level2_features, level3_features),
            (level1_mask, level2_mask, level3_mask),
        )


class PointNet2SegmentationHead(nn.Module):
    """PointNet++ feature-propagation decoder for point-wise outputs."""

    def __init__(
        self,
        input_dim: int,
        level1_dim: int,
        level2_dim: int,
        global_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.fp3 = FeaturePropagation(level2_dim + global_dim, (256, 256))
        self.fp2 = FeaturePropagation(level1_dim + 256, (256, 128))
        self.fp1 = FeaturePropagation(input_dim + 128, (128, 128, 128))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv1d(128, output_dim, 1)

    def forward(
        self,
        original_points: torch.Tensor,
        original_features: torch.Tensor | None,
        original_mask: torch.Tensor,
        level_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        level_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        level_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        level1_points, level2_points, level3_points = level_points
        level1_features, level2_features, level3_features = level_features
        level1_mask, level2_mask, level3_mask = level_masks
        decoded2 = self.fp3(
            level2_points,
            level3_points,
            level2_features,
            level3_features,
            level2_mask,
            level3_mask,
        )
        decoded1 = self.fp2(
            level1_points,
            level2_points,
            level1_features,
            decoded2,
            level1_mask,
            level2_mask,
        )
        original_skip = original_points
        if original_features is not None:
            original_skip = torch.cat((original_points, original_features), dim=-1)
        decoded0 = self.fp1(
            original_points,
            level1_points,
            original_skip,
            decoded1,
            original_mask,
            level1_mask,
        )
        prediction = self.output(self.dropout(decoded0.transpose(1, 2))).transpose(1, 2)
        return prediction.masked_fill(~original_mask.unsqueeze(-1), 0.0)
