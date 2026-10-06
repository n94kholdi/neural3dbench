from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def feature_transform_regularizer(transform: torch.Tensor) -> torch.Tensor:
    """Orthogonality regularizer proposed for PointNet feature transforms."""

    if transform.ndim != 3 or transform.shape[1] != transform.shape[2]:
        raise ValueError("transform must have shape [B, K, K].")
    identity = torch.eye(
        transform.shape[1], device=transform.device, dtype=transform.dtype
    ).unsqueeze(0)
    residual = torch.bmm(transform, transform.transpose(1, 2)) - identity
    return torch.linalg.matrix_norm(residual, ord="fro", dim=(-2, -1)).mean()


def masked_max(features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Max-pool ``[B, C, N]`` features while ignoring padded points."""

    if mask is None:
        return features.max(dim=2).values
    valid = mask.unsqueeze(1)
    masked = features.masked_fill(~valid, torch.finfo(features.dtype).min)
    return masked.max(dim=2).values


class SharedPointMLP(nn.Module):
    """A sequence of 1x1 convolutions shared by every point."""

    def __init__(self, channels: Sequence[int], *, final_activation: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        for index, (in_channels, out_channels) in enumerate(
            zip(channels[:-1], channels[1:])
        ):
            is_last = index == len(channels) - 2
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False))
            if final_activation or not is_last:
                layers.extend((nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class TransformNet(nn.Module):
    """PointNet T-Net predicting a data-dependent square transformation."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.point_mlp = SharedPointMLP((feature_dim, 64, 128, 1024))
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, feature_dim * feature_dim)
        self.activation = nn.ReLU(inplace=True)

        # The residual identity makes the initial transform stable.
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = masked_max(self.point_mlp(features), mask)
        hidden = self.activation(self.bn1(self.fc1(hidden)))
        hidden = self.activation(self.bn2(self.fc2(hidden)))
        transform = self.fc3(hidden).view(-1, self.feature_dim, self.feature_dim)
        identity = torch.eye(
            self.feature_dim, device=features.device, dtype=features.dtype
        ).unsqueeze(0)
        return transform + identity


class PointNetEncoder(nn.Module):
    """Standard PointNet encoder with optional input and feature T-Nets."""

    def __init__(
        self,
        input_dim: int,
        global_dim: int = 1024,
        *,
        input_transform: bool = True,
        feature_transform: bool = True,
    ):
        super().__init__()
        self.input_transform = TransformNet(3) if input_transform else None
        self.first_mlp = SharedPointMLP((input_dim, 64, 64))
        self.feature_transform = TransformNet(64) if feature_transform else None
        self.second_mlp = SharedPointMLP((64, 64, 128, global_dim))

    def forward(
        self,
        points: torch.Tensor,
        point_features: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        input_matrix = None
        transformed_points = points
        if self.input_transform is not None:
            input_matrix = self.input_transform(points.transpose(1, 2), mask)
            transformed_points = torch.bmm(points, input_matrix)

        combined = transformed_points
        if point_features is not None:
            combined = torch.cat((combined, point_features), dim=-1)
        local = self.first_mlp(combined.transpose(1, 2))

        feature_matrix = None
        if self.feature_transform is not None:
            feature_matrix = self.feature_transform(local, mask)
            local = torch.bmm(local.transpose(1, 2), feature_matrix).transpose(1, 2)

        encoded = self.second_mlp(local)
        global_feature = masked_max(encoded, mask)
        return local, global_feature, {
            "input_transform": input_matrix,
            "feature_transform": feature_matrix,
        }


class PointNetClassificationHead(nn.Module):
    def __init__(self, global_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(global_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, global_feature: torch.Tensor) -> torch.Tensor:
        return self.layers(global_feature)


class PointNetSegmentationHead(nn.Module):
    def __init__(self, global_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.point_mlp = SharedPointMLP((global_dim + 64, 512, 256, 128))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv1d(128, output_dim, kernel_size=1)

    def forward(
        self, local: torch.Tensor, global_feature: torch.Tensor
    ) -> torch.Tensor:
        point_count = local.shape[2]
        global_expanded = global_feature.unsqueeze(2).expand(-1, -1, point_count)
        hidden = self.point_mlp(torch.cat((local, global_expanded), dim=1))
        return self.output(self.dropout(hidden)).transpose(1, 2)
