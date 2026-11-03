from __future__ import annotations

import torch
from torch import nn

from models.pointnet2.components import index_points, square_distance


def knn(
    features: torch.Tensor, neighbor_count: int, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest valid neighbors in feature space, including the point itself."""

    distances = square_distance(features, features)
    distances = distances.masked_fill(~mask.unsqueeze(1), float("inf"))
    selected_distances, indices = distances.topk(
        min(neighbor_count, features.shape[1]), dim=-1, largest=False
    )
    return indices, torch.isfinite(selected_distances)


class EdgeConv(nn.Module):
    """Dynamic EdgeConv: shared MLP on ``[x_j - x_i, x_i]`` with max aggregation."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        neighbor_count: int,
        *,
        normalization: str = "batch",
    ):
        super().__init__()
        self.neighbor_count = neighbor_count
        layers: list[nn.Module] = [
            nn.Conv2d(
                2 * input_channels, output_channels, 1, bias=normalization == "none"
            )
        ]
        if normalization == "batch":
            layers.append(nn.BatchNorm2d(output_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        indices, neighbor_mask = knn(features, self.neighbor_count, mask)
        neighbors = index_points(features, indices)
        centers = features.unsqueeze(2).expand_as(neighbors)
        edges = torch.cat((neighbors - centers, centers), dim=-1).permute(0, 3, 1, 2)
        encoded = self.mlp(edges).masked_fill(
            ~neighbor_mask.unsqueeze(1), torch.finfo(edges.dtype).min
        )
        aggregated = encoded.max(dim=-1).values.transpose(1, 2)
        return aggregated.masked_fill(~mask.unsqueeze(-1), 0.0)
