from __future__ import annotations

import torch
from torch import nn


def activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation '{name}'.") from exc


class InputEmbedding(nn.Module):
    """Concatenate coordinates/features and apply the official two-layer MLP."""

    def __init__(self, spatial_dim: int, input_dim: int, hidden_dim: int, act: str) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(spatial_dim + input_dim, hidden_dim * 2),
            activation(act),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self, coordinates: torch.Tensor, features: torch.Tensor | None = None
    ) -> torch.Tensor:
        combined = coordinates if features is None else torch.cat((coordinates, features), dim=-1)
        return self.network(combined)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, act: str) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim), activation(act), nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class OutputHead(nn.Module):
    """The final LayerNorm and linear field projection used by Transolver."""

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(x))
