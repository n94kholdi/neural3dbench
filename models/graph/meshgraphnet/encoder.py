from __future__ import annotations

import torch
from torch import nn

from .blocks import MeshGraphMLP


class MeshGraphNetEncoder(nn.Module):
    """Encode raw node and edge attributes with separate MLPs."""

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        latent_dim: int,
        *,
        mlp_hidden_dim: int,
        mlp_layers: int,
        activation: str,
        normalization: str,
    ) -> None:
        super().__init__()
        common = {
            "hidden_dim": mlp_hidden_dim,
            "hidden_layers": mlp_layers,
            "activation": activation,
            "normalization": normalization,
        }
        self.node_encoder = MeshGraphMLP(
            node_input_dim, latent_dim, **common
        )
        self.edge_encoder = MeshGraphMLP(
            edge_input_dim, latent_dim, **common
        )

    def forward(
        self, node_features: torch.Tensor, edge_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.node_encoder(node_features), self.edge_encoder(edge_features)
