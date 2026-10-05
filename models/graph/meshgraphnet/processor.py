from __future__ import annotations

import torch
from torch import nn

from .blocks import MeshGraphNetBlock


class MeshGraphNetProcessor(nn.Module):
    """A stack of independently parameterized message-passing blocks."""

    def __init__(
        self,
        latent_dim: int,
        processor_steps: int,
        *,
        mlp_hidden_dim: int,
        mlp_layers: int,
        activation: str,
        normalization: str,
        residual: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            MeshGraphNetBlock(
                latent_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                mlp_layers=mlp_layers,
                activation=activation,
                normalization=normalization,
                residual=residual,
            )
            for _ in range(processor_steps)
        )

    def forward(
        self,
        node_latent: torch.Tensor,
        edge_latent: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            node_latent, edge_latent = block(
                node_latent, edge_latent, edge_index
            )
        return node_latent, edge_latent
