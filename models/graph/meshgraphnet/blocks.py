from __future__ import annotations

import torch
from torch import nn

from ..layers import _get_activation


class MeshGraphMLP(nn.Module):
    """Canonical MeshGraphNet MLP: hidden layers, output layer, then optional norm."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        hidden_layers: int,
        activation: str,
        normalization: str | None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend(
                (nn.Linear(current_dim, hidden_dim), _get_activation(activation))
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

        normalization = (normalization or "none").lower()
        if normalization == "layer_norm":
            self.normalization = nn.LayerNorm(output_dim)
        elif normalization == "none":
            self.normalization = nn.Identity()
        else:
            raise ValueError(
                "MeshGraphMLP normalization must be 'layer_norm' or 'none'."
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.normalization(self.network(features))


class MeshGraphNetBlock(nn.Module):
    """Edge-first interaction-network block with receiver-sum aggregation."""

    def __init__(
        self,
        latent_dim: int,
        *,
        mlp_hidden_dim: int,
        mlp_layers: int,
        activation: str,
        normalization: str,
        residual: bool,
    ) -> None:
        super().__init__()
        self.edge_mlp = MeshGraphMLP(
            3 * latent_dim,
            latent_dim,
            hidden_dim=mlp_hidden_dim,
            hidden_layers=mlp_layers,
            activation=activation,
            normalization=normalization,
        )
        self.node_mlp = MeshGraphMLP(
            2 * latent_dim,
            latent_dim,
            hidden_dim=mlp_hidden_dim,
            hidden_layers=mlp_layers,
            activation=activation,
            normalization=normalization,
        )
        self.residual = residual

    def forward(
        self,
        node_latent: torch.Tensor,
        edge_latent: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        senders, receivers = edge_index
        edge_inputs = torch.cat(
            (node_latent[senders], node_latent[receivers], edge_latent), dim=-1
        )
        edge_update = self.edge_mlp(edge_inputs)

        aggregated = node_latent.new_zeros(node_latent.shape)
        aggregated.index_add_(0, receivers, edge_update)
        node_inputs = torch.cat((node_latent, aggregated), dim=-1)
        node_update = self.node_mlp(node_inputs)

        if self.residual:
            return node_latent + node_update, edge_latent + edge_update
        return node_update, edge_update
