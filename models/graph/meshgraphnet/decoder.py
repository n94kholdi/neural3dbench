from __future__ import annotations

from torch import nn

from .blocks import MeshGraphMLP


class MeshGraphNetDecoder(nn.Module):
    """Decode final node latents into task-neutral per-node predictions."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        *,
        mlp_hidden_dim: int,
        mlp_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.mlp = MeshGraphMLP(
            latent_dim,
            output_dim,
            hidden_dim=mlp_hidden_dim,
            hidden_layers=mlp_layers,
            activation=activation,
            normalization="none",
        )

    def forward(self, node_latent):
        return self.mlp(node_latent)
