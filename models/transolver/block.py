from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .components import FeedForward
from .physics_attention import PhysicsAttention


class TransolverBlock(nn.Module):
    """Pre-normalized residual Physics-Attention and feed-forward block."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        num_slices: int,
        mlp_ratio: float,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = PhysicsAttention(
            hidden_dim,
            num_heads=num_heads,
            num_slices=num_slices,
            dropout=dropout,
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = FeedForward(
            hidden_dim, int(hidden_dim * mlp_ratio), activation
        )

    def forward(
        self,
        x: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        result = self.attention(
            self.attention_norm(x), point_mask, return_attention=return_attention
        )
        if return_attention:
            attended, auxiliary = result
        else:
            attended, auxiliary = result, None
        x = x + attended
        x = x + self.feed_forward(self.feed_forward_norm(x))
        if point_mask is not None:
            x = x.masked_fill(~point_mask.unsqueeze(-1), 0.0)
        return (x, auxiliary) if return_attention else x
