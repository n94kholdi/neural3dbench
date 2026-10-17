from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PhysicsAttention(nn.Module):
    """Physics-Attention for irregular meshes from the original Transolver.

    ``N`` point features are softly assigned to ``G`` learned physical slices
    independently in each of ``H`` heads. Attention is evaluated between the
    resulting slice tokens, not between the original points.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 8,
        num_slices: int = 32,
        dropout: float = 0.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = dim
        self.num_heads = num_heads
        self.num_slices = num_slices
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.eps = eps

        self.in_project_x = nn.Linear(dim, dim)
        self.in_project_fx = nn.Linear(dim, dim)
        self.in_project_slice = nn.Linear(self.head_dim, num_slices)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.temperature = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.5))
        self.to_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [B, N, {self.dim}].")
        batch_size, point_count, _ = x.shape
        if point_mask is not None:
            if point_mask.shape != x.shape[:2] or point_mask.dtype != torch.bool:
                raise ValueError("point_mask must be boolean with shape [B, N].")
            valid = point_mask[:, None, :, None]
        else:
            valid = None

        # [B,N,C] -> [B,H,N,D]. Separate projections reproduce the official
        # implementation: x_mid assigns slices; fx_mid supplies token values.
        x_mid = self.in_project_x(x).reshape(
            batch_size, point_count, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        fx_mid = self.in_project_fx(x).reshape(
            batch_size, point_count, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        # Softmax is over G: every valid point distributes unit mass over slices.
        slice_logits = self.in_project_slice(x_mid) / self.temperature
        slice_weights = slice_logits.softmax(dim=-1)  # [B,H,N,G]
        if valid is not None:
            slice_weights = slice_weights * valid
            fx_mid = fx_mid * valid

        # Weighted spatial aggregation and per-slice normalization.
        slice_norm = slice_weights.sum(dim=2)  # [B,H,G]
        slice_tokens = torch.einsum(
            "bhnd,bhng->bhgd", fx_mid, slice_weights
        ) / slice_norm.clamp_min(self.eps).unsqueeze(-1)  # [B,H,G,D]

        # Canonical attention is applied to G physical tokens, never N points.
        queries = self.to_q(slice_tokens)
        keys = self.to_k(slice_tokens)
        values = self.to_v(slice_tokens)
        attention = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        attention = self.attention_dropout(attention.softmax(dim=-1))  # [B,H,G,G]
        output_tokens = torch.matmul(attention, values)  # [B,H,G,D]

        # Deslice with the same point-to-slice weights: [B,H,N,D] -> [B,N,C].
        output = torch.einsum(
            "bhgd,bhng->bhnd", output_tokens, slice_weights
        ).permute(0, 2, 1, 3).reshape(batch_size, point_count, self.dim)
        output = self.to_out(output)
        if point_mask is not None:
            output = output.masked_fill(~point_mask.unsqueeze(-1), 0.0)

        if not return_attention:
            return output
        return output, {
            "slice_weights": slice_weights,
            "slice_norm": slice_norm,
            "slice_tokens": slice_tokens,
            "attention_weights": attention,
            "output_tokens": output_tokens,
        }
