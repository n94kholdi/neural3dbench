from __future__ import annotations

import torch
from torch import nn


def _get_activation(name: str | None) -> nn.Module:
    name = (name or "relu").lower()
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "elu": nn.ELU(),
        "tanh": nn.Tanh(),
        "identity": nn.Identity(),
        "leaky_relu": nn.LeakyReLU(0.01),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Available: {sorted(activations)}")
    return activations[name]


class GCNConv(nn.Module):
    """A lightweight GCN-style message-passing layer using explicit edge_index tensors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected node features [N, F], got {tuple(x.shape)}.")
        if edge_index.numel() == 0:
            return self.linear(x)

        source, target = edge_index
        if source.numel() == 0:
            return self.linear(x)

        source_degree = torch.bincount(source, minlength=x.size(0)).clamp_min(1).to(x.dtype)
        target_degree = torch.bincount(target, minlength=x.size(0)).clamp_min(1).to(x.dtype)

        messages = self.linear(x[source])
        incoming = messages * source_degree[source].unsqueeze(-1).pow(-0.5) * target_degree[target].unsqueeze(-1).pow(-0.5)

        aggregated = torch.zeros(x.size(0), messages.size(-1), device=x.device, dtype=x.dtype)
        aggregated.index_add_(0, target, incoming)
        return aggregated


class GCNBlock(nn.Module):
    """Reusable block combining graph convolution, normalization, activation, and dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        normalization: str | None = None,
        residual: bool = False,
    ):
        super().__init__()
        self.conv = GCNConv(in_features, out_features)
        self.activation = _get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual and in_features == out_features
        self.normalization = normalization.lower() if normalization is not None else None

        if self.normalization == "layer_norm":
            self.norm = nn.LayerNorm(out_features)
        elif self.normalization == "batch_norm":
            self.norm = nn.BatchNorm1d(out_features)
        elif self.normalization == "instance_norm":
            self.norm = nn.InstanceNorm1d(out_features)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x, edge_index)
        if self.norm is not None:
            x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.residual and residual.size(-1) == x.size(-1):
            x = x + residual
        return x
