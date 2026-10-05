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


class GATConv(nn.Module):
    """Multi-head graph attention over directed ``source -> target`` edges.

    Attention is normalized over all incoming edges for each target node. Heads
    are either concatenated or averaged, matching the original GAT formulation.
    """

    def __init__(
        self,
        in_features: int,
        out_features_per_head: int,
        *,
        num_heads: int = 1,
        concat_heads: bool = True,
        attention_dropout: float = 0.0,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_head = out_features_per_head
        self.num_heads = num_heads
        self.concat_heads = concat_heads

        self.linear = nn.Linear(
            in_features,
            num_heads * out_features_per_head,
            bias=False,
        )
        self.attention_source = nn.Parameter(torch.empty(num_heads, out_features_per_head))
        self.attention_target = nn.Parameter(torch.empty(num_heads, out_features_per_head))
        output_dim = num_heads * out_features_per_head if concat_heads else out_features_per_head
        self.bias = nn.Parameter(torch.empty(output_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        if self.concat_heads:
            return self.num_heads * self.out_features_per_head
        return self.out_features_per_head

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)
        nn.init.zeros_(self.bias)

    @staticmethod
    def _incoming_softmax(
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Apply a stable softmax independently per target node and head."""
        maxima = logits.new_full((num_nodes, logits.size(1)), -torch.inf)
        maxima.scatter_reduce_(
            0,
            targets[:, None].expand_as(logits),
            logits,
            reduce="amax",
            include_self=True,
        )
        exponentials = torch.exp(logits - maxima[targets])
        denominators = logits.new_zeros((num_nodes, logits.size(1)))
        denominators.index_add_(0, targets, exponentials)
        return exponentials / denominators[targets].clamp_min(torch.finfo(logits.dtype).tiny)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(f"Expected node features [N, F], got {tuple(x.shape)}.")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")

        projected = self.linear(x).view(
            x.size(0), self.num_heads, self.out_features_per_head
        )
        source, target = edge_index
        if source.numel() == 0:
            aggregated = projected.new_zeros(projected.shape)
            attention = projected.new_empty((0, self.num_heads))
        else:
            source_scores = (projected * self.attention_source).sum(dim=-1)
            target_scores = (projected * self.attention_target).sum(dim=-1)
            logits = self.leaky_relu(source_scores[source] + target_scores[target])
            attention = self._incoming_softmax(logits, target, x.size(0))
            dropped_attention = self.attention_dropout(attention)
            messages = projected[source] * dropped_attention.unsqueeze(-1)
            aggregated = projected.new_zeros(projected.shape)
            aggregated.index_add_(0, target, messages)

        if self.concat_heads:
            output = aggregated.reshape(x.size(0), self.output_dim)
        else:
            output = aggregated.mean(dim=1)
        output = output + self.bias

        if return_attention_weights:
            return output, attention
        return output


class GATBlock(nn.Module):
    """Graph attention followed by normalization, activation, and dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_heads: int = 1,
        concat_heads: bool = True,
        activation: str = "elu",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        normalization: str | None = None,
        residual: bool = False,
    ):
        super().__init__()
        if concat_heads:
            if out_features % num_heads != 0:
                raise ValueError(
                    "out_features must be divisible by num_heads when concat_heads=True."
                )
            features_per_head = out_features // num_heads
        else:
            features_per_head = out_features

        self.conv = GATConv(
            in_features,
            features_per_head,
            num_heads=num_heads,
            concat_heads=concat_heads,
            attention_dropout=attention_dropout,
        )
        self.activation = _get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.normalization = normalization.lower() if normalization is not None else None
        if self.normalization == "layer_norm":
            self.norm = nn.LayerNorm(out_features)
        elif self.normalization == "batch_norm":
            self.norm = nn.BatchNorm1d(out_features)
        else:
            self.norm = None

        if not residual:
            self.residual_projection = None
        elif in_features == out_features:
            self.residual_projection = nn.Identity()
        else:
            self.residual_projection = nn.Linear(in_features, out_features, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        residual = x
        result = self.conv(
            x,
            edge_index,
            return_attention_weights=return_attention_weights,
        )
        if return_attention_weights:
            x, attention = result
        else:
            x = result

        if self.norm is not None:
            x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.residual_projection is not None:
            x = x + self.residual_projection(residual)

        if return_attention_weights:
            return x, attention
        return x
