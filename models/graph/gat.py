from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.configs import BaseModelConfig
from models.types import ModelInput, ModelOutput

from .base import BaseGraphNetwork
from .layers import GATBlock


@dataclass
class GATConfig(BaseModelConfig):
    input_dim: int = 1
    output_dim: int = 1
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int | tuple[int, ...] = 4
    activation: str = "elu"
    dropout: float = 0.0
    attention_dropout: float = 0.0
    normalization: str = "none"
    residual: bool = True
    concat_heads: bool = True
    add_self_loops: bool = True
    return_attention_weights: bool = False
    representation: str = "GRAPH"

    def __post_init__(self) -> None:
        self.name = "gat"
        self.model_type = "gat"
        self.representation = "GRAPH"
        if isinstance(self.num_heads, list):
            self.num_heads = tuple(self.num_heads)
        self._validate()

    @property
    def layer_heads(self) -> tuple[int, ...]:
        if isinstance(self.num_heads, int):
            return (self.num_heads,) * self.num_layers
        return tuple(self.num_heads)

    def _validate(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        if not isinstance(self.num_heads, (int, tuple)):
            raise TypeError("num_heads must be an integer or a sequence of integers.")
        heads = self.layer_heads
        if len(heads) != self.num_layers:
            raise ValueError("A num_heads sequence must contain one value per GAT layer.")
        if any(not isinstance(head, int) or isinstance(head, bool) or head < 1 for head in heads):
            raise ValueError("Every num_heads value must be a positive integer.")
        if self.concat_heads and any(self.hidden_dim % head != 0 for head in heads):
            raise ValueError(
                "hidden_dim must be divisible by every num_heads value when concat_heads=True."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in the range [0, 1).")
        valid_activations = {"relu", "gelu", "silu", "elu", "tanh", "identity", "leaky_relu"}
        if self.activation.lower() not in valid_activations:
            raise ValueError(f"Unknown activation '{self.activation}'. Available: {sorted(valid_activations)}")
        valid_normalizations = {"none", "layer_norm", "batch_norm"}
        if self.normalization.lower() not in valid_normalizations:
            raise ValueError(
                f"Unknown normalization '{self.normalization}'. Available: {sorted(valid_normalizations)}"
            )


class GAT(BaseGraphNetwork):
    """Node-level Graph Attention Network with a fixed-width hidden state."""

    config_class = GATConfig

    def __init__(self, config: GATConfig | None = None):
        super().__init__(config=config or GATConfig())
        self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.layers = nn.ModuleList(
            GATBlock(
                self.config.hidden_dim,
                self.config.hidden_dim,
                num_heads=heads,
                concat_heads=self.config.concat_heads,
                activation=self.config.activation,
                dropout=self.config.dropout,
                attention_dropout=self.config.attention_dropout,
                normalization=(
                    self.config.normalization
                    if self.config.normalization.lower() != "none"
                    else None
                ),
                residual=self.config.residual,
            )
            for heads in self.config.layer_heads
        )
        self.output_proj = nn.Linear(self.config.hidden_dim, self.config.output_dim)

    def validate_inputs(self, inputs: ModelInput) -> bool:
        super().validate_inputs(inputs)
        node_features = self._ensure_tensor(inputs.node_features, name="node_features")
        if not node_features.is_floating_point():
            raise TypeError("node_features must use a floating-point dtype.")
        if node_features.shape[1] != self.config.input_dim:
            raise ValueError(
                f"GAT expects input feature dimension {self.config.input_dim}, "
                f"got {node_features.shape[1]}."
            )

        edge_index = self._ensure_tensor(inputs.edge_index, name="edge_index")
        if edge_index.dtype == torch.bool or edge_index.is_floating_point() or edge_index.is_complex():
            raise TypeError("edge_index must use an integer dtype.")
        return True

    def _prepare_edge_index(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        edge_index = torch.as_tensor(
            edge_index,
            dtype=torch.long,
            device=self.input_proj.weight.device,
        )
        if not self.config.add_self_loops:
            return edge_index

        # Replace all provided self-loops with exactly one loop per node. Other
        # directed edges, including duplicates, are kept as supplied.
        non_loop_edges = edge_index[:, edge_index[0] != edge_index[1]]
        nodes = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
        self_loops = torch.stack((nodes, nodes), dim=0)
        return torch.cat((non_loop_edges, self_loops), dim=1)

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        node_features = self._ensure_tensor(inputs.node_features, name="node_features").to(
            device=self.input_proj.weight.device,
            dtype=self.input_proj.weight.dtype,
        )
        edge_index = self._prepare_edge_index(inputs.edge_index, node_features.size(0))

        hidden = self.input_proj(node_features)
        attention_weights = [] if self.config.return_attention_weights else None
        for layer in self.layers:
            if attention_weights is None:
                hidden = layer(hidden, edge_index)
            else:
                hidden, coefficients = layer(
                    hidden,
                    edge_index,
                    return_attention_weights=True,
                )
                attention_weights.append(
                    {"edge_index": edge_index, "coefficients": coefficients}
                )

        predictions = self.output_proj(hidden)
        auxiliary = None
        if attention_weights is not None:
            auxiliary = {"attention_weights": attention_weights}
        metadata = {"batch": inputs.batch} if inputs.batch is not None else {}
        return ModelOutput(
            predictions=predictions,
            latent=hidden,
            auxiliary=auxiliary,
            metadata=metadata,
        )
