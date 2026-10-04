from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.configs import BaseModelConfig
from models.types import ModelInput, ModelOutput

from .base import BaseGraphNetwork
from .layers import GCNBlock


@dataclass
class GCNConfig(BaseModelConfig):
    input_dim: int = 1
    output_dim: int = 1
    hidden_dim: int = 64
    num_layers: int = 2
    activation: str = "relu"
    dropout: float = 0.0
    normalization: str = "none"
    residual: bool = True
    add_self_loops: bool = True
    representation: str = "GRAPH"

    def __post_init__(self) -> None:
        self.name = "gcn"
        self.model_type = "gcn"
        self.representation = "GRAPH"
        self._validate()

    def _validate(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        valid_activations = {"relu", "gelu", "silu", "elu", "tanh", "identity", "leaky_relu"}
        if self.activation.lower() not in valid_activations:
            raise ValueError(f"Unknown activation '{self.activation}'. Available: {sorted(valid_activations)}")
        valid_normalizations = {"none", "layer_norm", "batch_norm", "instance_norm"}
        if self.normalization.lower() not in valid_normalizations:
            raise ValueError(f"Unknown normalization '{self.normalization}'. Available: {sorted(valid_normalizations)}")


class GCN(BaseGraphNetwork):
    config_class = GCNConfig

    def __init__(self, config: GCNConfig | None = None):
        super().__init__(config=config or GCNConfig())

        self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)

        layer_dims = [self.config.hidden_dim] * self.config.num_layers
        self.layers = nn.ModuleList()
        for idx, out_dim in enumerate(layer_dims):
            in_dim = self.config.hidden_dim if idx > 0 else self.config.hidden_dim
            self.layers.append(
                GCNBlock(
                    in_dim,
                    out_dim,
                    activation=self.config.activation,
                    dropout=self.config.dropout,
                    normalization=self.config.normalization if idx < self.config.num_layers - 1 else None,
                    residual=self.config.residual,
                )
            )

        self.output_proj = nn.Linear(self.config.hidden_dim, self.config.output_dim)

    @property
    def required_inputs(self) -> set[str]:
        return {"node_features", "edge_index"}

    def validate_inputs(self, inputs: ModelInput) -> bool:
        super().validate_inputs(inputs)

        node_features = self._ensure_tensor(inputs.node_features, name="node_features")
        if node_features.shape[1] != self.config.input_dim:
            raise ValueError(
                f"GCN expects input feature dimension {self.config.input_dim}, got {node_features.shape[1]}."
            )

        edge_index = self._ensure_tensor(inputs.edge_index, name="edge_index")
        if edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")

        return True

    def _prepare_edge_index(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=self.input_proj.weight.device)
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")

        if self.config.add_self_loops:
            self_loops = torch.arange(num_nodes, device=edge_index.device, dtype=edge_index.dtype).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)

        return edge_index

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)

        node_features = self._ensure_tensor(inputs.node_features, name="node_features").to(self.input_proj.weight.device)
        edge_index = self._prepare_edge_index(inputs.edge_index, node_features.size(0))

        hidden = self.input_proj(node_features)
        for layer in self.layers:
            hidden = layer(hidden, edge_index)
        predictions = self.output_proj(hidden)
        return ModelOutput(predictions=predictions)
