from __future__ import annotations

from abc import ABC

import torch

from models.base import BaseNetwork
from models.types import ModelCapabilities


class BaseGraphNetwork(BaseNetwork, ABC):
    """Shared validation and capability metadata for graph-based models."""

    @property
    def required_inputs(self) -> set[str]:
        return {"node_features", "edge_index"}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            representation="GRAPH",
            requires=set(self.required_inputs),
            supports={"batch", "edge_features", "coordinates"},
            variable_geometry=True,
            variable_node_count=True,
        )

    def validate_inputs(self, inputs) -> bool:
        super().validate_inputs(inputs)

        node_features = self._ensure_tensor(inputs.node_features, name="node_features")
        if node_features.dim() != 2:
            raise ValueError(f"node_features must have shape [N, F], got {tuple(node_features.shape)}.")
        if node_features.size(0) == 0:
            raise ValueError("node_features must contain at least one node.")

        edge_index = self._ensure_tensor(inputs.edge_index, name="edge_index")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
        if edge_index.numel() > 0:
            if edge_index.min() < 0 or edge_index.max() >= node_features.size(0):
                raise ValueError(
                    f"edge_index entries must be in the range [0, {node_features.size(0) - 1}] for N={node_features.size(0)}."
                )

        if inputs.batch is not None:
            batch = self._ensure_tensor(inputs.batch, name="batch")
            if batch.dim() != 1 or batch.numel() != node_features.size(0):
                raise ValueError(f"batch must have shape [N], got {tuple(batch.shape)} for N={node_features.size(0)}.")

        return True
