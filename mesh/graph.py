from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from models.types import ModelInput

from .types import TriangularMesh


class TriangleMeshGraphBuilder:
    """Rebuild a bidirectional graph and geometric features from a mesh."""

    def __init__(
        self,
        node_feature_fields: Sequence[str],
        *,
        include_coordinates: bool = False,
    ) -> None:
        if not node_feature_fields and not include_coordinates:
            raise ValueError("at least one node feature source is required.")
        self.node_feature_fields = tuple(node_feature_fields)
        self.include_coordinates = include_coordinates

    def __call__(
        self, mesh: TriangularMesh, state: Mapping[str, torch.Tensor]
    ) -> ModelInput:
        features: list[torch.Tensor] = []
        for name in self.node_feature_fields:
            if name in state:
                value = torch.as_tensor(state[name], device=mesh.vertices.device)
            elif name in mesh.node_data:
                value = mesh.node_data[name].to(mesh.vertices.device)
            else:
                raise KeyError(f"node feature '{name}' is absent from state and mesh data.")
            if value.shape[0] != mesh.num_nodes:
                raise ValueError(f"node feature '{name}' must have N entries.")
            if value.ndim == 1:
                value = value[:, None]
            features.append(value.to(dtype=mesh.vertices.dtype))
        if self.include_coordinates:
            features.append(mesh.vertices)

        undirected = mesh.undirected_edges()
        directed = torch.cat((undirected, undirected.flip(1)), dim=0)
        edge_index = directed.t().contiguous()
        relative = mesh.vertices[edge_index[1]] - mesh.vertices[edge_index[0]]
        distance = torch.linalg.vector_norm(relative, dim=1, keepdim=True)
        edge_features = torch.cat((relative, distance), dim=1)
        return ModelInput(
            coordinates=mesh.vertices,
            node_features=torch.cat(features, dim=1),
            edge_index=edge_index,
            edge_features=edge_features,
            batch=torch.zeros(mesh.num_nodes, dtype=torch.long, device=mesh.vertices.device),
            metadata={"mesh": mesh, "num_cells": mesh.num_cells},
            representation="GRAPH",
        )

    def to_pyg_data(
        self, mesh: TriangularMesh, state: Mapping[str, torch.Tensor]
    ) -> Any:
        """Build optional PyG ``Data`` without making PyG a core dependency."""
        try:
            from torch_geometric.data import Data
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "PyTorch Geometric is optional; install the 'pyg' extra to use Data."
            ) from error
        graph = self(mesh, state)
        return Data(
            x=graph.node_features,
            pos=graph.coordinates,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_features,
        )

