from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class TransferMode(str, Enum):
    """How values associated with mesh nodes are transferred."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BOUNDARY = "boundary"


@dataclass
class TriangularMesh:
    """A validated, dependency-free 2D triangular mesh.

    Boundary conditions live on explicit boundary edges. Material and region
    identifiers should live in ``cell_data`` so refinement can inherit them
    without interpolating across interfaces.
    """

    vertices: torch.Tensor
    triangles: torch.Tensor
    node_data: dict[str, torch.Tensor] = field(default_factory=dict)
    cell_data: dict[str, torch.Tensor] = field(default_factory=dict)
    node_data_modes: dict[str, TransferMode] = field(default_factory=dict)
    boundary_edges: torch.Tensor | None = None
    boundary_tags: torch.Tensor | None = None
    cell_levels: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = torch.as_tensor(self.vertices)
        self.triangles = torch.as_tensor(self.triangles, dtype=torch.long)
        if self.boundary_edges is not None:
            self.boundary_edges = torch.as_tensor(self.boundary_edges, dtype=torch.long)
        if self.boundary_tags is not None:
            self.boundary_tags = torch.as_tensor(self.boundary_tags)
        if self.cell_levels is None:
            self.cell_levels = torch.zeros(self.num_cells, dtype=torch.long)
        else:
            self.cell_levels = torch.as_tensor(self.cell_levels, dtype=torch.long)
        self.node_data = {key: torch.as_tensor(value) for key, value in self.node_data.items()}
        self.cell_data = {key: torch.as_tensor(value) for key, value in self.cell_data.items()}
        self.node_data_modes = {
            key: mode if isinstance(mode, TransferMode) else TransferMode(mode)
            for key, mode in self.node_data_modes.items()
        }
        self.validate()

    @property
    def num_nodes(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_cells(self) -> int:
        return int(self.triangles.shape[0])

    def validate(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("vertices must have shape [N, 2].")
        if not self.vertices.is_floating_point():
            raise TypeError("vertices must use a floating-point dtype.")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangles must have shape [T, 3].")
        if self.num_nodes < 3 or self.num_cells < 1:
            raise ValueError("a triangular mesh needs at least three nodes and one cell.")
        if self.triangles.min() < 0 or self.triangles.max() >= self.num_nodes:
            raise ValueError("triangle indices are outside the vertex range.")
        if (self.triangles.sort(dim=1).values.diff(dim=1) == 0).any():
            raise ValueError("triangles may not contain repeated vertices.")
        points = self.vertices[self.triangles]
        twice_area = (
            (points[:, 1, 0] - points[:, 0, 0])
            * (points[:, 2, 1] - points[:, 0, 1])
            - (points[:, 1, 1] - points[:, 0, 1])
            * (points[:, 2, 0] - points[:, 0, 0])
        )
        if torch.any(twice_area.abs() <= torch.finfo(self.vertices.dtype).eps):
            raise ValueError("mesh contains a degenerate triangle.")
        if self.cell_levels is None or self.cell_levels.shape != (self.num_cells,):
            raise ValueError("cell_levels must have shape [T].")
        for name, values in self.node_data.items():
            if values.shape[0] != self.num_nodes:
                raise ValueError(f"node_data['{name}'] must have N entries.")
        for name, values in self.cell_data.items():
            if values.shape[0] != self.num_cells:
                raise ValueError(f"cell_data['{name}'] must have T entries.")
        if self.boundary_edges is not None:
            if self.boundary_edges.ndim != 2 or self.boundary_edges.shape[1] != 2:
                raise ValueError("boundary_edges must have shape [B, 2].")
            if self.boundary_edges.numel() and (
                self.boundary_edges.min() < 0 or self.boundary_edges.max() >= self.num_nodes
            ):
                raise ValueError("boundary edge indices are outside the vertex range.")
            mesh_edges = self.undirected_edges()
            mesh_keys = {tuple(edge.tolist()) for edge in mesh_edges.cpu()}
            for edge in self.boundary_edges.sort(dim=1).values.cpu():
                if tuple(edge.tolist()) not in mesh_keys:
                    raise ValueError("every boundary edge must be a triangle edge.")
            if self.boundary_tags is not None and self.boundary_tags.shape[0] != self.boundary_edges.shape[0]:
                raise ValueError("boundary_tags must have one entry per boundary edge.")
        elif self.boundary_tags is not None:
            raise ValueError("boundary_tags require boundary_edges.")

    def undirected_edges(self) -> torch.Tensor:
        tri = self.triangles
        edges = torch.cat((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]), dim=0)
        return torch.unique(edges.sort(dim=1).values, dim=0)


@dataclass(frozen=True)
class NodeMapping:
    """Barycentric correspondence from each new node to old nodes."""

    parent_indices: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        if self.parent_indices.ndim != 2 or self.weights.shape != self.parent_indices.shape:
            raise ValueError("parent_indices and weights must have equal [N_new, K] shape.")
        if not torch.allclose(
            self.weights.sum(dim=1),
            torch.ones(self.weights.shape[0], dtype=self.weights.dtype, device=self.weights.device),
        ):
            raise ValueError("mapping weights must sum to one for each new node.")

