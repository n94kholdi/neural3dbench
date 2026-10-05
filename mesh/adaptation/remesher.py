from __future__ import annotations

from typing import Mapping

import torch

from mesh.types import NodeMapping, TransferMode, TriangularMesh

from .base import BaseRemesher, RemeshResult
from .config import MeshAdaptationConfig
from .state_transfer import BarycentricStateTransfer


class TriangleCentroidRemesher(BaseRemesher):
    """Conforming local refinement by splitting marked triangles at centroids.

    A marked triangle is replaced by three triangles. No edge gets a hanging
    node, boundary edges are untouched, and child cells inherit their parent's
    region/material data. Coarsening and anisotropic edge operations are future
    backend work and are intentionally not advertised by the configuration.
    """

    def adapt(
        self,
        mesh: TriangularMesh,
        state: Mapping[str, torch.Tensor],
        indicators: torch.Tensor,
        config: MeshAdaptationConfig,
    ) -> RemeshResult:
        del state
        indicators = torch.as_tensor(indicators, device=mesh.triangles.device)
        if indicators.shape != (mesh.num_cells,):
            raise ValueError("indicators must have shape [num_cells].")
        candidates = torch.nonzero(
            (indicators > config.refinement_threshold)
            & (mesh.cell_levels < config.max_refinement_level),
            as_tuple=False,
        ).flatten()
        candidates = candidates[torch.argsort(indicators[candidates], descending=True)]

        capacity = candidates.numel()
        if config.max_nodes is not None:
            capacity = min(capacity, max(0, config.max_nodes - mesh.num_nodes))
        if config.max_elements is not None:
            capacity = min(capacity, max(0, (config.max_elements - mesh.num_cells) // 2))
        selected = candidates[:capacity]
        if selected.numel() == 0:
            return RemeshResult(
                mesh=mesh,
                mapping=self._identity_mapping(mesh),
                changed=False,
                metadata={"refined_cells": 0, "limited": candidates.numel() > 0},
            )

        selected_set = set(selected.tolist())
        new_vertices = [mesh.vertices]
        new_triangles: list[torch.Tensor] = []
        parent_cells: list[int] = []
        new_parent_rows: list[torch.Tensor] = []
        new_weight_rows: list[torch.Tensor] = []
        new_levels: list[int] = []
        next_node = mesh.num_nodes

        for cell_index, triangle in enumerate(mesh.triangles):
            if cell_index not in selected_set:
                new_triangles.append(triangle)
                parent_cells.append(cell_index)
                new_levels.append(int(mesh.cell_levels[cell_index]))
                continue
            centroid = mesh.vertices[triangle].mean(dim=0, keepdim=True)
            new_vertices.append(centroid)
            a, b, c = triangle.unbind()
            for child in (
                torch.stack((a, b, torch.as_tensor(next_node, device=triangle.device))),
                torch.stack((b, c, torch.as_tensor(next_node, device=triangle.device))),
                torch.stack((c, a, torch.as_tensor(next_node, device=triangle.device))),
            ):
                new_triangles.append(child)
                parent_cells.append(cell_index)
                new_levels.append(int(mesh.cell_levels[cell_index]) + 1)
            new_parent_rows.append(triangle)
            new_weight_rows.append(
                torch.full((3,), 1.0 / 3.0, dtype=mesh.vertices.dtype, device=mesh.vertices.device)
            )
            next_node += 1

        mapping = self._identity_mapping(mesh)
        mapping = NodeMapping(
            parent_indices=torch.cat(
                (mapping.parent_indices, torch.stack(new_parent_rows)), dim=0
            ),
            weights=torch.cat((mapping.weights, torch.stack(new_weight_rows)), dim=0),
        )
        parent_index = torch.tensor(parent_cells, dtype=torch.long, device=mesh.triangles.device)
        vertices = torch.cat(new_vertices, dim=0)
        triangles = torch.stack(new_triangles)
        transfer = BarycentricStateTransfer()
        placeholder = TriangularMesh(
            vertices=vertices,
            triangles=triangles,
            cell_data={key: values[parent_index.to(values.device)] for key, values in mesh.cell_data.items()},
            boundary_edges=mesh.boundary_edges,
            boundary_tags=mesh.boundary_tags,
            cell_levels=torch.tensor(new_levels, dtype=torch.long, device=mesh.cell_levels.device),
            metadata=dict(mesh.metadata),
        )
        node_data = transfer.transfer(
            mesh,
            placeholder,
            mesh.node_data,
            mapping,
            mesh.node_data_modes,
        )
        new_mesh = TriangularMesh(
            vertices=vertices,
            triangles=triangles,
            node_data=node_data,
            node_data_modes=dict(mesh.node_data_modes),
            cell_data=placeholder.cell_data,
            boundary_edges=mesh.boundary_edges,
            boundary_tags=mesh.boundary_tags,
            cell_levels=placeholder.cell_levels,
            metadata=dict(mesh.metadata),
        )
        return RemeshResult(
            mesh=new_mesh,
            mapping=mapping,
            changed=True,
            metadata={
                "refined_cells": int(selected.numel()),
                "limited": selected.numel() < candidates.numel(),
                "strategy": "triangle_centroid_split",
            },
        )

    @staticmethod
    def _identity_mapping(mesh: TriangularMesh) -> NodeMapping:
        indices = torch.arange(mesh.num_nodes, device=mesh.triangles.device)
        return NodeMapping(
            parent_indices=indices[:, None].expand(-1, 3).clone(),
            weights=torch.tensor(
                [1.0, 0.0, 0.0], dtype=mesh.vertices.dtype, device=mesh.vertices.device
            ).expand(mesh.num_nodes, -1).clone(),
        )

