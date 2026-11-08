"""Shared triangle-mesh contract for models that read meshes directly (MeshNet, DiffusionNet, MeshWalker).

A batch of meshes is concatenated, like a batch of graphs:

- ``coordinates`` float ``[V, 3]``: vertices of all meshes.
- ``faces`` int ``[F, 3]``: triangles, indexing rows of ``coordinates``.
- ``batch`` int ``[V]``: mesh id of each vertex, sorted (``0, 0, ..., 1, 1, ...``); optional for one mesh.

``collate_meshes`` builds this from a list of ``(vertices, faces)`` pairs.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .types import ModelInput


def collate_meshes(meshes: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> ModelInput:
    """Concatenate ``(vertices [V_i, 3], faces [F_i, 3])`` pairs into one ``ModelInput``."""
    vertices, faces, batch, offset = [], [], [], 0
    for index, (verts, tris) in enumerate(meshes):
        verts = torch.as_tensor(verts, dtype=torch.float32)
        tris = torch.as_tensor(tris, dtype=torch.long)
        vertices.append(verts)
        faces.append(tris + offset)
        batch.append(torch.full((len(verts),), index, dtype=torch.long))
        offset += len(verts)
    return ModelInput(
        coordinates=torch.cat(vertices),
        faces=torch.cat(faces),
        batch=torch.cat(batch),
        representation="MESH",
    )


def validate_mesh_inputs(inputs: ModelInput, model_name: str) -> None:
    """Check the concatenated ``[V, 3]`` / ``[F, 3]`` / ``[V]`` mesh contract."""
    vertices = torch.as_tensor(inputs.coordinates)
    faces = torch.as_tensor(inputs.faces)
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"{model_name} coordinates must have shape [V, 3], got {tuple(vertices.shape)}.")
    if not vertices.is_floating_point():
        raise TypeError("coordinates must use a floating-point dtype.")
    if faces.ndim != 2 or faces.shape[-1] != 3 or len(faces) == 0:
        raise ValueError(f"{model_name} faces must have shape [F, 3] with F >= 1, got {tuple(faces.shape)}.")
    if faces.is_floating_point() or faces.dtype == torch.bool:
        raise TypeError("faces must use an integer dtype.")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces index outside the coordinates array.")
    if inputs.batch is not None:
        batch = torch.as_tensor(inputs.batch)
        if tuple(batch.shape) != (len(vertices),):
            raise ValueError(f"batch must have shape [{len(vertices)}], got {tuple(batch.shape)}.")
        if len(batch) and (batch[0] != 0 or (batch.diff() < 0).any() or (batch.diff() > 1).any()):
            raise ValueError("batch must be sorted mesh ids 0, 1, ... without gaps (see collate_meshes).")
        if (batch[faces[:, 0]] != batch[faces[:, 1]]).any() or (batch[faces[:, 0]] != batch[faces[:, 2]]).any():
            raise ValueError("a face connects vertices of different meshes.")


def prepare_mesh_inputs(
    inputs: ModelInput, parameter: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return ``(vertices, faces, batch, num_meshes)`` on the model device."""
    vertices = torch.as_tensor(inputs.coordinates, device=parameter.device, dtype=parameter.dtype)
    faces = torch.as_tensor(inputs.faces, device=parameter.device, dtype=torch.long)
    if inputs.batch is None:
        batch = torch.zeros(len(vertices), device=parameter.device, dtype=torch.long)
    else:
        batch = torch.as_tensor(inputs.batch, device=parameter.device, dtype=torch.long)
    return vertices, faces, batch, int(batch.max()) + 1 if len(batch) else 1


def face_neighbors(faces: torch.Tensor) -> torch.Tensor:
    """``[F, 3]`` index of the face across each edge (v0v1, v1v2, v2v0); the face itself on a boundary.

    On a non-manifold edge shared by more than two faces, faces are paired in sorted order.
    """
    num_faces = len(faces)
    edges = torch.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=1).sort(dim=-1).values
    key = (edges[..., 0] * (int(faces.max()) + 1) + edges[..., 1]).reshape(-1)  # [3F], one id per undirected edge
    order = key.argsort(stable=True)
    shared = key[order][1:] == key[order][:-1]
    first, second = order[:-1][shared], order[1:][shared]
    neighbors = torch.arange(num_faces, device=faces.device).repeat_interleave(3)
    neighbors[first] = second // 3
    neighbors[second] = first // 3
    return neighbors.reshape(num_faces, 3)


def mesh_edges(faces: torch.Tensor) -> torch.Tensor:
    """Unique undirected edges ``[E, 2]`` (``i < j``) of a triangle list."""
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).sort(dim=1).values
    edges = edges[edges[:, 0] != edges[:, 1]]
    return torch.unique(edges, dim=0)


def face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Unit normals ``[F, 3]``; zero for degenerate triangles."""
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normal = torch.cross(b - a, c - a, dim=-1)
    return normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
