"""Per-mesh DiffusionNet operators: lumped mass, Laplacian eigenbasis and tangent-space gradients.

Computed once per mesh on the CPU with NumPy/SciPy (``pip install scipy``), then batched with
``batch_operators`` into the tensors ``DiffusionNet`` reads from ``ModelInput.metadata``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def _cotan_laplacian(vertices: np.ndarray, faces: np.ndarray):
    """Cotangent Laplacian ``L`` (PSD, rows sum to 0) and lumped vertex mass ``[V]``."""
    import scipy.sparse as sp

    num_vertices = len(vertices)
    corners = vertices[faces]  # [F, 3, 3]
    rows, cols, weights = [], [], []
    for i in range(3):  # cotangent at corner i weights the opposite edge (j, k)
        j, k = (i + 1) % 3, (i + 2) % 3
        u, v = corners[:, j] - corners[:, i], corners[:, k] - corners[:, i]
        cross = np.linalg.norm(np.cross(u, v), axis=1)
        cot = np.clip((u * v).sum(1) / np.maximum(cross, 1e-12), -1e3, 1e3)  # degenerate triangles
        rows += [faces[:, j], faces[:, k]]
        cols += [faces[:, k], faces[:, j]]
        weights += [0.5 * cot, 0.5 * cot]
    rows, cols, weights = np.concatenate(rows), np.concatenate(cols), np.concatenate(weights)
    off = sp.coo_matrix((-weights, (rows, cols)), shape=(num_vertices, num_vertices)).tocsr()
    laplacian = off - sp.diags(np.asarray(off.sum(axis=1)).ravel())

    area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    mass = np.bincount(faces.ravel(), weights=np.repeat(area / 3, 3), minlength=num_vertices)
    mass = np.maximum(mass, 1e-6 * max(mass.mean(), 1e-12))  # vertices on zero-area faces only
    return laplacian.tocsc(), mass


def _eigenbasis(laplacian, mass: np.ndarray, k_eig: int) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``k_eig`` generalized eigenpairs of ``L x = lambda M x``; zero-padded for tiny meshes."""
    import scipy.linalg
    import scipy.sparse as sp
    import scipy.sparse.linalg

    num_vertices = len(mass)
    count = min(k_eig, num_vertices)
    eps = 1e-8
    for _ in range(5):  # shift by eps * M until the factorization succeeds
        shifted = laplacian + eps * sp.diags(mass)
        try:
            if num_vertices <= 2 * k_eig + 1:
                evals, evecs = scipy.linalg.eigh(shifted.toarray(), np.diag(mass))
                evals, evecs = evals[:count], evecs[:, :count]
            else:
                evals, evecs = scipy.sparse.linalg.eigsh(shifted, k=count, M=sp.diags(mass).tocsc(), sigma=1e-8)
            break
        except (RuntimeError, np.linalg.LinAlgError, scipy.linalg.LinAlgError):
            eps *= 10
    else:
        raise RuntimeError("Laplacian eigendecomposition failed; the mesh is badly degenerate.")
    order = np.argsort(evals)
    evals, evecs = np.clip(evals[order], 0.0, None), evecs[:, order]
    padded_vals, padded_vecs = np.zeros(k_eig), np.zeros((num_vertices, k_eig))
    padded_vals[:count], padded_vecs[:, :count] = evals, evecs
    return padded_vals, padded_vecs


def _gradient(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares tangent gradient from edge differences: COO ``(rows, cols, grad_x, grad_y)``.

    Each vertex gets a tangent frame (X, Y) normal to its area-weighted normal; its gradient
    solves ``min sum_j (g . e_ij - (f_j - f_i))^2`` over its edge neighbours j.
    """
    num_vertices = len(vertices)
    corners = vertices[faces]
    face_normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normal, faces[:, i], face_normal)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.where(length > 1e-12, normal / np.maximum(length, 1e-12), [0.0, 0.0, 1.0])
    helper = np.where(np.abs(normal[:, :1]) < 0.9, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    basis_x = helper - (helper * normal).sum(1, keepdims=True) * normal
    basis_x /= np.linalg.norm(basis_x, axis=1, keepdims=True)
    basis_y = np.cross(normal, basis_x)

    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    src = np.concatenate((edges[:, 0], edges[:, 1]))
    dst = np.concatenate((edges[:, 1], edges[:, 0]))
    offset = vertices[dst] - vertices[src]
    tangent = np.stack(((offset * basis_x[src]).sum(1), (offset * basis_y[src]).sum(1)), axis=1)  # [E, 2]

    gram = np.zeros((num_vertices, 2, 2))
    np.add.at(gram, src, tangent[:, :, None] * tangent[:, None, :])
    gram += 1e-5 * np.eye(2)
    det = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] ** 2
    inverse = np.stack((gram[:, 1, 1], -gram[:, 0, 1], -gram[:, 1, 0], gram[:, 0, 0]), axis=1).reshape(-1, 2, 2)
    inverse /= det[:, None, None]
    coeff = np.einsum("eab,eb->ea", inverse[src], tangent)  # weight of f_j in grad f_i; f_i gets minus the sum
    diagonal = np.zeros((num_vertices, 2))
    np.add.at(diagonal, src, -coeff)
    index = np.arange(num_vertices)
    rows, cols = np.concatenate((src, index)), np.concatenate((dst, index))
    values = np.concatenate((coeff, diagonal))
    return rows, cols, values[:, 0], values[:, 1]


def compute_operators(vertices: np.ndarray, faces: np.ndarray, k_eig: int = 128) -> dict[str, np.ndarray]:
    """Operators of one mesh: ``mass [V]``, ``evals [K]``, ``evecs [V, K]`` and gradient COO arrays."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    laplacian, mass = _cotan_laplacian(vertices, faces)
    evals, evecs = _eigenbasis(laplacian, mass, k_eig)
    rows, cols, grad_x, grad_y = _gradient(vertices, faces)
    return {"mass": mass, "evals": evals, "evecs": evecs,
            "grad_rows": rows, "grad_cols": cols, "grad_x": grad_x, "grad_y": grad_y}


def batch_operators(operators: Sequence[dict[str, np.ndarray]], device: torch.device | str = "cpu",
                    dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Concatenate per-mesh operators in ``collate_meshes`` order; gradients become block-diagonal sparse."""
    offsets = np.cumsum([0] + [len(op["mass"]) for op in operators])
    total = int(offsets[-1])
    rows = torch.as_tensor(np.concatenate([op["grad_rows"] + o for op, o in zip(operators, offsets)]))
    cols = torch.as_tensor(np.concatenate([op["grad_cols"] + o for op, o in zip(operators, offsets)]))
    index = torch.stack((rows, cols)).to(device)

    def cat(key: str) -> torch.Tensor:
        return torch.as_tensor(np.concatenate([op[key] for op in operators]), dtype=dtype, device=device)

    def sparse(key: str) -> torch.Tensor:
        return torch.sparse_coo_tensor(index, cat(key), (total, total)).coalesce()

    return {
        "mass": cat("mass"),
        "evals": torch.as_tensor(np.stack([op["evals"] for op in operators]), dtype=dtype, device=device),
        "evecs": cat("evecs"),
        "grad_x": sparse("grad_x"),
        "grad_y": sparse("grad_y"),
    }
