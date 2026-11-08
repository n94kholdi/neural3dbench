from __future__ import annotations

import numpy as np
import pytest
import torch

from models import (
    DiffusionNet,
    DiffusionNetConfig,
    MeshNet,
    MeshNetConfig,
    MeshWalker,
    MeshWalkerConfig,
    ModelInput,
    ModelRegistry,
    collate_meshes,
    create_model,
)
from models.mesh_inputs import face_neighbors, mesh_edges

pytest.importorskip("scipy")

from models.diffusionnet import batch_operators, compute_operators  # noqa: E402

SMALL = {
    "meshnet": {"num_kernel": 8},
    "diffusionnet": {"width": 16, "num_blocks": 2, "k_eig": 8},
    "meshwalker": {"walk_length": 20, "num_walks_eval": 2, "fc_dims": (16,), "gru_dims": (16,)},
}


def grid_mesh(n: int, bump: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    u, v = torch.meshgrid(torch.linspace(0, 1, n), torch.linspace(0, 1, n), indexing="ij")
    vertices = torch.stack((u.ravel(), v.ravel(), bump * torch.sin(3 * u.ravel())), dim=1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    return vertices, torch.tensor(faces)


TETRA = (torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]), torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))


def batch() -> ModelInput:
    return collate_meshes([TETRA, grid_mesh(8), grid_mesh(5, 0.5)])


@pytest.mark.parametrize("name,cls,config_cls", [
    ("meshnet", MeshNet, MeshNetConfig), ("diffusionnet", DiffusionNet, DiffusionNetConfig),
    ("meshwalker", MeshWalker, MeshWalkerConfig),
])
def test_registry_forward_and_backward(name, cls, config_cls):
    assert ModelRegistry.get(name) is cls
    model = create_model({"name": name, "output_dim": 4, **SMALL[name]})
    assert isinstance(model.config, config_cls)
    output = model(batch())
    assert output.predictions.shape == (3, 4)
    assert torch.isfinite(output.predictions).all()
    output.predictions.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    model.eval()
    assert model(batch()).predictions.shape == (3, 4)


@pytest.mark.parametrize("name", ["meshnet", "diffusionnet", "meshwalker"])
def test_requires_faces_and_valid_batch(name):
    model = create_model({"name": name, **SMALL[name]})
    with pytest.raises(ValueError, match="faces"):
        model(ModelInput(coordinates=torch.randn(4, 3)))
    with pytest.raises(ValueError):
        model(ModelInput(coordinates=torch.randn(4, 3), faces=torch.tensor([[0, 1, 9]])))
    with pytest.raises(ValueError, match="different meshes"):
        model(ModelInput(coordinates=torch.randn(4, 3), faces=torch.tensor([[0, 1, 2]]), batch=torch.tensor([0, 0, 1, 1])))


def test_configs_validate():
    with pytest.raises(ValueError):
        MeshNetConfig(sigma=0)
    with pytest.raises(ValueError):
        DiffusionNetConfig(task="detection")
    with pytest.raises(ValueError):
        MeshWalkerConfig(walk_length=1)


def test_meshnet_is_invariant_to_face_order():
    model = create_model({"name": "meshnet", **SMALL["meshnet"]}).eval()
    vertices, faces = grid_mesh(6)
    shuffled = faces[torch.randperm(len(faces))]
    rolled = faces.roll(1, dims=1)  # same triangles, corners rotated
    reference = model(collate_meshes([(vertices, faces)])).predictions
    assert torch.allclose(reference, model(collate_meshes([(vertices, shuffled)])).predictions, atol=1e-5)
    assert torch.allclose(reference, model(collate_meshes([(vertices, rolled)])).predictions, atol=1e-5)


def test_face_neighbors_and_edges():
    neighbors = face_neighbors(TETRA[1])
    for face, ring in enumerate(neighbors.tolist()):
        assert sorted(ring) == sorted(set(range(4)) - {face})
    assert len(mesh_edges(TETRA[1])) == 6
    boundary = face_neighbors(torch.tensor([[0, 1, 2]]))
    assert boundary.tolist() == [[0, 0, 0]]


def test_diffusionnet_precomputed_operators_match_on_the_fly():
    model = create_model({"name": "diffusionnet", **SMALL["diffusionnet"]}).eval()
    meshes = [TETRA, grid_mesh(8)]
    inputs = collate_meshes(meshes)
    expected = model(inputs).predictions
    inputs.metadata["diffusion_operators"] = batch_operators([compute_operators(v.numpy(), f.numpy(), 8) for v, f in meshes])
    assert torch.allclose(model(inputs).predictions, expected, atol=1e-5)
    segmenter = create_model({"name": "diffusionnet", "task": "segmentation", "output_dim": 3, **SMALL["diffusionnet"]})
    assert segmenter(inputs).predictions.shape == (len(inputs.coordinates), 3)


def test_diffusionnet_operators():
    vertices, faces = grid_mesh(10, bump=0.0)
    ops = compute_operators(vertices.numpy(), faces.numpy(), 8)
    gram = ops["evecs"].T @ (ops["mass"][:, None] * ops["evecs"])
    assert np.allclose(gram, np.eye(8), atol=1e-6)  # M-orthonormal
    assert ops["evals"][0] < 1e-6 and np.all(np.diff(ops["evals"]) >= -1e-9)
    assert np.isclose(ops["mass"].sum(), 1.0)  # unit square
    f = 2 * vertices[:, 0].numpy() + 3 * vertices[:, 1].numpy()
    gx = np.bincount(ops["grad_rows"], ops["grad_x"] * f[ops["grad_cols"]], minlength=len(f))
    gy = np.bincount(ops["grad_rows"], ops["grad_y"] * f[ops["grad_cols"]], minlength=len(f))
    assert np.allclose(np.hypot(gx, gy), np.hypot(2, 3), atol=1e-2)  # frame is arbitrary, the length is not
    tiny = compute_operators(TETRA[0].numpy(), TETRA[1].numpy(), 8)  # fewer vertices than k_eig: zero padded
    assert tiny["evecs"].shape == (4, 8) and np.all(tiny["evecs"][:, 4:] == 0)


def test_meshwalker_walks_follow_edges_within_each_mesh():
    from models.meshwalker import random_walks

    inputs = collate_meshes([grid_mesh(6), TETRA])
    edges = mesh_edges(inputs.faces).numpy()
    adjacent = {tuple(e) for e in edges} | {tuple(e[::-1]) for e in edges}
    walks = random_walks(edges, [0, 36, 40], 3, 30, np.random.default_rng(0))
    assert walks.shape == (6, 30)
    assert walks[:3].max() < 36 and walks[3:].min() >= 36
    grid_walk = walks[0]
    assert len(set(grid_walk.tolist())) == 30  # 36 vertices: no revisit needed
    steps = list(zip(grid_walk[:-1].tolist(), grid_walk[1:].tolist()))
    assert sum(step in adjacent for step in steps) >= 25  # jumps only when stuck


def test_simplify_mesh_caps_faces():
    from tools.dashboard.data_sources import simplify_mesh

    vertices, faces = grid_mesh(40)
    small_v, small_f = simplify_mesh(vertices.numpy().astype(np.float64), faces.numpy(), 500)
    assert 0 < len(small_f) <= 500
    assert small_f.max() < len(small_v) and len(np.unique(small_f)) == len(small_v)
