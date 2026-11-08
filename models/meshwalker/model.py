"""MeshWalker (Lahav & Tal, TOG 2020): random walks along mesh edges, read by an RNN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.mesh_inputs import mesh_edges, prepare_mesh_inputs, validate_mesh_inputs
from models.types import ModelCapabilities, ModelInput, ModelOutput


@dataclass
class MeshWalkerConfig(BaseModelConfig):
    """MeshWalker shape classifier on triangle meshes.

    walk_length: vertices per walk (the network reads walk_length - 1 xyz steps).
    num_walks / num_walks_eval: walks per mesh in training / evaluation; logits are averaged.
    fc_dims, gru_dims: widths of the step embedding and of the stacked GRUs.
    """

    output_dim: int = 10
    walk_length: int = 200
    num_walks: int = 1
    num_walks_eval: int = 8
    fc_dims: tuple[int, ...] = (128, 256)
    gru_dims: tuple[int, ...] = (1024, 1024, 512)
    dropout: float = 0.0
    representation: str = "MESH"

    def __post_init__(self) -> None:
        self.name = "meshwalker"
        self.model_type = "meshwalker"
        self.representation = "MESH"
        self.fc_dims, self.gru_dims = tuple(self.fc_dims), tuple(self.gru_dims)
        if self.output_dim <= 0 or self.num_walks <= 0 or self.num_walks_eval <= 0:
            raise ValueError("output_dim, num_walks and num_walks_eval must be positive.")
        if self.walk_length < 2:
            raise ValueError("walk_length must be at least 2.")
        if not self.fc_dims or not self.gru_dims or min(self.fc_dims + self.gru_dims) <= 0:
            raise ValueError("fc_dims and gru_dims need at least one positive width each.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")


def random_walks(
    edges: np.ndarray, offsets: list[int], walks_per_mesh: int, length: int, rng: np.random.Generator
) -> np.ndarray:
    """``[B * walks_per_mesh, length]`` global vertex ids, mesh-major.

    Each step moves to a random unvisited edge neighbour; when there is none it jumps to a random
    unvisited vertex of the same mesh. Once every vertex is visited, revisits are allowed.
    """
    num_vertices = offsets[-1]
    both = np.concatenate((edges, edges[:, ::-1]))
    order = np.argsort(both[:, 0], kind="stable")
    indices = both[order, 1]
    indptr = np.searchsorted(both[order, 0], np.arange(num_vertices + 1))
    walks = np.empty((len(offsets) - 1) * walks_per_mesh * length, dtype=np.int64).reshape(-1, length)
    row = 0
    for start, end in zip(offsets[:-1], offsets[1:]):
        size = end - start
        for _ in range(walks_per_mesh):
            visited = np.zeros(size, dtype=bool)
            current = int(rng.integers(size))
            visited[current] = True
            walks[row, 0] = current
            for step in range(1, length):
                neighbors = indices[indptr[start + current] : indptr[start + current + 1]] - start
                candidates = neighbors[~visited[neighbors]]
                if candidates.size == 0:
                    if visited.all():
                        visited[:] = False
                        visited[current] = True
                        candidates = neighbors[neighbors != current]
                    if candidates.size == 0:
                        candidates = np.flatnonzero(~visited)  # jump
                current = int(candidates[rng.integers(candidates.size)])
                visited[current] = True
                walks[row, step] = current
            walks[row] += start
            row += 1
    return walks


class MeshWalker(BaseNetwork):
    """MeshWalker: random walks over mesh vertices; each step's xyz displacement is embedded by
    FC layers and read by stacked GRUs; the last step classifies. Returns logits ``[B, output_dim]``
    averaged over ``num_walks`` (training) or ``num_walks_eval`` (eval) walks per mesh.
    """

    config_class = MeshWalkerConfig

    def __init__(self, config: MeshWalkerConfig | None = None):
        super().__init__(config=config or MeshWalkerConfig())
        cfg = self.config
        dims = (3, *cfg.fc_dims)
        self.embedding = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(nn.InstanceNorm1d(d, affine=True) for d in cfg.fc_dims)
        dims = (cfg.fc_dims[-1], *cfg.gru_dims)
        self.grus = nn.ModuleList(nn.GRU(a, b, batch_first=True) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.gru_dims[-1], cfg.output_dim)

    @property
    def required_inputs(self) -> set[str]:
        return {"coordinates", "faces"}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            representation="MESH",
            requires={"coordinates", "faces"},
            supports={"batch"},
            variable_geometry=True,
            variable_node_count=True,
        )

    def validate_inputs(self, inputs: ModelInput) -> bool:
        super().validate_inputs(inputs)
        validate_mesh_inputs(inputs, "MeshWalker")
        return True

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        vertices, faces, batch, num_meshes = prepare_mesh_inputs(inputs, next(self.parameters()))
        counts = torch.bincount(batch, minlength=num_meshes)
        offsets = [0, *torch.cumsum(counts, 0).tolist()]
        walks_per_mesh = self.config.num_walks if self.training else self.config.num_walks_eval
        rng = np.random.default_rng(int(torch.randint(2**31, (1,)).item()))  # follows torch.manual_seed
        walks = random_walks(mesh_edges(faces).cpu().numpy(), offsets, walks_per_mesh, self.config.walk_length, rng)
        walks = torch.as_tensor(walks, device=vertices.device)

        x = vertices[walks].diff(dim=1)  # [W, L - 1, 3]
        for linear, norm in zip(self.embedding, self.norms):
            x = torch.relu(norm(linear(x).transpose(1, 2)).transpose(1, 2))
        for gru in self.grus:
            x, _ = gru(self.dropout(x))
        logits = self.head(x[:, -1]).reshape(num_meshes, walks_per_mesh, -1)
        return ModelOutput(predictions=logits.mean(dim=1), auxiliary={"walks": walks})
