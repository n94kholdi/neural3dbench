"""DiffusionNet (Sharp et al., TOG 2022): learned heat diffusion, spatial gradient features and per-vertex MLPs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.mesh_inputs import prepare_mesh_inputs, validate_mesh_inputs
from models.types import ModelCapabilities, ModelInput, ModelOutput

from .operators import batch_operators, compute_operators

OPERATOR_KEY = "diffusion_operators"


@dataclass
class DiffusionNetConfig(BaseModelConfig):
    """DiffusionNet on triangle meshes, for shape classification or per-vertex segmentation.

    input_dim: per-vertex input channels (3 = xyz, the default when node_features is omitted).
    width: channels of every block. k_eig: Laplacian eigenvectors used for spectral diffusion.
    task: classification (mass-weighted mean over vertices, ``[B, output_dim]``) or segmentation
    (``[V, output_dim]``).
    """

    input_dim: int = 3
    output_dim: int = 10
    width: int = 128
    num_blocks: int = 4
    k_eig: int = 128
    dropout: float = 0.5
    task: str = "classification"
    with_gradient_features: bool = True
    with_gradient_rotations: bool = True
    representation: str = "MESH"

    def __post_init__(self) -> None:
        self.name = "diffusionnet"
        self.model_type = "diffusionnet"
        self.representation = "MESH"
        self.task = self.task.lower()
        for key in ("input_dim", "output_dim", "width", "num_blocks", "k_eig"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive.")
        if self.task not in {"classification", "segmentation"}:
            raise ValueError("task must be 'classification' or 'segmentation'.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")


class LearnedTimeDiffusion(nn.Module):
    """Spectral heat diffusion with one learned time per channel: ``Phi exp(-lambda t) Phi^T M x``."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.diffusion_time = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor, ops: dict[str, torch.Tensor], offsets: list[int]) -> torch.Tensor:
        time = self.diffusion_time.clamp_min(1e-8)
        out = []
        for b, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            evecs = ops["evecs"][start:end]
            spectral = evecs.T @ (ops["mass"][start:end, None] * x[start:end])  # [K, C]
            out.append(evecs @ (torch.exp(-ops["evals"][b, :, None] * time) * spectral))
        return torch.cat(out)


class SpatialGradientFeatures(nn.Module):
    """``tanh(Re(conj(g) . A g))`` per channel, with ``A`` a learned (complex) channel mixing."""

    def __init__(self, width: int, with_rotations: bool) -> None:
        super().__init__()
        self.with_rotations = with_rotations
        self.a_re = nn.Linear(width, width, bias=False)
        self.a_im = nn.Linear(width, width, bias=False) if with_rotations else None

    def forward(self, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
        if self.with_rotations:
            bx, by = self.a_re(gx) - self.a_im(gy), self.a_re(gy) + self.a_im(gx)
        else:
            bx, by = self.a_re(gx), self.a_re(gy)
        return torch.tanh(gx * bx + gy * by)


class DiffusionNetBlock(nn.Module):
    """Diffuse, take gradient features, mix with a per-vertex MLP, add the input back."""

    def __init__(self, width: int, dropout: float, with_gradient_features: bool, with_rotations: bool) -> None:
        super().__init__()
        self.diffusion = LearnedTimeDiffusion(width)
        self.gradient = SpatialGradientFeatures(width, with_rotations) if with_gradient_features else None
        mlp_in = width * (3 if with_gradient_features else 2)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor, ops: dict[str, torch.Tensor], offsets: list[int]) -> torch.Tensor:
        diffused = self.diffusion(x, ops, offsets)
        features = [x, diffused]
        if self.gradient is not None:
            features.append(self.gradient(torch.sparse.mm(ops["grad_x"], diffused), torch.sparse.mm(ops["grad_y"], diffused)))
        return x + self.mlp(torch.cat(features, dim=1))


class DiffusionNet(BaseNetwork):
    """DiffusionNet: blocks of learned spectral diffusion, gradient features and vertex MLPs.

    Reads ``coordinates [V, 3]`` and ``faces [F, 3]`` (concatenated batch with ``batch [V]``).
    Operators come from ``metadata["diffusion_operators"]`` (see ``batch_operators``) or are
    computed on the CPU in the forward pass, which is slow; precompute them for training.
    """

    config_class = DiffusionNetConfig

    def __init__(self, config: DiffusionNetConfig | None = None):
        super().__init__(config=config or DiffusionNetConfig())
        cfg = self.config
        self.first = nn.Linear(cfg.input_dim, cfg.width)
        self.blocks = nn.ModuleList(
            DiffusionNetBlock(cfg.width, cfg.dropout, cfg.with_gradient_features, cfg.with_gradient_rotations)
            for _ in range(cfg.num_blocks)
        )
        self.last = nn.Linear(cfg.width, cfg.output_dim)

    @property
    def required_inputs(self) -> set[str]:
        return {"coordinates", "faces"}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            representation="MESH",
            requires={"coordinates", "faces"},
            supports={"batch", "node_features", "metadata.diffusion_operators"},
            variable_geometry=True,
            variable_node_count=True,
        )

    def validate_inputs(self, inputs: ModelInput) -> bool:
        super().validate_inputs(inputs)
        validate_mesh_inputs(inputs, "DiffusionNet")
        num_vertices = len(inputs.coordinates)
        if inputs.node_features is None:
            if self.config.input_dim != 3:
                raise ValueError(f"DiffusionNet expects node_features [V, {self.config.input_dim}] (input_dim != 3).")
        elif tuple(inputs.node_features.shape) != (num_vertices, self.config.input_dim):
            raise ValueError(f"node_features must have shape [{num_vertices}, {self.config.input_dim}], "
                             f"got {tuple(inputs.node_features.shape)}.")
        ops = inputs.metadata.get(OPERATOR_KEY)
        if ops is not None:
            if len(ops["mass"]) != num_vertices or ops["evecs"].shape[1] != self.config.k_eig:
                raise ValueError(f"{OPERATOR_KEY} must cover {num_vertices} vertices with k_eig={self.config.k_eig}.")
        return True

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        parameter = next(self.parameters())
        vertices, faces, batch, num_meshes = prepare_mesh_inputs(inputs, parameter)
        counts = torch.bincount(batch, minlength=num_meshes)
        offsets = [0, *torch.cumsum(counts, 0).tolist()]
        ops = inputs.metadata.get(OPERATOR_KEY)
        if ops is None:
            ops = self._compute_operators(vertices, faces, offsets)
        ops = {k: v.to(device=parameter.device, dtype=parameter.dtype) for k, v in ops.items()}

        x = vertices if inputs.node_features is None else torch.as_tensor(
            inputs.node_features, device=parameter.device, dtype=parameter.dtype)
        x = self.first(x)
        for block in self.blocks:
            x = block(x, ops, offsets)
        x = self.last(x)  # [V, output_dim]
        if self.config.task == "segmentation":
            return ModelOutput(predictions=x, metadata={"batch": batch})
        mass = ops["mass"].unsqueeze(1)
        pooled = torch.zeros(num_meshes, x.shape[1], device=x.device, dtype=x.dtype).index_add(0, batch, mass * x)
        total = torch.zeros(num_meshes, 1, device=x.device, dtype=x.dtype).index_add(0, batch, mass)
        return ModelOutput(predictions=pooled / total, latent=x, metadata={"batch": batch})

    def _compute_operators(self, vertices: torch.Tensor, faces: torch.Tensor, offsets: list[int]) -> dict[str, torch.Tensor]:
        verts = vertices.detach().cpu().numpy()
        tris = faces.cpu().numpy()
        per_mesh = []
        for start, end in zip(offsets[:-1], offsets[1:]):
            mine = tris[(tris[:, 0] >= start) & (tris[:, 0] < end)] - start
            per_mesh.append(compute_operators(verts[start:end], mine, self.config.k_eig))
        return batch_operators(per_mesh)
