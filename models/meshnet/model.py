"""MeshNet (Feng et al., AAAI 2019): faces as the unit, spatial and structural descriptors, mesh convolution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.mesh_inputs import face_neighbors, face_normals, prepare_mesh_inputs, validate_mesh_inputs
from models.types import ModelCapabilities, ModelInput, ModelOutput


def _mlp(*dims: int) -> nn.Sequential:
    """Per-face Linear + BatchNorm + ReLU stack (the reference uses 1x1 convolutions)."""
    layers: list[nn.Module] = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU()]
    return nn.Sequential(*layers)


@dataclass
class MeshNetConfig(BaseModelConfig):
    """MeshNet shape classifier on triangle meshes.

    output_dim: number of classes. num_kernel: face-kernel-correlation kernels, each with 4 learned
    unit normals compared to a face's normal and its 3 neighbours' normals (Gaussian width sigma).
    """

    output_dim: int = 10
    num_kernel: int = 64
    sigma: float = 0.2
    dropout: float = 0.5
    representation: str = "MESH"

    def __post_init__(self) -> None:
        self.name = "meshnet"
        self.model_type = "meshnet"
        self.representation = "MESH"
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if self.num_kernel <= 0:
            raise ValueError("num_kernel must be positive.")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")


class FaceRotateConvolution(nn.Module):
    """Shared MLP over the three corner-vector pairs of a face, averaged (invariant to corner order shift)."""

    def __init__(self) -> None:
        super().__init__()
        self.rotate_mlp = _mlp(6, 32, 32)
        self.fusion_mlp = _mlp(32, 64, 64)

    def forward(self, corners: torch.Tensor) -> torch.Tensor:
        pairs = (corners[:, :6], corners[:, 3:9], torch.cat((corners[:, 6:], corners[:, :3]), dim=1))
        return self.fusion_mlp(sum(self.rotate_mlp(p) for p in pairs) / 3)


class FaceKernelCorrelation(nn.Module):
    """Gaussian correlation between {face normal, 3 neighbour normals} and learned kernels of 4 unit normals."""

    def __init__(self, num_kernel: int, sigma: float) -> None:
        super().__init__()
        self.sigma = sigma
        self.alpha = nn.Parameter(torch.rand(num_kernel, 4) * math.pi)
        self.beta = nn.Parameter(torch.rand(num_kernel, 4) * 2 * math.pi)
        self.norm = nn.BatchNorm1d(num_kernel)

    def forward(self, normals: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        ring = torch.cat((normals.unsqueeze(1), normals[neighbors]), dim=1)  # [F, 4, 3]
        kernel = torch.stack(
            (self.alpha.sin() * self.beta.cos(), self.alpha.sin() * self.beta.sin(), self.alpha.cos()), dim=-1
        )  # [K, 4, 3]
        # |r - k|^2 with |k| = 1, expanded to avoid a [F, K, 4, 4, 3] tensor.
        dist = ring.pow(2).sum(-1)[:, None, :, None] + 1 - 2 * torch.einsum("fid,kjd->fkij", ring, kernel)
        correlation = torch.exp(-dist / (2 * self.sigma**2)).mean(dim=(2, 3))
        return torch.relu(self.norm(correlation))


class MeshConvolution(nn.Module):
    """Combine spatial with structural features; aggregate structural features over the 3 face neighbours."""

    def __init__(self, spatial_in: int, structural_in: int, spatial_out: int, structural_out: int) -> None:
        super().__init__()
        self.combination_mlp = _mlp(spatial_in + structural_in, spatial_out)
        self.concat_mlp = _mlp(2 * structural_in, structural_in)
        self.aggregation_mlp = _mlp(structural_in, structural_out)

    def forward(self, spatial: torch.Tensor, structural: torch.Tensor, neighbors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = self.combination_mlp(torch.cat((spatial, structural), dim=1))
        pairs = torch.cat((structural.unsqueeze(1).expand(-1, 3, -1), structural[neighbors]), dim=-1)  # [F, 3, 2C]
        aggregated = self.concat_mlp(pairs.reshape(-1, pairs.shape[-1])).reshape(len(structural), 3, -1).amax(dim=1)
        return spatial, self.aggregation_mlp(aggregated)


class MeshNet(BaseNetwork):
    """MeshNet: per-face spatial (center) and structural (corners, normals, neighbour normals) descriptors,
    two mesh convolutions, max pooling over faces and an MLP classifier. Returns logits ``[B, output_dim]``.
    """

    config_class = MeshNetConfig

    def __init__(self, config: MeshNetConfig | None = None):
        super().__init__(config=config or MeshNetConfig())
        cfg = self.config
        self.spatial_mlp = _mlp(3, 64, 64)
        self.rotate = FaceRotateConvolution()
        self.kernel_correlation = FaceKernelCorrelation(cfg.num_kernel, cfg.sigma)
        self.structural_mlp = _mlp(64 + cfg.num_kernel + 3, 131)
        self.mesh_conv1 = MeshConvolution(64, 131, 256, 256)
        self.mesh_conv2 = MeshConvolution(256, 256, 512, 512)
        self.fusion_mlp = _mlp(512 + 512, 1024)
        self.concat_mlp = _mlp(256 + 512 + 1024, 1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(256, cfg.output_dim),
        )

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
        validate_mesh_inputs(inputs, "MeshNet")
        return True

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        vertices, faces, batch, num_meshes = prepare_mesh_inputs(inputs, next(self.parameters()))
        corners = vertices[faces]  # [F, 3, 3]
        center = corners.mean(dim=1)
        corners = (corners - center.unsqueeze(1)).reshape(len(faces), 9)
        normals = face_normals(vertices, faces)
        neighbors = face_neighbors(faces)

        spatial = self.spatial_mlp(center)
        structural = self.structural_mlp(
            torch.cat((self.rotate(corners), self.kernel_correlation(normals, neighbors), normals), dim=1)
        )
        spatial1, structural1 = self.mesh_conv1(spatial, structural, neighbors)
        spatial2, structural2 = self.mesh_conv2(spatial1, structural1, neighbors)
        spatial3 = self.fusion_mlp(torch.cat((spatial2, structural2), dim=1))
        features = self.concat_mlp(torch.cat((spatial1, spatial2, spatial3), dim=1))  # [F, 1024]

        face_batch = batch[faces[:, 0]]
        index = face_batch.unsqueeze(1).expand_as(features)
        pooled = features.new_full((num_meshes, features.shape[1]), float("-inf"))
        pooled = pooled.scatter_reduce(0, index, features, reduce="amax", include_self=True)
        return ModelOutput(predictions=self.classifier(pooled), latent=pooled, metadata={"face_batch": face_batch})
