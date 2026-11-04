from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.point_inputs import prepare_point_cloud_inputs, validate_point_cloud_inputs
from models.pointnet.components import SharedPointMLP, masked_max
from models.types import ModelCapabilities, ModelInput, ModelOutput


@dataclass
class PCNConfig(BaseModelConfig):
    """PCN completion network: ``num_coarse`` coarse points, each refined into a ``grid_size**2`` patch.

    ``input_dim`` counts XYZ plus optional point features; ``latent_dim`` is the global feature
    width; ``grid_scale`` is the half-width of the 2D patch folded around each coarse point.
    """

    input_dim: int = 3
    latent_dim: int = 1024
    hidden_dim: int = 512
    num_coarse: int = 256
    grid_size: int = 2
    grid_scale: float = 0.05
    representation: str = "POINT_CLOUD"

    def __post_init__(self) -> None:
        self.name = "pcn"
        self.model_type = "pcn"
        self.representation = "POINT_CLOUD"
        if self.input_dim < 3:
            raise ValueError("input_dim must include at least XYZ coordinates (3 channels).")
        for key in ("latent_dim", "hidden_dim", "num_coarse", "grid_size"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive.")
        if self.grid_scale <= 0:
            raise ValueError("grid_scale must be positive.")

    @property
    def feature_dim(self) -> int:
        return self.input_dim - 3

    @property
    def num_output_points(self) -> int:
        return self.num_coarse * self.grid_size**2


class PCN(BaseNetwork):
    """Point Completion Network (Yuan et al., 2018): two stacked PointNets, coarse MLP, folding refinement.

    Input a partial ``coordinates [B, N, 3]``; output the dense completion
    ``predictions [B, num_coarse * grid_size**2, 3]`` and ``auxiliary["coarse"] [B, num_coarse, 3]``.
    Train with a Chamfer or EMD loss on both.
    """

    config_class = PCNConfig

    def __init__(self, config: PCNConfig | None = None):
        super().__init__(config=config or PCNConfig())
        cfg = self.config
        self.first = SharedPointMLP((cfg.input_dim, 128, 256), final_activation=False, normalization="none")
        self.second = SharedPointMLP((512, 512, cfg.latent_dim), final_activation=False, normalization="none")
        self.coarse = nn.Sequential(
            nn.Linear(cfg.latent_dim, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * cfg.num_coarse),
        )
        self.folding = SharedPointMLP((cfg.latent_dim + 5, cfg.hidden_dim, cfg.hidden_dim, 3), final_activation=False, normalization="none")
        axis = torch.linspace(-cfg.grid_scale, cfg.grid_scale, cfg.grid_size)
        patch = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=0).reshape(2, -1)
        self.register_buffer("patch", patch.repeat(1, cfg.num_coarse), persistent=False)  # [2, M]

    @property
    def required_inputs(self) -> set[str]:
        return {"coordinates"}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            representation="POINT_CLOUD",
            requires={"coordinates"},
            supports={"point_features", "point_mask"},
            variable_geometry=True,
            variable_node_count=True,
        )

    def validate_inputs(self, inputs: ModelInput) -> bool:
        super().validate_inputs(inputs)
        validate_point_cloud_inputs(inputs, self.config.feature_dim, "PCN")
        return True

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        points, features, mask = prepare_point_cloud_inputs(inputs, self.patch)
        x = points if features is None else torch.cat([points, features], dim=-1)
        local = self.first(x.transpose(1, 2))  # [B, 256, N]
        pooled = masked_max(local, mask).unsqueeze(-1).expand_as(local)
        code = masked_max(self.second(torch.cat([local, pooled], dim=1)), mask)  # [B, latent]

        batch = code.shape[0]
        coarse = self.coarse(code).reshape(batch, self.config.num_coarse, 3)
        centers = coarse.repeat_interleave(self.config.grid_size**2, dim=1).transpose(1, 2)  # [B, 3, M]
        patch = self.patch.unsqueeze(0).expand(batch, -1, -1)
        code_map = code.unsqueeze(-1).expand(-1, -1, patch.shape[-1])
        fine = self.folding(torch.cat([patch, centers, code_map], dim=1)) + centers
        return ModelOutput(predictions=fine.transpose(1, 2), latent=code, auxiliary={"coarse": coarse})
