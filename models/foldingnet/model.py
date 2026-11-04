from __future__ import annotations

from dataclasses import dataclass

import torch

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.point_inputs import prepare_point_cloud_inputs, validate_point_cloud_inputs
from models.pointnet.components import SharedPointMLP, masked_max
from models.types import ModelCapabilities, ModelInput, ModelOutput


@dataclass
class FoldingNetConfig(BaseModelConfig):
    """FoldingNet point-set decoder: ``grid_size**2`` output points folded from a 2D grid.

    ``input_dim`` counts XYZ plus optional point features; ``latent_dim`` is the codeword width;
    ``grid_scale`` is the half-width of the square 2D grid that is folded onto the surface.
    """

    input_dim: int = 3
    latent_dim: int = 512
    hidden_dim: int = 512
    grid_size: int = 32
    grid_scale: float = 0.3
    representation: str = "POINT_CLOUD"

    def __post_init__(self) -> None:
        self.name = "foldingnet"
        self.model_type = "foldingnet"
        self.representation = "POINT_CLOUD"
        if self.input_dim < 3:
            raise ValueError("input_dim must include at least XYZ coordinates (3 channels).")
        for key in ("latent_dim", "hidden_dim", "grid_size"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive.")
        if self.grid_scale <= 0:
            raise ValueError("grid_scale must be positive.")

    @property
    def feature_dim(self) -> int:
        return self.input_dim - 3

    @property
    def num_output_points(self) -> int:
        return self.grid_size**2


class FoldingNet(BaseNetwork):
    """FoldingNet (Yang et al., 2018): encode a cloud to a codeword, fold a 2D grid onto it twice.

    Input ``coordinates [B, N, 3]``; output ``predictions [B, grid_size**2, 3]``, a new point set
    whose size does not depend on N. Train with a Chamfer or EMD loss.
    """

    config_class = FoldingNetConfig

    def __init__(self, config: FoldingNetConfig | None = None):
        super().__init__(config=config or FoldingNetConfig())
        cfg = self.config
        # shortcut: PointNet-style encoder instead of the paper's k-NN graph layers; swap in DGCNN features if detail suffers.
        self.encoder = SharedPointMLP((cfg.input_dim, 64, 128, cfg.latent_dim), final_activation=False, normalization="none")
        self.fold1 = SharedPointMLP((cfg.latent_dim + 2, cfg.hidden_dim, cfg.hidden_dim, 3), final_activation=False, normalization="none")
        self.fold2 = SharedPointMLP((cfg.latent_dim + 3, cfg.hidden_dim, cfg.hidden_dim, 3), final_activation=False, normalization="none")
        axis = torch.linspace(-cfg.grid_scale, cfg.grid_scale, cfg.grid_size)
        grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=0).reshape(2, -1)
        self.register_buffer("grid", grid, persistent=False)  # [2, M]

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
        validate_point_cloud_inputs(inputs, self.config.feature_dim, "FoldingNet")
        return True

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        points, features, mask = prepare_point_cloud_inputs(inputs, self.grid)
        x = points if features is None else torch.cat([points, features], dim=-1)
        code = masked_max(self.encoder(x.transpose(1, 2)), mask)  # [B, latent]

        num_out = self.grid.shape[1]
        code_map = code.unsqueeze(-1).expand(-1, -1, num_out)
        grid = self.grid.unsqueeze(0).expand(code.shape[0], -1, -1)
        first = self.fold1(torch.cat([code_map, grid], dim=1))
        second = self.fold2(torch.cat([code_map, first], dim=1))
        return ModelOutput(
            predictions=second.transpose(1, 2),
            latent=code,
            auxiliary={"first_fold": first.transpose(1, 2)},
        )
