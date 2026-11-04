from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.dgcnn.components import EdgeConv
from models.point_inputs import prepare_point_cloud_inputs, validate_point_cloud_inputs
from models.types import ModelCapabilities, ModelInput, ModelOutput


@dataclass
class DCPConfig(BaseModelConfig):
    """Deep Closest Point (DCP-v2) rigid registration config, paper defaults.

    ``edge_channels`` are the DGCNN EdgeConv widths (k-NN with ``neighbor_count``) shared by both
    clouds; their concatenation is projected to ``embedding_dim``. ``num_heads``, ``ff_dim`` and
    ``num_layers`` size the transformer that lets each cloud attend to the other. Set
    ``use_transformer=False`` for DCP-v1 (independent embeddings).
    """

    neighbor_count: int = 20
    edge_channels: tuple[int, ...] = (64, 64, 128, 256)
    embedding_dim: int = 512
    num_heads: int = 4
    ff_dim: int = 1024
    num_layers: int = 1
    dropout: float = 0.0
    use_transformer: bool = True
    representation: str = "POINT_CLOUD"

    def __post_init__(self) -> None:
        self.name = "dcp"
        self.model_type = "dcp"
        self.representation = "POINT_CLOUD"
        self.edge_channels = tuple(self.edge_channels)
        if not self.edge_channels or min(self.edge_channels) < 1:
            raise ValueError("edge_channels must contain positive channel sizes.")
        if min(self.neighbor_count, self.embedding_dim, self.num_heads, self.ff_dim, self.num_layers) < 1:
            raise ValueError("neighbor_count, embedding_dim, num_heads, ff_dim and num_layers must be positive.")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")


class DCP(BaseNetwork):
    """Deep Closest Point (Wang & Solomon, 2019): DGCNN embeddings, cross-attention, soft matching, SVD.

    Input the source ``coordinates [B, N, 3]`` and the target ``query_coordinates [B, M, 3]``
    (optional ``point_mask [B, N]`` for the source). Output the rigid transform mapping source onto
    target, ``predictions [B, 4, 4]``, with ``auxiliary["rotation"] [B, 3, 3]``,
    ``auxiliary["translation"] [B, 3]``, ``auxiliary["correspondences"] [B, N, 3]`` (soft matches
    in the target) and ``auxiliary["scores"] [B, N, M]``. Train with ``registration_loss``.
    """

    config_class = DCPConfig

    def __init__(self, config: DCPConfig | None = None):
        super().__init__(config=config or DCPConfig())
        cfg = self.config
        channels = (3, *cfg.edge_channels)
        self.edge_convs = nn.ModuleList(
            EdgeConv(c_in, c_out, cfg.neighbor_count) for c_in, c_out in zip(channels[:-1], channels[1:])
        )
        self.embedding = nn.Sequential(
            nn.Conv1d(sum(cfg.edge_channels), cfg.embedding_dim, 1, bias=False),
            nn.BatchNorm1d(cfg.embedding_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.transformer = (
            nn.Transformer(
                cfg.embedding_dim, cfg.num_heads, cfg.num_layers, cfg.num_layers, cfg.ff_dim, cfg.dropout,
                batch_first=True,
            )
            if cfg.use_transformer
            else None
        )

    @property
    def required_inputs(self) -> set[str]:
        return {"coordinates", "query_coordinates"}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            representation="POINT_CLOUD",
            requires={"coordinates", "query_coordinates"},
            supports={"point_mask"},
            variable_geometry=True,
            variable_node_count=True,
        )

    def validate_inputs(self, inputs: ModelInput) -> bool:
        super().validate_inputs(inputs)
        validate_point_cloud_inputs(inputs, 0, "DCP")
        target = torch.as_tensor(inputs.query_coordinates)
        if target.ndim != 3 or target.shape[-1] != 3 or target.shape[0] != torch.as_tensor(inputs.coordinates).shape[0]:
            raise ValueError(f"DCP query_coordinates must have shape [B, M, 3], got {tuple(target.shape)}.")
        if not target.is_floating_point() or target.shape[1] < 1:
            raise ValueError("query_coordinates must be a non-empty floating-point cloud.")
        return True

    def _embed(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features, layers = points, []
        for conv in self.edge_convs:
            features = conv(features, mask)
            layers.append(features)
        return self.embedding(torch.cat(layers, dim=-1).transpose(1, 2)).transpose(1, 2)  # [B, N, E]

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        src, _, src_mask = prepare_point_cloud_inputs(inputs, self.embedding[0].weight)
        tgt = torch.as_tensor(inputs.query_coordinates, device=src.device, dtype=src.dtype)
        tgt_mask = torch.ones(tgt.shape[:2], dtype=torch.bool, device=tgt.device)

        src_emb, tgt_emb = self._embed(src, src_mask), self._embed(tgt, tgt_mask)
        if self.transformer is not None:
            # nn.Transformer(a, b) returns b-shaped features that attended to a: phi(X) = X + T(Y, X).
            src_emb, tgt_emb = (
                src_emb + self.transformer(tgt_emb, src_emb, tgt_key_padding_mask=~src_mask),
                tgt_emb + self.transformer(src_emb, tgt_emb, src_key_padding_mask=~src_mask,
                                           memory_key_padding_mask=~src_mask),
            )

        scores = torch.softmax(src_emb @ tgt_emb.transpose(1, 2) / src_emb.shape[-1] ** 0.5, dim=-1)  # [B, N, M]
        corr = scores @ tgt  # [B, N, 3]

        # Procrustes over valid source points: R, t minimise sum ||R x_i + t - y_i||^2.
        w = src_mask.unsqueeze(-1).to(src.dtype)
        count = w.sum(1, keepdim=True)
        src_c, corr_c = (src * w).sum(1, keepdim=True) / count, (corr * w).sum(1, keepdim=True) / count
        cov = ((src - src_c) * w).transpose(1, 2) @ (corr - corr_c)  # [B, 3, 3]
        u, _, vh = torch.linalg.svd(cov)
        v = vh.transpose(1, 2)
        flip = torch.ones_like(cov[:, 0])
        flip[:, 2] = torch.linalg.det(v @ u.transpose(1, 2)).sign()  # avoid reflections
        rotation = v @ torch.diag_embed(flip) @ u.transpose(1, 2)
        translation = corr_c.squeeze(1) - (rotation @ src_c.transpose(1, 2)).squeeze(-1)

        transform = torch.eye(4, dtype=src.dtype, device=src.device).repeat(src.shape[0], 1, 1)
        transform[:, :3, :3], transform[:, :3, 3] = rotation, translation
        return ModelOutput(
            predictions=transform,
            auxiliary={"rotation": rotation, "translation": translation, "correspondences": corr, "scores": scores},
        )

    @staticmethod
    def registration_loss(
        output: ModelOutput, gt_rotation: torch.Tensor, gt_translation: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Paper loss ``||R^T R_gt - I||^2 + ||t - t_gt||^2``, averaged over the batch.

        ``gt_rotation [B, 3, 3]`` and ``gt_translation [B, 3]`` map source onto target. Weight decay
        (the paper's third term) belongs in the optimizer.
        """
        rotation, translation = output.auxiliary["rotation"], output.auxiliary["translation"]
        gt_rotation, gt_translation = gt_rotation.to(rotation), gt_translation.to(translation)
        eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        rotation_loss = (rotation.transpose(1, 2) @ gt_rotation - eye).pow(2).sum((1, 2)).mean()
        translation_loss = (translation - gt_translation).pow(2).sum(-1).mean()
        return {"rotation": rotation_loss, "translation": translation_loss, "total": rotation_loss + translation_loss}
