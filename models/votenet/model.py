from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.point_inputs import prepare_point_cloud_inputs, validate_point_cloud_inputs
from models.pointnet2.components import FeaturePropagation, SetAbstraction, square_distance
from models.types import ModelCapabilities, ModelInput, ModelOutput

SA_CHANNELS = ((64, 64, 128), (128, 128, 256), (128, 128, 256), (128, 128, 256))


@dataclass
class VoteNetConfig(BaseModelConfig):
    """VoteNet detector config (paper defaults for indoor scans, z up, metres).

    ``input_dim`` counts XYZ plus optional point features (e.g. RGB or height). One semantic class
    and one size cluster per ``mean_sizes`` entry ``(length, width, height)``; set them to the
    per-class mean box sizes of your dataset. ``sample_counts``, ``radii`` and ``neighbor_counts``
    configure the four PointNet++ set-abstraction levels; seeds are the second level's points.
    ``num_proposals`` vote clusters are sampled with radius ``proposal_radius``.
    """

    input_dim: int = 3
    mean_sizes: tuple[tuple[float, float, float], ...] = ((1.0, 1.0, 1.0),)
    num_heading_bins: int = 12
    sample_counts: tuple[int, ...] = (2048, 1024, 512, 256)
    radii: tuple[float, ...] = (0.2, 0.4, 0.8, 1.2)
    neighbor_counts: tuple[int, ...] = (64, 32, 16, 16)
    num_proposals: int = 256
    proposal_radius: float = 0.3
    representation: str = "POINT_CLOUD"

    def __post_init__(self) -> None:
        self.name = "votenet"
        self.model_type = "votenet"
        self.representation = "POINT_CLOUD"
        if self.input_dim < 3:
            raise ValueError("input_dim must include at least XYZ coordinates (3 channels).")
        if not self.mean_sizes or any(len(s) != 3 or min(s) <= 0 for s in self.mean_sizes):
            raise ValueError("mean_sizes needs one positive (length, width, height) per class.")
        if not len(self.sample_counts) == len(self.radii) == len(self.neighbor_counts) == 4:
            raise ValueError("sample_counts, radii and neighbor_counts need 4 entries.")
        if min(self.num_heading_bins, self.num_proposals, *self.sample_counts, *self.neighbor_counts) <= 0:
            raise ValueError("Counts and num_heading_bins must be positive.")
        if min(self.proposal_radius, *self.radii) <= 0:
            raise ValueError("Radii must be positive.")

    @property
    def feature_dim(self) -> int:
        return self.input_dim - 3

    @property
    def num_classes(self) -> int:
        return len(self.mean_sizes)


class VoteNet(BaseNetwork):
    """VoteNet (Qi et al., 2019): PointNet++ seeds vote for object centres; vote clusters become boxes.

    Input ``coordinates [B, N, 3]`` (+ ``point_features``). Output decoded boxes
    ``predictions [B, P, 7]`` as ``(x, y, z, l, w, h, yaw)``, one per proposal, with
    ``auxiliary["objectness"] [B, P]`` and ``auxiliary["class_scores"] [B, P, num_classes]``.
    Train with ``detection_loss``; filter by objectness and run NMS for final detections.
    """

    config_class = VoteNetConfig

    def __init__(self, config: VoteNetConfig | None = None):
        super().__init__(config=config or VoteNetConfig())
        cfg = self.config
        feature_in = [cfg.feature_dim] + [c[-1] for c in SA_CHANNELS[:-1]]
        self.sa = nn.ModuleList(
            SetAbstraction(n, r, k, f, c)
            for n, r, k, f, c in zip(cfg.sample_counts, cfg.radii, cfg.neighbor_counts, feature_in, SA_CHANNELS)
        )
        self.fp1 = FeaturePropagation(256 + 256, (256, 256))
        self.fp2 = FeaturePropagation(256 + 256, (256, 256))
        self.vote = nn.Sequential(
            nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 3 + 256, 1),
        )
        self.cluster = SetAbstraction(cfg.num_proposals, cfg.proposal_radius, 16, 256, (128, 128, 128))
        nh, nc = cfg.num_heading_bins, cfg.num_classes
        self.proposal = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 2 + 3 + 2 * nh + 4 * nc + nc, 1),
        )
        self.register_buffer("mean_sizes", torch.tensor(cfg.mean_sizes, dtype=torch.float32), persistent=False)

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
        validate_point_cloud_inputs(inputs, self.config.feature_dim, "VoteNet")
        return True

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        points, features, mask = prepare_point_cloud_inputs(inputs, self.mean_sizes)
        levels = [(points, features, mask)]
        for sa in self.sa:
            levels.append(sa(*levels[-1]))
        (p2, f2, m2), (p3, f3, m3), (p4, f4, m4) = levels[2:]
        f3 = self.fp1(p3, p4, f3, f4, m3, m4)
        seed_feat = self.fp2(p2, p3, f2, f3, m2, m3)  # [B, S, 256]

        vote = self.vote(seed_feat.transpose(1, 2)).transpose(1, 2)
        votes = p2 + vote[..., :3]
        vote_feat = F.normalize(seed_feat + vote[..., 3:], dim=-1)
        centres, cluster_feat, proposal_mask = self.cluster(votes, vote_feat, m2)
        raw = self.proposal(cluster_feat.transpose(1, 2)).transpose(1, 2)  # [B, P, C]

        nh, nc = self.config.num_heading_bins, self.config.num_classes
        parts = dict(zip(
            ("objectness_logits", "center_offset", "heading_logits", "heading_residual", "size_logits", "size_residual", "class_logits"),
            raw.split([2, 3, nh, nh, nc, 3 * nc, nc], dim=-1),
        ))
        parts["size_residual"] = parts["size_residual"].unflatten(-1, (nc, 3))
        boxes = self.decode(centres, parts)
        return ModelOutput(
            predictions=boxes,
            latent=cluster_feat,
            auxiliary={
                **parts,
                "objectness": parts["objectness_logits"].softmax(-1)[..., 1],
                "class_scores": parts["class_logits"].softmax(-1),
                "proposal_centers": centres, "proposal_mask": proposal_mask,
                "seeds": p2, "seed_mask": m2, "votes": votes,
            },
        )

    def decode(self, centres: torch.Tensor, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        """Proposal heads to ``(x, y, z, l, w, h, yaw)``: heading bin + residual, size cluster + residual."""
        bin_width = 2 * math.pi / self.config.num_heading_bins
        heading_bin = parts["heading_logits"].argmax(-1, keepdim=True)
        residual = parts["heading_residual"].gather(-1, heading_bin) * (bin_width / 2)
        yaw = heading_bin.to(centres.dtype) * bin_width + residual
        size_bin = parts["size_logits"].argmax(-1)
        mean = self.mean_sizes[size_bin]
        size_res = parts["size_residual"].gather(2, size_bin[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        size = (mean + size_res * mean).clamp_min(1e-3)
        return torch.cat([centres + parts["center_offset"], size, yaw], dim=-1)

    def detection_loss(
        self, output: ModelOutput, gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
        near: float = 0.3, far: float = 0.6,
    ) -> dict[str, torch.Tensor]:
        """Vote, objectness, box and semantic losses with the paper's weights (1, 0.5, 1, 0.1).

        ``gt_boxes [B, M, 7]`` as ``(x, y, z, l, w, h, yaw)``; ``gt_labels [B, M]`` class ids, ``-1`` pads.
        A proposal is positive within ``near`` metres of a box centre and negative beyond ``far``.
        """
        aux = output.auxiliary
        gt_boxes = gt_boxes.to(self.mean_sizes)
        gt_labels = gt_labels.to(self.mean_sizes.device).long()
        if not gt_boxes.shape[1]:  # empty scenes keep a padding slot so the box axis exists
            gt_boxes = gt_boxes.new_zeros(gt_boxes.shape[0], 1, 7)
            gt_labels = gt_labels.new_full((gt_boxes.shape[0], 1), -1)
        valid = gt_labels >= 0
        cfg = self.config

        # Votes: each seed inside a box regresses to that box centre.
        seeds, votes = aux["seeds"], aux["votes"]
        local = seeds.unsqueeze(2) - gt_boxes[:, None, :, :3]  # [B, S, M, 3]
        cos, sin = gt_boxes[..., 6].cos().unsqueeze(1), gt_boxes[..., 6].sin().unsqueeze(1)
        u = cos * local[..., 0] + sin * local[..., 1]
        v = -sin * local[..., 0] + cos * local[..., 1]
        inside = (torch.stack([u, v, local[..., 2]], -1).abs() <= gt_boxes[:, None, :, 3:6] / 2).all(-1)
        inside &= valid.unsqueeze(1) & aux["seed_mask"].unsqueeze(-1)
        seed_box = inside.float().argmax(-1)
        has_box = inside.any(-1)
        seed_target = gt_boxes.gather(1, seed_box.unsqueeze(-1).expand(-1, -1, 7))[..., :3]
        vote_loss = (votes - seed_target).abs().sum(-1)[has_box].sum() / has_box.sum().clamp_min(1)

        # Objectness: nearest box centre to each vote cluster.
        dist = square_distance(aux["proposal_centers"], gt_boxes[..., :3]).sqrt()
        dist = dist.masked_fill(~valid.unsqueeze(1), float("inf"))
        nearest, match = dist.min(-1)
        live = aux["proposal_mask"]
        positive, negative = live & (nearest < near), live & (nearest > far)
        obj_target = positive.long()
        obj_loss = F.cross_entropy(aux["objectness_logits"][positive | negative], obj_target[positive | negative],
                                   weight=aux["objectness_logits"].new_tensor([0.2, 0.8]), reduction="sum")
        obj_loss = obj_loss / (positive | negative).sum().clamp_min(1)

        # Box and class terms on positive proposals only.
        num_pos = positive.sum().clamp_min(1)
        box = gt_boxes.gather(1, match.unsqueeze(-1).expand(-1, -1, 7))[positive]
        label = gt_labels.gather(1, match)[positive]
        p = {k: aux[k][positive] for k in ("center_offset", "heading_logits", "heading_residual", "size_logits", "size_residual", "class_logits")}
        center = aux["proposal_centers"][positive] + p["center_offset"]
        center_loss = F.smooth_l1_loss(center, box[:, :3], reduction="sum") / num_pos

        bin_width = 2 * math.pi / cfg.num_heading_bins
        shifted = (box[:, 6] + bin_width / 2) % (2 * math.pi)  # bins are centred on k * bin_width
        heading_bin = (shifted // bin_width).long().clamp_max(cfg.num_heading_bins - 1)
        heading_res = (shifted - heading_bin * bin_width - bin_width / 2) / (bin_width / 2)
        heading_cls_loss = F.cross_entropy(p["heading_logits"], heading_bin, reduction="sum") / num_pos
        pred_res = p["heading_residual"].gather(-1, heading_bin.unsqueeze(-1)).squeeze(-1)
        heading_res_loss = F.smooth_l1_loss(pred_res, heading_res, reduction="sum") / num_pos

        mean = self.mean_sizes[label]
        size_cls_loss = F.cross_entropy(p["size_logits"], label, reduction="sum") / num_pos
        pred_size = p["size_residual"][torch.arange(label.numel(), device=label.device), label]
        size_res_loss = F.smooth_l1_loss(pred_size, (box[:, 3:6] - mean) / mean, reduction="sum") / num_pos
        sem_loss = F.cross_entropy(p["class_logits"], label, reduction="sum") / num_pos

        box_loss = center_loss + 0.1 * heading_cls_loss + heading_res_loss + 0.1 * size_cls_loss + size_res_loss
        loss = vote_loss + 0.5 * obj_loss + box_loss + 0.1 * sem_loss
        return {"loss": loss, "vote": vote_loss, "objectness": obj_loss, "box": box_loss, "semantic": sem_loss}
