from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.point_inputs import prepare_point_cloud_inputs, validate_point_cloud_inputs
from models.types import ModelCapabilities, ModelInput, ModelOutput

Floats = tuple[float, ...]


@dataclass
class PointPillarsConfig(BaseModelConfig):
    """PointPillars detector config. Defaults follow the KITTI setting (car, pedestrian, cyclist).

    ``input_dim`` counts XYZ plus optional point features (4 with LiDAR intensity).
    ``point_cloud_range`` is ``(x_min, y_min, z_min, x_max, y_max, z_max)`` in metres and
    ``pillar_size`` the ``(x, y)`` pillar footprint. One class per ``anchor_sizes`` entry
    ``(length, width, height)``, placed at ``anchor_bottom_heights``; each gets two anchors per
    cell (heading 0 and pi/2). ``layer_*`` and ``upsample_*`` describe the 2D backbone.
    ``positive_iou`` / ``negative_iou`` are per-class BEV IoU thresholds for anchor matching.
    """

    input_dim: int = 4
    point_cloud_range: Floats = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)
    pillar_size: Floats = (0.16, 0.16)
    pillar_channels: int = 64
    layer_nums: tuple[int, ...] = (3, 5, 5)
    layer_strides: tuple[int, ...] = (2, 2, 2)
    num_filters: tuple[int, ...] = (64, 128, 256)
    upsample_strides: tuple[int, ...] = (1, 2, 4)
    upsample_filters: tuple[int, ...] = (128, 128, 128)
    anchor_sizes: tuple[Floats, ...] = ((3.9, 1.6, 1.56), (0.8, 0.6, 1.73), (1.76, 0.6, 1.73))
    anchor_bottom_heights: Floats = (-1.78, -0.6, -0.6)
    positive_iou: Floats = (0.6, 0.5, 0.5)
    negative_iou: Floats = (0.45, 0.35, 0.35)
    representation: str = "POINT_CLOUD"

    def __post_init__(self) -> None:
        self.name = "pointpillars"
        self.model_type = "pointpillars"
        self.representation = "POINT_CLOUD"
        if self.input_dim < 3:
            raise ValueError("input_dim must include at least XYZ coordinates (3 channels).")
        if len(self.point_cloud_range) != 6 or len(self.pillar_size) != 2 or min(self.pillar_size) <= 0:
            raise ValueError("point_cloud_range needs 6 values and pillar_size 2 positive values.")
        lengths = {len(self.layer_nums), len(self.layer_strides), len(self.num_filters),
                   len(self.upsample_strides), len(self.upsample_filters)}
        if len(lengths) != 1:
            raise ValueError("layer_nums, layer_strides, num_filters and upsample_* need equal lengths.")
        per_class = {len(self.anchor_sizes), len(self.anchor_bottom_heights), len(self.positive_iou), len(self.negative_iou)}
        if len(per_class) != 1 or not self.anchor_sizes:
            raise ValueError("anchor_sizes, anchor_bottom_heights, positive_iou and negative_iou need one entry per class.")
        strides = [math.prod(self.layer_strides[: i + 1]) / up for i, up in enumerate(self.upsample_strides)]
        if len(set(strides)) != 1:
            raise ValueError("Every upsampled backbone level must land on the same stride.")
        total = math.prod(self.layer_strides)
        if any(size % total for size in self.grid_size):
            raise ValueError(f"The pillar grid {self.grid_size} must be divisible by the backbone stride {total}.")

    @property
    def feature_dim(self) -> int:
        return self.input_dim - 3

    @property
    def num_classes(self) -> int:
        return len(self.anchor_sizes)

    @property
    def grid_size(self) -> tuple[int, int]:
        """Pillar grid ``(nx, ny)``."""
        x0, y0, _, x1, y1, _ = self.point_cloud_range
        return round((x1 - x0) / self.pillar_size[0]), round((y1 - y0) / self.pillar_size[1])

    @property
    def feature_stride(self) -> int:
        return int(self.layer_strides[0] / self.upsample_strides[0])


def limit_period(angle: torch.Tensor, period: float) -> torch.Tensor:
    """Wrap ``angle`` into ``[0, period)``."""
    return angle - torch.floor(angle / period) * period


def nearest_bev_iou(boxes: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    """Pairwise BEV IoU ``[..., A, M]`` of ``[..., A, 7]`` and ``[..., M, 7]`` boxes ``(x, y, z, l, w, h, yaw)``.

    Each box is snapped to the nearest axis-aligned rectangle first, as in SECOND/OpenPCDet matching.
    """

    def corners(b: torch.Tensor) -> torch.Tensor:
        swap = torch.sin(b[..., 6]).abs() > torch.cos(b[..., 6]).abs()
        half = torch.where(swap.unsqueeze(-1), b[..., [4, 3]], b[..., [3, 4]]) / 2
        return torch.cat([b[..., :2] - half, b[..., :2] + half], dim=-1)

    a, m = corners(boxes).unsqueeze(-2), corners(others).unsqueeze(-3)
    wh = (torch.minimum(a[..., 2:], m[..., 2:]) - torch.maximum(a[..., :2], m[..., :2])).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area = lambda c: (c[..., 2] - c[..., 0]) * (c[..., 3] - c[..., 1])  # noqa: E731
    return inter / (area(a) + area(m) - inter).clamp_min(1e-6)


def _conv(cin: int, cout: int, stride: int = 1) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01), nn.ReLU(inplace=True)]


class PointPillars(BaseNetwork):
    """PointPillars (Lang et al., 2019): pillar PointNet, BEV pseudo-image, 2D CNN, SSD anchor head.

    Input LiDAR ``coordinates [B, N, 3]`` (+ ``point_features``, e.g. intensity). Output decoded
    boxes ``predictions [B, A, 7]`` as ``(x, y, z, l, w, h, yaw)`` with z at the box centre, one per
    anchor, plus ``auxiliary["scores"] [B, A, num_classes]``. Train with ``detection_loss``;
    filter by score and run NMS for final detections.
    """

    config_class = PointPillarsConfig

    def __init__(self, config: PointPillarsConfig | None = None):
        super().__init__(config=config or PointPillarsConfig())
        cfg = self.config
        self.pillar_net = nn.Linear(cfg.input_dim + 5, cfg.pillar_channels, bias=False)
        self.pillar_norm = nn.BatchNorm1d(cfg.pillar_channels, eps=1e-3, momentum=0.01)

        self.blocks, self.ups = nn.ModuleList(), nn.ModuleList()
        cin = cfg.pillar_channels
        for num, stride, cout, up, up_out in zip(cfg.layer_nums, cfg.layer_strides, cfg.num_filters,
                                                 cfg.upsample_strides, cfg.upsample_filters):
            layers = _conv(cin, cout, stride)
            for _ in range(num):
                layers += _conv(cout, cout)
            self.blocks.append(nn.Sequential(*layers))
            self.ups.append(nn.Sequential(nn.ConvTranspose2d(cout, up_out, up, up, bias=False),
                                          nn.BatchNorm2d(up_out, eps=1e-3, momentum=0.01), nn.ReLU(inplace=True)))
            cin = cout

        per_cell = 2 * cfg.num_classes
        width = sum(cfg.upsample_filters)
        self.cls_head = nn.Conv2d(width, per_cell * cfg.num_classes, 1)
        self.box_head = nn.Conv2d(width, per_cell * 7, 1)
        self.dir_head = nn.Conv2d(width, per_cell * 2, 1)
        nn.init.constant_(self.cls_head.bias, -math.log((1 - 0.01) / 0.01))  # focal-loss prior

        anchors, anchor_class = self._anchors()
        self.register_buffer("anchors", anchors, persistent=False)  # [A, 7]
        self.register_buffer("anchor_class", anchor_class, persistent=False)  # [A]

    def _anchors(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        x0, y0, _, x1, y1, _ = cfg.point_cloud_range
        nx, ny = (n // cfg.feature_stride for n in cfg.grid_size)
        xs = x0 + (torch.arange(nx, dtype=torch.float32) + 0.5) * (x1 - x0) / nx
        ys = y0 + (torch.arange(ny, dtype=torch.float32) + 0.5) * (y1 - y0) / ny
        sizes = torch.tensor(cfg.anchor_sizes, dtype=torch.float32)  # [C, 3]
        zs = torch.tensor(cfg.anchor_bottom_heights) + sizes[:, 2] / 2
        rot = torch.tensor([0.0, math.pi / 2])
        grid_y, grid_x, cls, r = torch.meshgrid(ys, xs, torch.arange(cfg.num_classes, dtype=ys.dtype), rot, indexing="ij")
        cls = cls.long()
        anchors = torch.stack([grid_x, grid_y, zs[cls], *sizes[cls].unbind(-1), r], dim=-1)
        return anchors.reshape(-1, 7), cls.reshape(-1)  # order [ny, nx, class, rotation] matches the heads

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
        validate_point_cloud_inputs(inputs, self.config.feature_dim, "PointPillars")
        return True

    def _pseudo_image(self, points: torch.Tensor, features: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        batch = points.shape[0]
        nx, ny = cfg.grid_size
        lo = points.new_tensor(cfg.point_cloud_range[:3])
        hi = points.new_tensor(cfg.point_cloud_range[3:])
        valid = mask & ((points >= lo) & (points < hi)).all(dim=-1)
        cell = ((points[..., :2] - lo[:2]) / points.new_tensor(cfg.pillar_size)).floor().long()
        ix, iy = cell[..., 0].clamp(0, nx - 1), cell[..., 1].clamp(0, ny - 1)
        dump = batch * ny * nx  # one extra bucket collects points outside the range
        pillar = torch.arange(batch, device=points.device).unsqueeze(1) * ny * nx + iy * nx + ix
        pillar = torch.where(valid, pillar, torch.full_like(pillar, dump)).reshape(-1)

        flat = points.reshape(-1, 3)
        count = torch.zeros(dump + 1, device=points.device).index_add_(0, pillar, torch.ones_like(flat[:, 0]))
        mean = torch.zeros(dump + 1, 3, device=points.device, dtype=flat.dtype).index_add_(0, pillar, flat)
        mean = mean / count.clamp_min(1).unsqueeze(-1)
        centre = torch.stack([ix, iy], dim=-1).reshape(-1, 2).to(flat.dtype) + 0.5
        centre = lo[:2] + centre * flat.new_tensor(cfg.pillar_size)
        parts = [flat] + ([] if features is None else [features.reshape(flat.shape[0], -1)])
        decorated = torch.cat(parts + [flat - mean[pillar], flat[:, :2] - centre], dim=-1)

        keep = valid.reshape(-1)
        encoded = F.relu(self.pillar_norm(self.pillar_net(decorated[keep])))
        canvas = torch.zeros(dump, cfg.pillar_channels, device=points.device, dtype=encoded.dtype)
        index = pillar[keep].unsqueeze(-1).expand_as(encoded)
        canvas = canvas.scatter_reduce(0, index, encoded, reduce="amax")  # ReLU >= 0, so zero init is safe
        return canvas.view(batch, ny, nx, -1).permute(0, 3, 1, 2)

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        points, features, mask = prepare_point_cloud_inputs(inputs, self.anchors)
        x = self._pseudo_image(points, features, mask)
        levels = []
        for block, up in zip(self.blocks, self.ups):
            x = block(x)
            levels.append(up(x))
        x = torch.cat(levels, dim=1)

        batch = x.shape[0]
        flat = lambda head, width: head(x).permute(0, 2, 3, 1).reshape(batch, -1, width)  # noqa: E731
        logits = flat(self.cls_head, self.config.num_classes)
        deltas = flat(self.box_head, 7)
        direction = flat(self.dir_head, 2)
        boxes = self.decode(deltas, direction)
        return ModelOutput(
            predictions=boxes,
            latent=x,
            auxiliary={"scores": logits.sigmoid(), "class_logits": logits, "box_deltas": deltas,
                       "direction_logits": direction, "anchors": self.anchors},
        )

    def decode(self, deltas: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """SECOND box coding: anchor-relative deltas to ``(x, y, z, l, w, h, yaw)``."""
        a = self.anchors
        diag = torch.sqrt(a[:, 3] ** 2 + a[:, 4] ** 2)
        xy = deltas[..., :2] * diag.unsqueeze(-1) + a[:, :2]
        z = deltas[..., 2] * a[:, 5] + a[:, 2]
        size = deltas[..., 3:6].exp() * a[:, 3:6]
        yaw = limit_period(deltas[..., 6] + a[:, 6], math.pi) + math.pi * direction.argmax(-1)
        return torch.cat([xy, z.unsqueeze(-1), size, yaw.unsqueeze(-1)], dim=-1)

    def encode(self, boxes: torch.Tensor) -> torch.Tensor:
        """Inverse of ``decode`` for target boxes ``[..., A, 7]`` matched to each anchor."""
        a = self.anchors
        diag = torch.sqrt(a[:, 3] ** 2 + a[:, 4] ** 2)
        return torch.cat([
            (boxes[..., :2] - a[:, :2]) / diag.unsqueeze(-1),
            ((boxes[..., 2] - a[:, 2]) / a[:, 5]).unsqueeze(-1),
            (boxes[..., 3:6].clamp_min(1e-6) / a[:, 3:6]).log(),
            (boxes[..., 6] - a[:, 6]).unsqueeze(-1),
        ], dim=-1)

    def detection_loss(
        self, output: ModelOutput, gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
        weights: tuple[float, float, float] = (1.0, 2.0, 0.2),
    ) -> dict[str, torch.Tensor]:
        """Anchor matching plus focal, smooth-L1 (sin yaw) and direction losses, as in the paper.

        ``gt_boxes [B, M, 7]`` as ``(x, y, z, l, w, h, yaw)``; ``gt_labels [B, M]`` class ids, ``-1`` pads.
        Returns the weighted ``loss`` plus its ``cls``, ``box`` and ``dir`` parts.
        """
        aux = output.auxiliary
        gt_boxes, gt_labels = _pad_empty(gt_boxes.to(self.anchors), gt_labels.to(self.anchors.device).long())
        cfg = self.config
        iou = nearest_bev_iou(self.anchors.expand(gt_boxes.shape[0], -1, -1), gt_boxes)  # [B, A, M]
        same = (self.anchor_class.view(1, -1, 1) == gt_labels.unsqueeze(1)) & (gt_labels >= 0).unsqueeze(1)
        iou = torch.where(same, iou, torch.full_like(iou, -1.0))
        best, match = iou.max(dim=-1)
        top = iou.max(dim=1, keepdim=True).values  # best anchor of every box
        forced = (iou == top) & (top > 0)
        match = torch.where(forced.any(-1), forced.float().argmax(-1), match)
        positive = (best >= self.anchors.new_tensor(cfg.positive_iou)[self.anchor_class]) | forced.any(-1)
        negative = (best < self.anchors.new_tensor(cfg.negative_iou)[self.anchor_class]) & ~positive
        num_pos = positive.sum().clamp_min(1).float()

        target_cls = F.one_hot(self.anchor_class, cfg.num_classes).float() * positive.unsqueeze(-1)
        cls_loss = _focal(aux["class_logits"], target_cls)[positive | negative].sum() / num_pos

        matched = torch.gather(gt_boxes, 1, match.unsqueeze(-1).expand(-1, -1, 7))
        target = self.encode(matched)[positive]
        pred = aux["box_deltas"][positive]
        diff = torch.cat([pred[:, :6] - target[:, :6], torch.sin(pred[:, 6:] - target[:, 6:])], dim=-1)
        box_loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=1 / 9, reduction="sum") / num_pos
        dir_target = (limit_period(matched[..., 6], 2 * math.pi) >= math.pi).long()
        dir_loss = F.cross_entropy(aux["direction_logits"][positive], dir_target[positive], reduction="sum") / num_pos
        loss = weights[0] * cls_loss + weights[1] * box_loss + weights[2] * dir_loss
        return {"loss": loss, "cls": cls_loss, "box": box_loss, "dir": dir_loss}


def _focal(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    prob = logits.sigmoid()
    pt = torch.where(target > 0, prob, 1 - prob)
    weight = torch.where(target > 0, alpha, 1 - alpha) * (1 - pt) ** gamma
    return weight * F.binary_cross_entropy_with_logits(logits, target, reduction="none")


def _pad_empty(gt_boxes: torch.Tensor, gt_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Give scenes without boxes one padding slot (label -1) so matching keeps a box axis."""
    if gt_boxes.shape[1]:
        return gt_boxes, gt_labels
    return gt_boxes.new_zeros(gt_boxes.shape[0], 1, 7), gt_labels.new_full((gt_boxes.shape[0], 1), -1)
