from __future__ import annotations

import math

import pytest
import torch

from models import ModelInput, ModelRegistry, PointPillars, PointPillarsConfig, VoteNet, VoteNetConfig, create_model

SMALL = {
    "pointpillars": {"point_cloud_range": (0.0, -6.4, -3.0, 12.8, 6.4, 1.0), "pillar_size": (0.4, 0.4),
                     "layer_nums": (1, 1, 1), "num_filters": (16, 16, 16), "upsample_filters": (8, 8, 8),
                     "pillar_channels": 16},
    "votenet": {"mean_sizes": ((1.0, 1.0, 1.0), (2.0, 1.0, 0.8)), "sample_counts": (128, 64, 32, 16),
                "num_proposals": 16, "proposal_radius": 0.5},
}
GT_BOXES = torch.tensor([[[5.0, 0.0, -1.0, 3.9, 1.6, 1.56, 0.3], [8.0, 2.0, -1.0, 0.8, 0.6, 1.73, 4.0]],
                         [[3.0, -2.0, -1.0, 3.9, 1.6, 1.56, 2.0], [0.0] * 7]])
GT_LABELS = torch.tensor([[0, 1], [0, -1]])


def scene(name: str, num_points: int = 400) -> torch.Tensor:
    if name == "pointpillars":
        return torch.rand(2, num_points, 3) * torch.tensor([12.8, 12.8, 4.0]) + torch.tensor([0.0, -6.4, -3.0])
    return torch.rand(2, num_points, 3) * torch.tensor([10.0, 6.0, 2.0]) + torch.tensor([1.0, -3.0, -2.0])


@pytest.mark.parametrize(("name", "model_cls", "config_cls"),
                         [("pointpillars", PointPillars, PointPillarsConfig), ("votenet", VoteNet, VoteNetConfig)])
def test_registry_and_config(name, model_cls, config_cls):
    assert ModelRegistry.get(name) is model_cls
    model = create_model({"name": name, **SMALL[name]})
    assert isinstance(model, model_cls) and isinstance(model.config, config_cls)
    with pytest.raises(ValueError):
        config_cls(input_dim=2)


def test_pointpillars_config_checks_grid_and_classes():
    with pytest.raises(ValueError):
        PointPillarsConfig(pillar_size=(0.3, 0.3))  # grid not divisible by the backbone stride
    with pytest.raises(ValueError):
        PointPillarsConfig(anchor_sizes=((1.0, 1.0, 1.0),))  # per-class tuples disagree


@pytest.mark.parametrize("name", ["pointpillars", "votenet"])
def test_forward_loss_and_padding(name):
    torch.manual_seed(0)
    model = create_model({"name": name, "input_dim": 4, **SMALL[name]})
    points = scene(name)
    mask = torch.ones(2, points.shape[1], dtype=torch.bool)
    mask[1, 300:] = False
    out = model(ModelInput(coordinates=points, point_features=torch.rand(2, points.shape[1], 1), point_mask=mask))
    count = model.anchors.shape[0] if name == "pointpillars" else model.config.num_proposals
    assert out.predictions.shape == (2, count, 7)

    losses = model.detection_loss(out, GT_BOXES, GT_LABELS)
    losses["loss"].backward()
    assert all(torch.isfinite(v) for v in losses.values())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    empty = model.detection_loss(out, torch.zeros(2, 0, 7), torch.zeros(2, 0))
    assert torch.isfinite(empty["loss"])
    with pytest.raises(ValueError):
        model(ModelInput(coordinates=points))  # missing the declared point feature


def test_pointpillars_box_coding_round_trip():
    model = create_model({"name": "pointpillars", **SMALL["pointpillars"]})
    for box in GT_BOXES[0]:
        target = box.expand(model.anchors.shape[0], 7)
        direction = torch.zeros(target.shape[0], 2)
        direction[:, int(box[6] % (2 * math.pi) >= math.pi)] = 1
        assert torch.allclose(model.decode(model.encode(target), direction), target, atol=1e-4)


def test_pointpillars_matches_box_and_learns():
    torch.manual_seed(0)
    model = create_model({"name": "pointpillars", "input_dim": 3, **SMALL["pointpillars"]})
    points = scene("pointpillars")
    boxes, labels = GT_BOXES[:1].expand(2, -1, -1), GT_LABELS[:1].expand(2, -1)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(15):
        loss = model.detection_loss(model(ModelInput(coordinates=points)), boxes, labels)["loss"]
        first = first if first is not None else loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert loss.item() < first


def test_votenet_heading_and_size_targets_decode_back():
    model = create_model({"name": "votenet", **SMALL["votenet"]})
    bins = model.config.num_heading_bins
    width = 2 * math.pi / bins
    box = torch.tensor([[2.0, 1.0, 0.5, 2.4, 1.1, 0.7, 2.0]])
    shifted = (box[:, 6] + width / 2) % (2 * math.pi)
    heading_bin = (shifted // width).long()
    residual = (shifted - heading_bin * width - width / 2) / (width / 2)
    parts = {
        "heading_logits": torch.nn.functional.one_hot(heading_bin, bins).float().unsqueeze(0),
        "heading_residual": torch.full((1, 1, bins), residual.item()),
        "size_logits": torch.tensor([[[0.0, 1.0]]]),
        "size_residual": ((box[:, 3:6] - model.mean_sizes[1]) / model.mean_sizes[1]).expand(2, 3)[None, None],
        "center_offset": torch.zeros(1, 1, 3),
    }
    assert torch.allclose(model.decode(box[None, :, :3], parts), box.unsqueeze(0), atol=1e-5)
