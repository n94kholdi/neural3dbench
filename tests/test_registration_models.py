from __future__ import annotations

import math

import pytest
import torch

from models import DCP, DCPConfig, ModelInput, ModelRegistry, create_model

SMALL = {"name": "dcp", "neighbor_count": 8, "edge_channels": (16, 16), "embedding_dim": 32, "ff_dim": 64}


def pair(batch: int = 2, num_points: int = 64):
    torch.manual_seed(0)
    source = torch.randn(batch, num_points, 3)
    angle = torch.rand(batch) * math.pi / 4
    cos, sin, zero, one = angle.cos(), angle.sin(), torch.zeros(batch), torch.ones(batch)
    rotation = torch.stack([cos, -sin, zero, sin, cos, zero, zero, zero, one], -1).reshape(batch, 3, 3)
    translation = torch.rand(batch, 3) - 0.5
    return source, source @ rotation.transpose(1, 2) + translation.unsqueeze(1), rotation, translation


def test_registry_and_config():
    assert ModelRegistry.get("dcp") is DCP
    assert isinstance(create_model(SMALL).config, DCPConfig)
    with pytest.raises(ValueError):
        DCPConfig(embedding_dim=30, num_heads=4)


def test_requires_target_cloud():
    model = create_model(SMALL)
    with pytest.raises(ValueError, match="query_coordinates"):
        model(ModelInput(coordinates=torch.randn(1, 16, 3)))
    with pytest.raises(ValueError):
        model(ModelInput(coordinates=torch.randn(1, 16, 3), query_coordinates=torch.randn(1, 16, 2)))


def test_forward_shapes_and_proper_rotation():
    source, target, _, _ = pair()
    output = create_model(SMALL).eval()(ModelInput(coordinates=source, query_coordinates=target[:, :48]))
    rotation = output.auxiliary["rotation"]
    assert output.predictions.shape == (2, 4, 4)
    assert output.auxiliary["scores"].shape == (2, 64, 48)
    assert torch.allclose(rotation @ rotation.transpose(1, 2), torch.eye(3).expand(2, 3, 3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(2), atol=1e-5)


def test_target_order_and_source_padding_do_not_change_pose():
    source, target, _, _ = pair()
    model = create_model(SMALL).eval()
    base = model(ModelInput(coordinates=source, query_coordinates=target)).predictions
    shuffled = model(ModelInput(coordinates=source, query_coordinates=target[:, torch.randperm(64)])).predictions
    assert torch.allclose(base, shuffled, atol=1e-4)

    padded = torch.cat([source, torch.full((2, 10, 3), 50.0)], dim=1)
    mask = torch.cat([torch.ones(2, 64, dtype=torch.bool), torch.zeros(2, 10, dtype=torch.bool)], dim=1)
    masked = model(ModelInput(coordinates=padded, query_coordinates=target, point_mask=mask)).predictions
    assert torch.allclose(base, masked, atol=1e-4)


def test_loss_decreases_on_fixed_pair():
    source, target, rotation, translation = pair()
    model = create_model(SMALL)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        loss = model.registration_loss(model(ModelInput(coordinates=source, query_coordinates=target)),
                                       rotation, translation)["total"]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < 0.5 * losses[0]
