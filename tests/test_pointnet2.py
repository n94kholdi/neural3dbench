from __future__ import annotations

import io

import pytest
import torch

from models import ModelInput, ModelRegistry, PointNet2, PointNet2Config, create_model
from models.pointnet2 import (
    farthest_point_sample,
    query_ball_point,
)
from pointcloud import PointCloudSample, collate_point_clouds


def _config(task: str, *, input_dim: int = 3, output_dim: int = 4):
    return PointNet2Config(
        input_dim=input_dim,
        output_dim=output_dim,
        task=task,
        sample_counts=(12, 4),
        radii=(0.8, 1.6),
        neighbor_counts=(8, 8),
        abstraction_channels=((16, 32), (32, 64)),
        global_dim=64,
        dropout=0.0,
    )


def _model(task: str, *, input_dim: int = 3, output_dim: int = 4) -> PointNet2:
    return PointNet2(_config(task, input_dim=input_dim, output_dim=output_dim))


def test_pointnet2_registry_config_and_alias():
    payload = {
        **_config("classification", input_dim=5, output_dim=3).to_dict(),
        "name": "pointnet++",
    }
    model = create_model(payload)
    assert isinstance(model, PointNet2)
    assert model.config.feature_dim == 2
    assert ModelRegistry.get("pointnet2") is PointNet2
    assert ModelRegistry.get("pointnet++") is PointNet2
    assert model.capabilities.variable_node_count


def test_sampling_and_radius_grouping_respect_masks():
    points = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    mask = torch.tensor([[True, True, True, False]])
    indices, sampled_mask = farthest_point_sample(points, 4, mask)
    assert indices.shape == (1, 4)
    assert sampled_mask.tolist() == [[True, True, True, False]]
    assert not torch.any(indices[:, :3] == 3)

    centroids = points[:, [0]]
    neighbors, neighbor_mask = query_ball_point(
        1.1, 4, points, centroids, point_mask=mask
    )
    assert neighbors.shape == (1, 1, 4)
    assert neighbor_mask.sum().item() == 2
    assert set(neighbors[neighbor_mask].tolist()) == {0, 1}


def test_classification_with_optional_features():
    model = _model("classification", input_dim=6, output_dim=5)
    output = model(
        ModelInput(
            coordinates=torch.randn(3, 20, 3),
            point_features=torch.randn(3, 20, 3),
        )
    )
    assert output.predictions.shape == (3, 5)
    assert output.latent.shape == (3, 64)
    assert len(output.auxiliary["sampled_coordinates"]) == 3


def test_segmentation_output_and_backward_pass():
    model = _model("segmentation", output_dim=2)
    output = model(ModelInput(coordinates=torch.randn(2, 18, 3)))
    assert output.predictions.shape == (2, 18, 2)
    output.predictions.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("point_count", [5, 13, 25])
def test_variable_point_counts_including_smaller_than_sampling_levels(point_count: int):
    model = _model("segmentation", output_dim=1).eval()
    with torch.no_grad():
        output = model(ModelInput(coordinates=torch.randn(2, point_count, 3)))
    assert output.predictions.shape == (2, point_count, 1)


def test_padded_batch_mask_and_pointwise_output():
    inputs, _ = collate_point_clouds(
        [PointCloudSample(torch.randn(7, 3)), PointCloudSample(torch.randn(15, 3))]
    )
    model = _model("segmentation", output_dim=2).eval()
    with torch.no_grad():
        output = model(inputs)
    assert output.predictions.shape == (2, 15, 2)
    assert torch.equal(output.predictions[0, 7:], torch.zeros(8, 2))
    assert output.auxiliary["sampled_masks"][0][0].sum().item() == 7


def test_classification_is_permutation_invariant():
    torch.manual_seed(21)
    model = _model("classification", output_dim=3).eval()
    points = torch.randn(2, 20, 3)
    permutation = torch.randperm(points.shape[1])
    with torch.no_grad():
        expected = model(ModelInput(coordinates=points)).predictions
        actual = model(ModelInput(coordinates=points[:, permutation])).predictions
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_segmentation_is_permutation_equivariant():
    torch.manual_seed(22)
    model = _model("segmentation", output_dim=2).eval()
    points = torch.randn(2, 20, 3)
    permutation = torch.randperm(points.shape[1])
    with torch.no_grad():
        expected = model(ModelInput(coordinates=points)).predictions
        actual = model(ModelInput(coordinates=points[:, permutation])).predictions
    torch.testing.assert_close(
        actual, expected[:, permutation], rtol=2e-5, atol=2e-6
    )


def test_state_dict_serialization_round_trip():
    torch.manual_seed(23)
    model = _model("classification", output_dim=3).eval()
    inputs = ModelInput(coordinates=torch.randn(2, 14, 3))
    with torch.no_grad():
        expected = model(inputs).predictions
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _model("classification", output_dim=3).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        actual = restored(inputs).predictions
    torch.testing.assert_close(actual, expected)


def test_invalid_config_and_inputs_are_rejected():
    with pytest.raises(ValueError, match="exactly two"):
        PointNet2Config(sample_counts=(8,))
    model = _model("classification", input_dim=5)
    with pytest.raises(ValueError, match="additional point features"):
        model(ModelInput(coordinates=torch.randn(2, 8, 3)))
    with pytest.raises(TypeError, match="boolean"):
        _model("classification")(
            ModelInput(
                coordinates=torch.randn(2, 8, 3),
                point_mask=torch.ones(2, 8),
            )
        )
