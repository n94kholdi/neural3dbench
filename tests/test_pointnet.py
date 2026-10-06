from __future__ import annotations

import io

import pytest
import torch

from mesh import TriangularMesh
from models import (
    ModelInput,
    ModelRegistry,
    PointCloudRepresentationAdapter,
    PointNet,
    PointNetConfig,
    create_model,
)
from models.pointnet import feature_transform_regularizer
from pointcloud import (
    PointCloudSample,
    collate_point_clouds,
    mesh_to_point_cloud,
    normalize_points,
    sample_points,
)


def _model(task: str, *, input_dim: int = 3, output_dim: int = 4) -> PointNet:
    return PointNet(
        PointNetConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            task=task,
            global_dim=128,
            dropout=0.0,
        )
    )


def test_pointnet_config_and_registry():
    model = create_model(
        {
            "name": "pointnet",
            "input_dim": 6,
            "output_dim": 5,
            "task": "classification",
            "global_dim": 128,
        }
    )
    assert isinstance(model, PointNet)
    assert model.config.feature_dim == 3
    assert ModelRegistry.get("pointnet") is PointNet
    assert model.capabilities.representation == "POINT_CLOUD"
    assert model.capabilities.variable_node_count


def test_point_cloud_representation_adapter_preserves_xyz_and_features():
    points = torch.randn(2, 9, 3)
    features = torch.randn(2, 9, 2)
    adapted = PointCloudRepresentationAdapter().adapt(
        {"points": points, "features": features}
    )
    assert adapted.representation == "POINT_CLOUD"
    assert adapted.coordinates is points
    assert adapted.point_features is features


def test_classification_output_and_optional_features():
    model = _model("classification", input_dim=7, output_dim=6)
    inputs = ModelInput(
        coordinates=torch.randn(3, 24, 3),
        point_features=torch.randn(3, 24, 4),
    )
    output = model(inputs)
    assert output.predictions.shape == (3, 6)
    assert output.latent.shape == (3, 128)
    assert output.auxiliary["input_transform"].shape == (3, 3, 3)
    assert output.auxiliary["feature_transform"].shape == (3, 64, 64)


def test_pointwise_output_and_backward_pass():
    model = _model("segmentation", output_dim=2)
    inputs = ModelInput(coordinates=torch.randn(2, 19, 3))
    output = model(inputs)
    assert output.predictions.shape == (2, 19, 2)
    loss = output.predictions.square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("point_count", [5, 17, 31])
def test_variable_point_counts_across_calls(point_count: int):
    model = _model("segmentation", output_dim=1).eval()
    with torch.no_grad():
        prediction = model(
            ModelInput(coordinates=torch.randn(2, point_count, 3))
        ).predictions
    assert prediction.shape == (2, point_count, 1)


def test_padded_batch_uses_mask_and_zeros_padded_predictions():
    inputs, targets = collate_point_clouds(
        [
            PointCloudSample(torch.randn(7, 3), target=torch.randn(7, 1)),
            PointCloudSample(torch.randn(11, 3), target=torch.randn(11, 1)),
        ]
    )
    model = _model("segmentation", output_dim=1).eval()
    with torch.no_grad():
        output = model(inputs)
    assert inputs.coordinates.shape == (2, 11, 3)
    assert inputs.point_mask.sum(dim=1).tolist() == [7, 11]
    assert torch.equal(output.predictions[0, 7:], torch.zeros(4, 1))
    assert isinstance(targets, list)


def test_classification_is_permutation_invariant():
    torch.manual_seed(3)
    model = _model("classification", output_dim=3).eval()
    points = torch.randn(2, 29, 3)
    permutation = torch.randperm(points.shape[1])
    with torch.no_grad():
        original = model(ModelInput(coordinates=points)).predictions
        permuted = model(ModelInput(coordinates=points[:, permutation])).predictions
    torch.testing.assert_close(original, permuted, rtol=1e-5, atol=1e-6)


def test_pointwise_prediction_is_permutation_equivariant():
    torch.manual_seed(4)
    model = _model("segmentation", output_dim=2).eval()
    points = torch.randn(2, 23, 3)
    permutation = torch.randperm(points.shape[1])
    with torch.no_grad():
        original = model(ModelInput(coordinates=points)).predictions
        permuted = model(ModelInput(coordinates=points[:, permutation])).predictions
    torch.testing.assert_close(
        original[:, permutation], permuted, rtol=1e-5, atol=1e-6
    )


def test_state_dict_serialization_round_trip():
    torch.manual_seed(5)
    model = _model("classification", output_dim=3).eval()
    inputs = ModelInput(coordinates=torch.randn(2, 16, 3))
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


def test_point_data_utilities_and_mesh_conversion():
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    normalized = normalize_points(points)
    torch.testing.assert_close(normalized.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0)
    assert torch.linalg.vector_norm(normalized, dim=1).max() == pytest.approx(1.0)

    sampled, features = sample_points(points, 5, torch.arange(3.0).unsqueeze(1))
    assert sampled.shape == (5, 3)
    assert features.shape == (5, 1)

    mesh = TriangularMesh(
        vertices=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        triangles=torch.tensor([[0, 1, 2]]),
        node_data={"temperature": torch.tensor([1.0, 2.0, 3.0])},
    )
    converted = mesh_to_point_cloud(mesh, feature_names=["temperature"])
    assert converted.coordinates.shape == (1, 3, 3)
    assert converted.point_features.shape == (1, 3, 1)
    assert torch.equal(mesh.vertices, torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))


def test_feature_transform_regularizer():
    identity = torch.eye(4).repeat(2, 1, 1)
    assert feature_transform_regularizer(identity).item() == pytest.approx(0.0)


def test_invalid_pointnet_inputs_are_rejected():
    model = _model("classification", input_dim=5)
    with pytest.raises(ValueError, match="additional point features"):
        model(ModelInput(coordinates=torch.randn(2, 8, 3)))
    with pytest.raises(ValueError, match=r"\[B, N, 3\]"):
        model(ModelInput(coordinates=torch.randn(8, 3)))
    with pytest.raises(TypeError, match="boolean"):
        _model("classification")(
            ModelInput(
                coordinates=torch.randn(2, 8, 3),
                point_mask=torch.ones(2, 8),
            )
        )
