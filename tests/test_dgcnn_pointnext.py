from __future__ import annotations

import io

import pytest
import torch

from models import (
    DGCNN,
    DGCNNConfig,
    ModelInput,
    ModelRegistry,
    PointNeXt,
    PointNeXtConfig,
    create_model,
)
from pointcloud import PointCloudSample, collate_point_clouds


def _config(name: str, task: str, *, input_dim: int = 3, output_dim: int = 4, **overrides):
    if name == "dgcnn":
        return DGCNNConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            task=task,
            neighbor_count=6,
            edge_channels=(16, 16, 32),
            embedding_dim=64,
            dropout=0.0,
            **overrides,
        )
    return PointNeXtConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        task=task,
        width=16,
        sample_counts=(12, 4),
        blocks=(2, 2, 1),
        radius=0.8,
        neighbor_count=8,
        dropout=0.0,
        **overrides,
    )


def _model(name: str, task: str, **kwargs):
    return create_model(_config(name, task, **kwargs))


MODELS = ["dgcnn", "pointnext"]


@pytest.mark.parametrize("name, model_class", [("dgcnn", DGCNN), ("pointnext", PointNeXt)])
def test_registry_and_dict_config(name: str, model_class):
    payload = {**_config(name, "classification", input_dim=5, output_dim=3).to_dict()}
    model = create_model(payload)
    assert isinstance(model, model_class)
    assert ModelRegistry.get(name) is model_class
    assert model.config.feature_dim == 2
    assert model.capabilities.variable_node_count


@pytest.mark.parametrize("name", MODELS)
def test_classification_with_optional_features(name: str):
    model = _model(name, "classification", input_dim=6, output_dim=5)
    output = model(
        ModelInput(coordinates=torch.randn(3, 20, 3), point_features=torch.randn(3, 20, 3))
    )
    assert output.predictions.shape == (3, 5)


@pytest.mark.parametrize("name", MODELS)
def test_segmentation_output_and_backward_pass(name: str):
    model = _model(name, "segmentation", output_dim=2)
    output = model(ModelInput(coordinates=torch.randn(2, 18, 3)))
    assert output.predictions.shape == (2, 18, 2)
    output.predictions.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("point_count", [3, 13])
def test_point_counts_smaller_than_neighbors_and_samples(name: str, point_count: int):
    model = _model(name, "segmentation", output_dim=1).eval()
    with torch.no_grad():
        output = model(ModelInput(coordinates=torch.randn(2, point_count, 3)))
    assert output.predictions.shape == (2, point_count, 1)


@pytest.mark.parametrize("name", MODELS)
def test_padded_batch_matches_unpadded_cloud(name: str):
    torch.manual_seed(31)
    short = torch.randn(7, 3)
    inputs, _ = collate_point_clouds(
        [PointCloudSample(short), PointCloudSample(torch.randn(15, 3))]
    )
    model = _model(name, "segmentation", output_dim=2).eval()
    with torch.no_grad():
        padded = model(inputs).predictions
        alone = model(ModelInput(coordinates=short.unsqueeze(0))).predictions
    assert torch.equal(padded[0, 7:], torch.zeros(8, 2))
    torch.testing.assert_close(padded[0, :7], alone[0], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", MODELS)
def test_classification_is_permutation_invariant(name: str):
    torch.manual_seed(32)
    model = _model(name, "classification", output_dim=3).eval()
    points = torch.randn(2, 20, 3)
    permutation = torch.randperm(points.shape[1])
    with torch.no_grad():
        expected = model(ModelInput(coordinates=points)).predictions
        actual = model(ModelInput(coordinates=points[:, permutation])).predictions
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("name", MODELS)
def test_batch_normalization_can_be_disabled(name: str):
    model = _model(name, "classification", output_dim=3, normalization="none")
    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()
    )
    assert model(ModelInput(coordinates=torch.randn(1, 16, 3))).predictions.shape == (1, 3)


@pytest.mark.parametrize("name", MODELS)
def test_state_dict_serialization_round_trip(name: str):
    torch.manual_seed(33)
    model = _model(name, "classification", output_dim=3).eval()
    inputs = ModelInput(coordinates=torch.randn(2, 14, 3))
    with torch.no_grad():
        expected = model(inputs).predictions
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _model(name, "classification", output_dim=3).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs).predictions, expected)


def test_invalid_config_and_inputs_are_rejected():
    with pytest.raises(ValueError, match="blocks"):
        PointNeXtConfig(sample_counts=(8, 4), blocks=(1, 1))
    with pytest.raises(ValueError, match="normalization"):
        DGCNNConfig(normalization="layer")
    with pytest.raises(ValueError, match="additional point features"):
        _model("dgcnn", "classification", input_dim=5)(
            ModelInput(coordinates=torch.randn(2, 8, 3))
        )
    with pytest.raises(TypeError, match="boolean"):
        _model("pointnext", "classification")(
            ModelInput(coordinates=torch.randn(2, 8, 3), point_mask=torch.ones(2, 8))
        )
