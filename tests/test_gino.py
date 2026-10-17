from __future__ import annotations

import io

import pytest
import torch

from models import GINO, GINOConfig, ModelInput, ModelRegistry, PointCloudRepresentationAdapter, create_model


def _config(**overrides) -> GINOConfig:
    values = {
        "input_dim": 2,
        "output_dim": 1,
        "latent_resolution": (4, 4, 4),
        "width": 4,
        "modes": (1, 1, 1),
        "num_layers": 2,
        "kernel_radius": 0.8,
        "kernel_hidden_dim": 8,
        "kernel_chunk_size": 16,
        "kernel_max_pairs": 64,
    }
    values.update(overrides)
    return GINOConfig(**values)


def _input(batch: int = 2, count: int = 7) -> ModelInput:
    return ModelInput(
        coordinates=torch.rand(batch, count, 3),
        point_features=torch.randn(batch, count, 2),
        representation="POINT_CLOUD",
    )


def test_config_serialization_registry_imports_and_factory():
    payload = {**_config().to_dict(), "name": "gino", "latent_resolution": [4, 5, 6], "modes": [1, 2, 2]}
    model = create_model(payload)
    assert isinstance(model, GINO)
    assert ModelRegistry.get("gino") is GINO
    assert model.config.latent_resolution == (4, 5, 6)
    assert model.config.modes == (1, 2, 2)
    assert GINOConfig.from_dict(model.config.to_dict()) == model.config
    assert isinstance(create_model(GINOConfig()), GINO)
    assert model.capabilities.representation == "POINT_CLOUD"
    assert model.required_inputs == {"coordinates", "point_features"}


def test_minimal_forward_returns_finite_query_point_predictions():
    model = GINO(_config()).eval()
    queries = torch.rand(2, 5, 3)
    with torch.no_grad():
        output = model(_input().__class__(
            coordinates=_input().coordinates,
            point_features=_input().point_features,
            query_coordinates=queries,
            representation="POINT_CLOUD",
        ))
    assert output.predictions.shape == (2, 5, 1)
    assert output.latent.shape == (2, 4, 4, 4, 4)
    assert torch.isfinite(output.predictions).all()
    assert output.metadata["query_coordinates"].shape == (2, 5, 3)


def test_query_coordinates_default_to_sources_and_adapter_forwards_queries():
    base = _input(batch=1)
    adapted = PointCloudRepresentationAdapter().adapt(
        {
            "points": base.coordinates,
            "features": base.point_features,
            "query_coordinates": torch.rand(1, 3, 3),
        }
    )
    assert adapted.query_coordinates.shape == (1, 3, 3)
    model = GINO(_config()).eval()
    with torch.no_grad():
        output = model(base)
    assert output.predictions.shape == (1, base.coordinates.shape[1], 1)
    assert output.metadata["query_is_source"] is True
    torch.testing.assert_close(output.metadata["query_coordinates"], base.coordinates)


def test_missing_malformed_inputs_modes_and_representation_fail_clearly():
    model = GINO(_config())
    with pytest.raises(ValueError, match="point_features"):
        model(ModelInput(coordinates=torch.rand(1, 4, 3)))
    with pytest.raises(ValueError, match=r"\[B, N, 3\]"):
        model(ModelInput(coordinates=torch.rand(4, 3), point_features=torch.rand(4, 2)))
    with pytest.raises(ValueError, match="point_features must have shape"):
        model(ModelInput(coordinates=torch.rand(1, 4, 3), point_features=torch.rand(1, 4, 3)))
    with pytest.raises(TypeError, match="floating-point"):
        model(ModelInput(coordinates=torch.ones(1, 4, 3, dtype=torch.long), point_features=torch.rand(1, 4, 2)))
    with pytest.raises(TypeError, match="point_features must use a floating-point dtype"):
        model(ModelInput(coordinates=torch.rand(1, 4, 3), point_features=torch.ones(1, 4, 2, dtype=torch.long)))
    with pytest.raises(ValueError, match="representation"):
        model(ModelInput(**{**_input(batch=1).__dict__, "representation": "GRID"}))
    with pytest.raises(ValueError, match=r"normalized to \[0, 1\]"):
        model(ModelInput(coordinates=torch.full((1, 4, 3), 2.0), point_features=torch.rand(1, 4, 2)))
    with pytest.raises(ValueError, match="query_coordinates must have shape"):
        model(ModelInput(coordinates=torch.rand(1, 4, 3), point_features=torch.rand(1, 4, 2), query_coordinates=torch.rand(4, 3)))
    with pytest.raises(ValueError, match="batch size must match"):
        model(ModelInput(coordinates=torch.rand(1, 4, 3), point_features=torch.rand(1, 4, 2), query_coordinates=torch.rand(2, 3, 3)))
    with pytest.raises(TypeError, match="query_coordinates must use a floating-point dtype"):
        model(ModelInput(coordinates=torch.rand(1, 4, 3), point_features=torch.rand(1, 4, 2), query_coordinates=torch.ones(1, 3, 3, dtype=torch.long)))
    with pytest.raises(ValueError, match="positive integer"):
        _config(latent_resolution=(4, 0, 4))
    with pytest.raises(ValueError, match="input_dim must be a positive integer"):
        _config(input_dim=0)
    with pytest.raises(ValueError, match=r"modes\[0\]"):
        _config(latent_resolution=4, modes=(3, 1, 1))
    with pytest.raises(ValueError, match="kernel_radius"):
        _config(kernel_radius=0.0)


def test_gradients_reach_encoder_fourier_operator_and_decoder():
    torch.manual_seed(9)
    model = GINO(_config())
    prediction = model(_input(batch=1)).predictions
    prediction.square().mean().backward()
    groups = (
        model.encoder.kernel[-1].weight,
        model.blocks[0].spectral.weights1,
        model.decoder.kernel[-1].weight,
        model.output_projection.weight,
    )
    for parameter in groups:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_batch_masks_are_independent_and_masked_default_predictions_are_zero():
    model = GINO(_config()).eval()
    inputs = _input(batch=2, count=6)
    mask = torch.tensor([[True, True, True, False, False, False], [True] * 6])
    inputs.point_mask = mask
    changed_features = inputs.point_features.clone()
    changed_features[0, 3:] = 1000.0
    changed_coordinates = inputs.coordinates.clone()
    changed_coordinates[0, 3:] = -10.0
    altered = ModelInput(
        coordinates=changed_coordinates,
        point_features=changed_features,
        point_mask=mask,
    )
    with torch.no_grad():
        first = model(inputs).predictions
        second = model(altered).predictions
    torch.testing.assert_close(first[0, :3], second[0, :3])
    assert torch.count_nonzero(first[0, 3:]) == 0
    assert first.shape == (2, 6, 1)


def test_cpu_execution_and_checkpoint_round_trip():
    torch.manual_seed(12)
    model = GINO(_config()).eval()
    inputs = _input(batch=1)
    with torch.no_grad():
        expected = model(inputs).predictions
    assert expected.device.type == "cpu"
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = GINO(_config()).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs).predictions, expected)