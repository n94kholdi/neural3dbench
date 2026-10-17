from __future__ import annotations

import io
import math

import pytest
import torch

from models import (
    ModelInput,
    ModelRegistry,
    SineLayer,
    SIREN,
    SIRENConfig,
    create_model,
)
from models.siren import SineLayer as PackageSineLayer
from pointcloud import pointwise_regression_loss


def _config(**overrides) -> SIRENConfig:
    values = dict(
        spatial_dim=3,
        input_dim=0,
        output_dim=1,
        hidden_dim=32,
        num_hidden_layers=2,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    )
    values.update(overrides)
    return SIRENConfig(**values)


def test_registry_config_round_trip_public_exports_and_capabilities():
    config = _config(input_dim=2, output_dim=4)
    restored_config = SIRENConfig.from_dict(config.to_dict())
    assert restored_config == config
    assert restored_config.name == restored_config.model_type == "siren"
    assert restored_config.representation == "POINT_CLOUD"

    from_dict = create_model(config.to_dict())
    from_config = create_model(config)
    assert isinstance(from_dict, SIREN)
    assert isinstance(from_config, SIREN)
    assert ModelRegistry.get("siren") is SIREN
    assert PackageSineLayer is SineLayer
    assert from_dict.capabilities.requires == {"coordinates"}
    assert from_dict.capabilities.supports == {"point_features", "point_mask"}
    assert from_dict.capabilities.representation == "POINT_CLOUD"
    assert from_dict.capabilities.variable_geometry
    assert from_dict.capabilities.variable_node_count
    assert isinstance(from_dict.first_layer, SineLayer)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spatial_dim": 0}, "spatial_dim"),
        ({"input_dim": -1}, "input_dim"),
        ({"output_dim": 0}, "output_dim"),
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"num_hidden_layers": -1}, "num_hidden_layers"),
        ({"first_omega_0": 0.0}, "first_omega_0"),
        ({"hidden_omega_0": float("inf")}, "hidden_omega_0"),
    ],
)
def test_invalid_config_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SIRENConfig(**kwargs)
    with pytest.raises(TypeError, match="outermost_linear"):
        SIRENConfig(outermost_linear=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("batch", "points", "spatial_dim", "input_dim", "output_dim"),
    [(1, 1, 2, 0, 1), (2, 7, 3, 4, 5), (3, 2, 2, 1, 2)],
)
def test_output_shapes(batch, points, spatial_dim, input_dim, output_dim):
    model = SIREN(
        _config(
            spatial_dim=spatial_dim,
            input_dim=input_dim,
            output_dim=output_dim,
        )
    )
    coordinates = torch.randn(batch, points, spatial_dim)
    features = None if input_dim == 0 else torch.randn(batch, points, input_dim)
    output = model(ModelInput(coordinates=coordinates, point_features=features))
    assert output.predictions.shape == (batch, points, output_dim)
    assert output.latent.shape == (batch, points, model.config.hidden_dim)


def test_input_validation_errors_are_helpful():
    model = SIREN(_config(input_dim=2))
    with pytest.raises(ValueError, match="requires inputs: coordinates"):
        model(ModelInput())
    with pytest.raises(ValueError, match=r"\[B, N, 3\]"):
        model(ModelInput(coordinates=torch.randn(8, 3)))
    with pytest.raises(ValueError, match=r"\[B, N, 3\]"):
        model(ModelInput(coordinates=torch.randn(1, 8, 2)))
    with pytest.raises(TypeError, match="floating-point"):
        model(ModelInput(coordinates=torch.ones(1, 8, 3, dtype=torch.long)))
    with pytest.raises(ValueError, match="expects 2"):
        model(ModelInput(coordinates=torch.randn(1, 8, 3)))
    with pytest.raises(ValueError, match="point_features must have shape"):
        model(
            ModelInput(
                coordinates=torch.randn(1, 8, 3),
                point_features=torch.randn(1, 7, 2),
            )
        )
    with pytest.raises(TypeError, match="point_features.*floating-point"):
        model(
            ModelInput(
                coordinates=torch.randn(1, 8, 3),
                point_features=torch.ones(1, 8, 2, dtype=torch.long),
            )
        )
    with pytest.raises(ValueError, match="representation"):
        model(
            ModelInput(
                coordinates=torch.randn(1, 8, 3),
                point_features=torch.randn(1, 8, 2),
                representation="GRID",
            )
        )
    with pytest.raises(ValueError, match="point_mask must have shape"):
        model(
            ModelInput(
                coordinates=torch.randn(1, 8, 3),
                point_features=torch.randn(1, 8, 2),
                point_mask=torch.ones(1, 7, dtype=torch.bool),
            )
        )
    with pytest.raises(TypeError, match="boolean"):
        model(
            ModelInput(
                coordinates=torch.randn(1, 8, 3),
                point_features=torch.randn(1, 8, 2),
                point_mask=torch.ones(1, 8),
            )
        )
    with pytest.raises(ValueError, match="valid point"):
        model(
            ModelInput(
                coordinates=torch.randn(1, 8, 3),
                point_features=torch.randn(1, 8, 2),
                point_mask=torch.zeros(1, 8, dtype=torch.bool),
            )
        )


def test_layer_count_and_optional_sine_output():
    linear_model = SIREN(_config(num_hidden_layers=4, outermost_linear=True))
    sine_model = SIREN(_config(num_hidden_layers=0, outermost_linear=False))
    assert len(linear_model.hidden_layers) == 4
    assert sum(isinstance(module, SineLayer) for module in linear_model.modules()) == 5
    output = sine_model(ModelInput(coordinates=torch.randn(2, 5, 3))).predictions
    assert torch.all(output.abs() <= 1.0)


def test_paper_initialization_bounds_include_final_linear():
    torch.manual_seed(11)
    config = _config(
        spatial_dim=3,
        input_dim=2,
        hidden_dim=24,
        output_dim=3,
        first_omega_0=17.0,
        hidden_omega_0=41.0,
    )
    model = SIREN(config)
    first_bound = 1.0 / (config.spatial_dim + config.input_dim)
    hidden_bound = math.sqrt(6.0 / config.hidden_dim) / config.hidden_omega_0
    assert model.first_layer.linear.weight.abs().max() <= first_bound
    assert model.first_layer.linear.weight.abs().max() > first_bound * 0.5
    for layer in model.hidden_layers:
        assert layer.linear.weight.abs().max() <= hidden_bound
    assert model.output_layer.weight.abs().max() <= hidden_bound
    # This tight bound would reject nn.Linear's standard Kaiming initialization.
    assert model.hidden_layers[0].linear.weight.abs().max() < 0.03


def test_forward_parameter_backward_and_coordinate_derivatives_are_finite():
    torch.manual_seed(3)
    model = SIREN(_config(spatial_dim=2, output_dim=2))
    coordinates = torch.randn(2, 6, 2, requires_grad=True)
    prediction = model(ModelInput(coordinates=coordinates)).predictions
    gradient = torch.autograd.grad(prediction.sum(), coordinates, create_graph=True)[0]
    second = torch.autograd.grad(
        gradient[..., 0].sum(), coordinates, create_graph=True
    )[0][..., 0]
    loss = prediction.square().mean() + second.square().mean()
    loss.backward()

    assert gradient.shape == coordinates.shape
    assert second.shape == coordinates.shape[:2]
    assert gradient.requires_grad and second.requires_grad
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(second).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_masked_padding_is_zero_and_valid_points_match_unpadded_evaluation():
    torch.manual_seed(5)
    model = SIREN(_config(input_dim=1, output_dim=2)).eval()
    coordinates = torch.randn(2, 8, 3)
    features = torch.randn(2, 8, 1)
    mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 8])
    padded = model(
        ModelInput(
            coordinates=coordinates,
            point_features=features,
            point_mask=mask,
        )
    )
    unpadded = model(
        ModelInput(coordinates=coordinates[:1, :5], point_features=features[:1, :5])
    )
    torch.testing.assert_close(padded.predictions[0, :5], unpadded.predictions[0])
    assert torch.count_nonzero(padded.predictions[0, 5:]) == 0
    assert padded.metadata["point_mask"] is not None
    target = torch.randn_like(padded.predictions)
    loss = pointwise_regression_loss(
        padded.predictions, target, point_mask=padded.metadata["point_mask"]
    )
    assert torch.isfinite(loss)


def test_checkpoint_round_trip_preserves_predictions():
    torch.manual_seed(7)
    config = _config()
    model = SIREN(config).eval()
    inputs = ModelInput(coordinates=torch.randn(2, 9, 3))
    with torch.no_grad():
        expected = model(inputs).predictions
    buffer = io.BytesIO()
    torch.save({"config": config.to_dict(), "model_state_dict": model.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored = create_model(checkpoint["config"]).eval()
    restored.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs).predictions, expected)


def test_short_deterministic_synthetic_fit_decreases_materially():
    torch.manual_seed(13)
    coordinates = 2 * torch.rand(1, 128, 2) - 1
    target = (
        torch.sin(2.0 * torch.pi * coordinates[..., 0])
        * torch.cos(3.0 * torch.pi * coordinates[..., 1])
    ).unsqueeze(-1)
    model = SIREN(_config(spatial_dim=2, hidden_dim=32, num_hidden_layers=2))
    inputs = ModelInput(coordinates=coordinates)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = pointwise_regression_loss(model(inputs).predictions, target).detach()
    for _ in range(80):
        loss = pointwise_regression_loss(model(inputs).predictions, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = pointwise_regression_loss(model(inputs).predictions, target).detach()
    assert final < initial * 0.25


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_forward_backward():
    model = SIREN(_config(input_dim=2, output_dim=3)).cuda()
    output = model(
        ModelInput(
            coordinates=torch.randn(2, 16, 3, device="cuda"),
            point_features=torch.randn(2, 16, 2, device="cuda"),
        )
    ).predictions
    output.square().mean().backward()
    assert output.is_cuda and torch.isfinite(output).all()
