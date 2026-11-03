"""Pointwise MLP configuration, batching, training, and derivatives."""

from __future__ import annotations

import json

import pytest
import torch

from models import MLP, MLPConfig, ModelInput, ModelRegistry, create_model


@pytest.mark.parametrize("activation", ("tanh", "silu", "relu", "gelu"))
def test_mlp_activations_features_residuals_and_training_step(activation):
    model = MLP(
        MLPConfig(
            spatial_dim=3,
            input_dim=2,
            output_dim=2,
            widths=(5, 7, 7),
            activation=activation,
            residual=True,
        )
    )
    output = model(
        ModelInput(coordinates=torch.rand(3, 4, 3), point_features=torch.rand(3, 4, 2))
    )
    assert output.predictions.shape == (3, 4, 2)
    assert output.latent.shape == (3, 4, 7)
    before = model.output_layer.weight.detach().clone()
    output.predictions.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert not torch.equal(model.output_layer.weight, before)
    assert model.hidden_layers[1].residual is False
    assert model.hidden_layers[2].residual is True


def test_mlp_config_json_round_trip_and_registry_creation():
    config = MLPConfig(
        spatial_dim=2,
        input_dim=1,
        output_dim=3,
        widths=(11, 9),
        activation="SiLU",
        residual=True,
        metadata={"purpose": "round-trip"},
    )
    payload = json.loads(json.dumps(config.to_dict()))
    restored = MLPConfig.from_dict(payload)
    assert restored == config
    assert restored.widths == (11, 9)
    assert ModelRegistry.get("mlp") is MLP

    model = create_model(payload)
    assert isinstance(model, MLP)
    assert model.config == config
    assert model.capabilities.requires == {"coordinates"}
    assert model.capabilities.supports == {"point_features", "point_mask"}


def test_mlp_first_second_and_third_coordinate_derivatives_train_parameters():
    model = MLP(
        MLPConfig(spatial_dim=2, widths=(8, 8), activation="tanh", residual=True)
    ).double()
    coordinates = torch.rand(2, 4, 2, dtype=torch.double, requires_grad=True)
    field = model(ModelInput(coordinates=coordinates)).predictions[..., 0]
    first = torch.autograd.grad(field.sum(), coordinates, create_graph=True)[0]
    second = torch.autograd.grad(first[..., 0].sum(), coordinates, create_graph=True)[0]
    third = torch.autograd.grad(second[..., 0].sum(), coordinates, create_graph=True)[0]
    assert first.shape == second.shape == third.shape == (2, 4, 2)
    assert all(torch.isfinite(t).all() for t in (field, first, second, third))

    loss = field.square().mean() + third.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_mlp_points_are_independent_and_masked_padding_is_zero():
    torch.manual_seed(5)
    model = MLP(MLPConfig(spatial_dim=3, widths=(16, 16))).eval()
    coordinates = torch.randn(2, 8, 3)
    mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 8])
    padded = model(ModelInput(coordinates=coordinates, point_mask=mask))
    unpadded = model(ModelInput(coordinates=coordinates[:1, :5]))
    torch.testing.assert_close(padded.predictions[0, :5], unpadded.predictions[0])
    assert torch.count_nonzero(padded.predictions[0, 5:]) == 0
    assert padded.metadata["point_mask"] is not None


@pytest.mark.parametrize(
    "overrides",
    (
        {"widths": ()},
        {"widths": (8, 0)},
        {"activation": "sin"},
        {"residual": 1},
        {"input_dim": -1},
    ),
)
def test_mlp_config_rejects_invalid_options(overrides):
    with pytest.raises((TypeError, ValueError)):
        MLPConfig(**overrides)
