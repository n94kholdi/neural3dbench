from __future__ import annotations

import pytest
import torch

from models import DeepNURBS, DeepNURBSConfig, ModelInput, ModelRegistry, create_model


def _input(coordinates: torch.Tensor | None = None) -> ModelInput:
    if coordinates is None:
        coordinates = torch.tensor(
            [[[0.0, 0.0], [0.25, 0.5], [1.0, 1.0]]], dtype=torch.float32
        )
    return ModelInput(coordinates=coordinates, representation="PARAMETRIC_SURFACE")


def test_config_factory_and_public_exports():
    config = DeepNURBSConfig(hidden_dim=8, num_hidden_layers=2)
    model = create_model(config)
    assert isinstance(model, DeepNURBS)
    assert ModelRegistry.get("deep_nurbs") is DeepNURBS
    assert DeepNURBSConfig.from_dict(config.to_dict()) == config
    assert isinstance(create_model({"name": "deep_nurbs", "hidden_dim": 8}), DeepNURBS)
    assert model.required_inputs == {"coordinates"}
    assert model.capabilities.representation == "PARAMETRIC_SURFACE"


def test_config_can_be_built_from_parametric_surface_mapping():
    default = DeepNURBSConfig()
    raw = {
        "representation": "PARAMETRIC_SURFACE",
        "degree": default.degree,
        "knot_vector_u": default.knot_vectors[0],
        "knot_vector_v": default.knot_vectors[1],
        "control_points": default.control_points,
        "weights": default.weights,
        "admissible_coefficients": default.admissible_coefficients,
        "lifting_coefficients": default.lifting_coefficients,
    }
    adapted = DeepNURBSConfig.from_parametric_surface(raw, hidden_dim=12)
    assert adapted.hidden_dim == 12
    assert adapted.control_points == default.control_points
    with pytest.raises(ValueError, match="missing"):
        DeepNURBSConfig.from_parametric_surface({"representation": "PARAMETRIC_SURFACE"})


def test_forward_contract_surface_and_boundary_values():
    model = create_model("deep_nurbs", hidden_dim=8).eval()
    output = model(_input())
    assert output.predictions.shape == (1, 3, 1)
    assert output.latent.shape == (1, 3, 1)
    assert output.main_prediction is output.predictions
    assert output.metadata["physical_coordinates"].shape == (1, 3, 2)
    torch.testing.assert_close(
        output.metadata["physical_coordinates"], _input().coordinates
    )
    torch.testing.assert_close(output.predictions[:, (0, 2)], torch.zeros(1, 2, 1))


def test_nurbs_surface_and_network_are_differentiable():
    model = DeepNURBS(DeepNURBSConfig(hidden_dim=8))
    coordinates = torch.tensor([[[0.3, 0.4], [0.6, 0.7]]], requires_grad=True)
    output = model(_input(coordinates))
    loss = output.predictions.square().sum() + output.metadata[
        "physical_coordinates"
    ].sum()
    loss.backward()
    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all()
    assert torch.count_nonzero(coordinates.grad) > 0
    assert model.output_layer.weight.grad is not None
    assert torch.isfinite(model.output_layer.weight.grad).all()
    assert torch.count_nonzero(model.output_layer.weight.grad) > 0


def test_missing_malformed_and_out_of_domain_inputs_are_rejected():
    model = DeepNURBS()
    with pytest.raises(ValueError, match="coordinates"):
        model(ModelInput())
    with pytest.raises(ValueError, match=r"\[B, N, 2\]"):
        model(ModelInput(coordinates=torch.rand(4, 2)))
    with pytest.raises(TypeError, match="floating-point"):
        model(ModelInput(coordinates=torch.ones(1, 2, 2, dtype=torch.long)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model(ModelInput(coordinates=torch.tensor([[[1.01, 0.5]]])))


@pytest.mark.parametrize(
    "override, message",
    [
        ({"degree": (0, 2)}, "degree"),
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"activation": "relu"}, "activation"),
        (
            {"degree": (1, 2), "knot_vectors": ((0, 0, 0, 1, 1), (0, 0, 0, 1, 1, 1))},
            "multiplicity must not exceed",
        ),
        ({"weights": ((1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0))}, "strictly positive"),
        ({"knot_vectors": ((0, 0, 0, 0.8, 0.5, 1), (0, 0, 0, 1, 1, 1))}, "nondecreasing"),
        ({"admissible_coefficients": ((0, 0, 0), (0, 1, 1), (0, 0, 0))}, "zero on all four"),
    ],
)
def test_invalid_configurations_are_rejected(override, message):
    with pytest.raises(ValueError, match=message):
        DeepNURBSConfig(**override)


def test_endpoint_and_interior_knot_evaluation_are_finite():
    model = DeepNURBS().eval()
    points = torch.tensor(
        [[[0.0, 0.5], [0.5, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]]]
    )
    output = model(_input(points))
    assert torch.isfinite(output.predictions).all()
    assert torch.isfinite(output.metadata["physical_coordinates"]).all()
    torch.testing.assert_close(
        output.metadata["physical_coordinates"][0, 0], torch.tensor([0.0, 0.5])
    )
    torch.testing.assert_close(
        output.metadata["physical_coordinates"][0, 2], torch.tensor([1.0, 0.5])
    )


def test_rational_weights_and_nonhomogeneous_lifting_are_applied():
    config = DeepNURBSConfig(
        weights=((1.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 1.0)),
        lifting_coefficients=(
            ((2.0,), (2.0,), (2.0,)),
            ((0.0,), (0.0,), (0.0,)),
            ((0.0,), (0.0,), (0.0,)),
        ),
    )
    model = DeepNURBS(config).eval()
    coordinates = torch.tensor([[[0.0, 0.5], [0.25, 0.5]]])
    output = model(_input(coordinates))
    torch.testing.assert_close(output.predictions[0, 0, 0], torch.tensor(2.0))
    assert not torch.allclose(
        output.metadata["physical_coordinates"][0, 1], coordinates[0, 1]
    )
