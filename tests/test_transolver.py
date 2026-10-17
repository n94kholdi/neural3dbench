from __future__ import annotations

import io

import pytest
import torch

from models import ModelInput, ModelRegistry, Transolver, TransolverConfig, create_model
from models.transolver import PhysicsAttention
from pointcloud import pointwise_regression_loss


def _config(**overrides) -> TransolverConfig:
    values = dict(
        spatial_dim=3,
        input_dim=2,
        output_dim=4,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        num_slices=7,
        mlp_ratio=2,
        dropout=0.0,
    )
    values.update(overrides)
    return TransolverConfig(**values)


def test_config_registry_and_capabilities():
    model = create_model({"name": "transolver", **_config().to_dict()})
    assert isinstance(model, Transolver)
    assert ModelRegistry.get("transolver") is Transolver
    assert model.capabilities.variable_geometry
    assert model.capabilities.variable_node_count
    assert "point_features" in model.capabilities.supports


@pytest.mark.parametrize(
    ("spatial_dim", "input_dim", "output_dim", "layers", "slices"),
    [(2, 0, 1, 1, 4), (3, 5, 4, 3, 8)],
)
def test_forward_backward_and_configurable_shapes(
    spatial_dim: int, input_dim: int, output_dim: int, layers: int, slices: int
):
    model = Transolver(
        _config(
            spatial_dim=spatial_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=layers,
            num_slices=slices,
        )
    )
    coordinates = torch.randn(2, 23, spatial_dim)
    features = None if input_dim == 0 else torch.randn(2, 23, input_dim)
    prediction = model(
        ModelInput(coordinates=coordinates, point_features=features)
    ).predictions
    assert prediction.shape == (2, 23, output_dim)
    prediction.square().mean().backward()
    attention = model.blocks[0].attention
    assert attention.in_project_slice.weight.grad is not None
    assert attention.to_q.weight.grad is not None
    assert torch.isfinite(attention.in_project_slice.weight.grad).all()


def test_physics_attention_tensor_flow_and_normalization():
    torch.manual_seed(1)
    attention = PhysicsAttention(24, num_heads=3, num_slices=5)
    x = torch.randn(2, 11, 24, requires_grad=True)
    mask = torch.tensor([[True] * 7 + [False] * 4, [True] * 11])
    output, data = attention(x, mask, return_attention=True)

    assert output.shape == (2, 11, 24)
    assert data["slice_weights"].shape == (2, 3, 11, 5)
    assert data["slice_norm"].shape == (2, 3, 5)
    assert data["slice_tokens"].shape == (2, 3, 5, 8)
    assert data["attention_weights"].shape == (2, 3, 5, 5)
    assert data["output_tokens"].shape == (2, 3, 5, 8)
    torch.testing.assert_close(
        data["slice_weights"][0, :, :7].sum(-1), torch.ones(3, 7)
    )
    assert torch.count_nonzero(data["slice_weights"][0, :, 7:]) == 0
    torch.testing.assert_close(
        data["attention_weights"].sum(-1), torch.ones(2, 3, 5)
    )
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert attention.in_project_slice.weight.grad is not None
    assert attention.to_v.weight.grad is not None


def test_physics_attention_matches_official_equations_without_mask():
    torch.manual_seed(2)
    layer = PhysicsAttention(12, num_heads=3, num_slices=4).eval()
    x = torch.randn(2, 6, 12)
    actual = layer(x)

    batch, points, _ = x.shape
    x_mid = layer.in_project_x(x).reshape(batch, points, 3, 4).permute(0, 2, 1, 3)
    fx_mid = layer.in_project_fx(x).reshape(batch, points, 3, 4).permute(0, 2, 1, 3)
    weights = (layer.in_project_slice(x_mid) / layer.temperature).softmax(-1)
    tokens = torch.einsum("bhnd,bhng->bhgd", fx_mid, weights)
    tokens = tokens / (weights.sum(2) + layer.eps).unsqueeze(-1)
    q, k, v = layer.to_q(tokens), layer.to_k(tokens), layer.to_v(tokens)
    token_attention = (q @ k.transpose(-1, -2) * layer.scale).softmax(-1)
    desliced = torch.einsum("bhgd,bhng->bhnd", token_attention @ v, weights)
    expected = layer.to_out(desliced.permute(0, 2, 1, 3).reshape(batch, points, 12))
    torch.testing.assert_close(actual, expected)


def test_padding_does_not_influence_valid_points():
    torch.manual_seed(3)
    model = Transolver(_config(input_dim=0, output_dim=2)).eval()
    coordinates = torch.randn(2, 13, 3)
    mask = torch.tensor([[True] * 8 + [False] * 5, [True] * 13])
    altered = coordinates.clone()
    altered[0, 8:] = torch.randn(5, 3) * 1000
    with torch.no_grad():
        first = model(ModelInput(coordinates=coordinates, point_mask=mask)).predictions
        second = model(ModelInput(coordinates=altered, point_mask=mask)).predictions
    torch.testing.assert_close(first[0, :8], second[0, :8])
    assert torch.count_nonzero(first[0, 8:]) == 0


def test_checkpoint_round_trip():
    torch.manual_seed(4)
    config = _config(input_dim=0)
    model = Transolver(config).eval()
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


def test_invalid_inputs_and_config_are_rejected():
    with pytest.raises(ValueError, match="divisible"):
        TransolverConfig(hidden_dim=30, num_heads=8)
    model = Transolver(_config(input_dim=2))
    with pytest.raises(ValueError, match="expects 2"):
        model(ModelInput(coordinates=torch.randn(1, 8, 3)))
    with pytest.raises(ValueError, match="point_mask"):
        model(
            ModelInput(
                coordinates=torch.randn(1, 8, 3),
                point_features=torch.randn(1, 8, 2),
                point_mask=torch.ones(1, 8),
            )
        )


def test_masked_regression_losses_ignore_padding():
    prediction = torch.tensor([[[2.0], [4.0], [1000.0]]])
    target = torch.tensor([[[1.0], [2.0], [-1000.0]]])
    mask = torch.tensor([[True, True, False]])
    assert pointwise_regression_loss(prediction, target, point_mask=mask, loss="mse") == pytest.approx(2.5)
    assert pointwise_regression_loss(prediction, target, point_mask=mask, loss="mae") == pytest.approx(1.5)
    assert pointwise_regression_loss(
        prediction, target, point_mask=mask, loss="relative_l2"
    ) == pytest.approx(1.0)


def test_synthetic_field_training_loss_decreases():
    torch.manual_seed(9)
    coordinates = 2 * torch.rand(4, 32, 2) - 1
    target = (
        torch.sin(torch.pi * coordinates[..., 0])
        * torch.cos(torch.pi * coordinates[..., 1])
    ).unsqueeze(-1)
    model = Transolver(
        _config(
            spatial_dim=2,
            input_dim=0,
            output_dim=1,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            num_slices=4,
        )
    )
    inputs = ModelInput(coordinates=coordinates)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    initial = pointwise_regression_loss(model(inputs).predictions, target).detach()
    for _ in range(30):
        loss = pointwise_regression_loss(model(inputs).predictions, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = pointwise_regression_loss(model(inputs).predictions, target).detach()
    assert final < initial


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_forward_backward():
    model = Transolver(_config()).cuda()
    output = model(
        ModelInput(
            coordinates=torch.randn(2, 32, 3, device="cuda"),
            point_features=torch.randn(2, 32, 2, device="cuda"),
        )
    ).predictions
    output.mean().backward()
    assert output.is_cuda and torch.isfinite(output).all()
