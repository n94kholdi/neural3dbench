from __future__ import annotations

import io

import pytest
import torch

from models import FNO3D, FNO3DConfig, ModelInput, ModelRegistry, SpectralConv3d, create_model
from models.fno import normalized_coordinate_grid


def _model(**kwargs) -> FNO3D:
    config = FNO3DConfig(
        input_channels=kwargs.pop("input_channels", 2),
        output_channels=kwargs.pop("output_channels", 1),
        width=kwargs.pop("width", 4),
        num_layers=kwargs.pop("num_layers", 2),
        modes=kwargs.pop("modes", (2, 2, 2)),
        projection_channels=kwargs.pop("projection_channels", 8),
        **kwargs,
    )
    return FNO3D(config)


def test_registry_config_serialization_and_initialization():
    payload = {
        "name": "fno3d",
        "input_channels": 3,
        "output_channels": 2,
        "width": 6,
        "num_layers": 3,
        "modes": [2, 3, 4],
        "projection_channels": 12,
        "append_coordinates": False,
        "padding": [1, 2, 3],
    }
    model = create_model(payload)
    assert isinstance(model, FNO3D)
    assert ModelRegistry.get("fno3d") is FNO3D
    assert model.config.modes == (2, 3, 4)
    assert model.config.padding == (1, 2, 3)
    assert FNO3DConfig.from_dict(model.config.to_dict()) == model.config
    assert model.capabilities.representation == "GRID"
    assert model.required_inputs == {"voxel_fields"}


@pytest.mark.parametrize(
    ("shape", "output_channels", "coordinates"),
    [
        ((2, 3, 8, 10, 12), 1, True),
        ((1, 1, 7, 9, 11), 4, True),
        ((3, 2, 9, 8, 7), 2, False),
    ],
)
def test_forward_preserves_batch_and_odd_spatial_shape(shape, output_channels, coordinates):
    model = _model(
        input_channels=shape[1],
        output_channels=output_channels,
        append_coordinates=coordinates,
    ).eval()
    with torch.no_grad():
        output = model(ModelInput(voxel_fields=torch.randn(shape), representation="GRID"))
    assert output.predictions.shape == (shape[0], output_channels, *shape[-3:])
    assert output.metadata["spatial_shape"] == shape[-3:]
    assert torch.isfinite(output.predictions).all()


def test_coordinate_grid_order_range_device_and_dtype():
    x = torch.zeros(2, 1, 3, 4, 5, dtype=torch.float64)
    grid = normalized_coordinate_grid(x)
    assert grid.shape == (2, 3, 3, 4, 5)
    assert grid.dtype == x.dtype and grid.device == x.device
    torch.testing.assert_close(grid[0, 0, :, 0, 0], torch.linspace(0, 1, 3, dtype=x.dtype))
    torch.testing.assert_close(grid[0, 1, 0, :, 0], torch.linspace(0, 1, 4, dtype=x.dtype))
    torch.testing.assert_close(grid[0, 2, 0, 0, :], torch.linspace(0, 1, 5, dtype=x.dtype))
    assert not grid.requires_grad


def test_spectral_shapes_sparse_regions_and_corner_indexing():
    layer = SpectralConv3d(2, 3, (2, 2, 3))
    x = torch.randn(1, 2, 7, 9, 11)
    x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
    out_ft = layer.multiply_modes(x_ft)
    assert x_ft.shape == (1, 2, 7, 9, 6)
    assert out_ft.shape == (1, 3, 7, 9, 6)
    assert all(weight.shape == (2, 3, 2, 2, 3) for weight in layer.spectral_weights)

    mask = torch.zeros_like(out_ft, dtype=torch.bool)
    mask[:, :, :2, :2, :3] = True
    mask[:, :, -2:, :2, :3] = True
    mask[:, :, :2, -2:, :3] = True
    mask[:, :, -2:, -2:, :3] = True
    assert torch.count_nonzero(out_ft[~mask]) == 0
    assert all(torch.count_nonzero(out_ft[:, :, ds, hs, :3]) > 0 for ds, hs in (
        (slice(0, 2), slice(0, 2)),
        (slice(-2, None), slice(0, 2)),
        (slice(0, 2), slice(-2, None)),
        (slice(-2, None), slice(-2, None)),
    ))


def test_spectral_fixed_weight_matches_explicit_reference_and_odd_inverse_size():
    layer = SpectralConv3d(1, 1, (1, 1, 2))
    with torch.no_grad():
        layer.weights1.fill_(1 + 2j)
        layer.weights2.fill_(2 - 1j)
        layer.weights3.fill_(-1 + 0.5j)
        layer.weights4.fill_(0.25 - 0.75j)
    x = torch.arange(5 * 7 * 9, dtype=torch.float32).reshape(1, 1, 5, 7, 9) / 100
    actual = layer(x)

    x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
    reference_ft = torch.zeros(1, 1, 5, 7, 5, dtype=x_ft.dtype)
    reference_ft[:, :, :1, :1, :2] = x_ft[:, :, :1, :1, :2] * (1 + 2j)
    reference_ft[:, :, -1:, :1, :2] = x_ft[:, :, -1:, :1, :2] * (2 - 1j)
    reference_ft[:, :, :1, -1:, :2] = x_ft[:, :, :1, -1:, :2] * (-1 + 0.5j)
    reference_ft[:, :, -1:, -1:, :2] = x_ft[:, :, -1:, -1:, :2] * (0.25 - 0.75j)
    expected = torch.fft.irfftn(reference_ft, s=(5, 7, 9), dim=(-3, -2, -1))
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)


def test_backward_reaches_every_complex_frequency_block():
    torch.manual_seed(4)
    model = _model(width=3, num_layers=3, modes=(2, 2, 2))
    prediction = model(ModelInput(voxel_fields=torch.randn(2, 2, 7, 9, 11))).predictions
    prediction.square().mean().backward()
    for block in model.blocks:
        for weight in block.spectral.spectral_weights:
            assert weight.grad is not None
            assert torch.isfinite(weight.grad).all()
            assert torch.count_nonzero(weight.grad) > 0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_padding_is_cropped_to_original_resolution():
    model = _model(padding=(1, 2, 3)).eval()
    x = torch.randn(1, 2, 7, 9, 11)
    with torch.no_grad():
        output = model(ModelInput(voxel_fields=x)).predictions
    assert output.shape == (1, 1, 7, 9, 11)


def test_checkpoint_round_trip():
    torch.manual_seed(5)
    model = _model().eval()
    inputs = ModelInput(voxel_fields=torch.randn(1, 2, 7, 9, 11))
    with torch.no_grad():
        expected = model(inputs).predictions
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _model().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs).predictions, expected)


def test_invalid_inputs_modes_and_config_are_rejected():
    model = _model(input_channels=2)
    with pytest.raises(ValueError, match=r"\[B, C, D, H, W\]"):
        model(ModelInput(voxel_fields=torch.randn(2, 8, 8, 8)))
    with pytest.raises(ValueError, match="expects 2"):
        model(ModelInput(voxel_fields=torch.randn(1, 1, 8, 8, 8)))
    with pytest.raises(TypeError, match="floating-point"):
        model(ModelInput(voxel_fields=torch.ones(1, 2, 8, 8, 8, dtype=torch.int64)))
    with pytest.raises(ValueError, match="representation"):
        model(ModelInput(voxel_fields=torch.randn(1, 2, 8, 8, 8), representation="GRAPH"))
    with pytest.raises(ValueError, match=r"modes\[0\]"):
        _model(modes=(5, 2, 2))(ModelInput(voxel_fields=torch.randn(1, 2, 8, 8, 8)))
    with pytest.raises(ValueError, match=r"modes\[1\]"):
        _model(modes=(2, 5, 2))(ModelInput(voxel_fields=torch.randn(1, 2, 8, 8, 8)))
    with pytest.raises(ValueError, match=r"modes\[2\]"):
        _model(modes=(2, 2, 6))(ModelInput(voxel_fields=torch.randn(1, 2, 8, 8, 8)))
    with pytest.raises(ValueError, match="positive"):
        FNO3DConfig(modes=(0, 2, 2))
    with pytest.raises(ValueError, match="dropout"):
        FNO3DConfig(dropout=1.0)


def test_cpu_execution():
    output = _model()(ModelInput(voxel_fields=torch.randn(1, 2, 6, 8, 10))).predictions
    assert output.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_forward_and_backward():
    model = _model().cuda()
    x = torch.randn(2, 2, 8, 8, 8, device="cuda")
    output = model(ModelInput(voxel_fields=x)).predictions
    output.mean().backward()
    assert output.is_cuda and torch.isfinite(output).all()
    assert all(weight.grad is not None for block in model.blocks for weight in block.spectral.spectral_weights)
