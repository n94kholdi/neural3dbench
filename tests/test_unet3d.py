from __future__ import annotations

import io

import pytest
import torch

from models import GridRepresentationAdapter, ModelInput, ModelRegistry, UNet3D, UNet3DConfig, create_model
from volumetric import point_cloud_to_voxel_grid, voxelize_points


def _model(**kwargs) -> UNet3D:
    config = UNet3DConfig(
        input_channels=kwargs.pop("input_channels", 2),
        output_channels=kwargs.pop("output_channels", 1),
        base_channels=kwargs.pop("base_channels", 4),
        levels=kwargs.pop("levels", 3),
        normalization=kwargs.pop("normalization", "group"),
        **kwargs,
    )
    return UNet3D(config)


def test_unet3d_registry_config_and_initialization():
    model = create_model(
        {
            "name": "unet3d",
            "input_channels": 4,
            "output_channels": 3,
            "base_channels": 4,
            "levels": 3,
            "normalization": "instance",
            "activation": "silu",
            "upsampling": "interpolate",
        }
    )
    assert isinstance(model, UNet3D)
    assert model.config.input_channels == 4
    assert model.config.output_channels == 3
    assert ModelRegistry.get("unet3d") is UNet3D
    assert model.capabilities.representation == "GRID"


@pytest.mark.parametrize(
    ("shape", "output_channels"),
    [
        ((2, 3, 16, 16, 16), 1),
        ((1, 2, 17, 23, 19), 4),
        ((2, 1, 12, 20, 16), 2),
    ],
)
def test_forward_preserves_batch_and_spatial_shape(shape, output_channels):
    model = _model(input_channels=shape[1], output_channels=output_channels).eval()
    inputs = ModelInput(voxel_fields=torch.randn(shape), representation="GRID")
    with torch.no_grad():
        output = model(inputs)
    assert output.predictions.shape == (shape[0], output_channels, *shape[-3:])
    assert output.metadata["spatial_shape"] == shape[-3:]


def test_irregular_dimensions_and_backward_pass():
    model = _model(input_channels=2, output_channels=3)
    volume = torch.randn(1, 2, 31, 47, 35)
    prediction = model(ModelInput(voxel_fields=volume)).predictions
    assert prediction.shape == (1, 3, 31, 47, 35)
    prediction.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_strided_downsampling_and_transpose_upsampling():
    model = _model(downsampling="strided_conv", upsampling="transpose_conv").eval()
    with torch.no_grad():
        prediction = model(ModelInput(voxel_fields=torch.randn(1, 2, 15, 17, 19))).predictions
    assert prediction.shape == (1, 1, 15, 17, 19)


def test_grid_adapter_accepts_tensor_and_geometry_mask():
    volume = torch.randn(2, 3, 8, 9, 10)
    mask = torch.ones(2, 1, 8, 9, 10, dtype=torch.bool)
    direct = GridRepresentationAdapter().adapt(volume)
    mapped = GridRepresentationAdapter().adapt({"grid": volume, "geometry_mask": mask})
    assert direct.voxel_fields is volume
    assert mapped.voxel_fields is volume
    assert mapped.boundary_mask is mask


def test_serialization_round_trip():
    torch.manual_seed(5)
    model = _model().eval()
    inputs = ModelInput(voxel_fields=torch.randn(1, 2, 11, 13, 15))
    with torch.no_grad():
        expected = model(inputs).predictions
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _model().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        actual = restored(inputs).predictions
    torch.testing.assert_close(actual, expected)


def test_invalid_inputs_and_config_are_rejected():
    model = _model(input_channels=2)
    with pytest.raises(ValueError, match=r"\[B, C, D, H, W\]"):
        model(ModelInput(voxel_fields=torch.randn(2, 8, 8, 8)))
    with pytest.raises(ValueError, match="expects 2"):
        model(ModelInput(voxel_fields=torch.randn(1, 1, 8, 8, 8)))
    with pytest.raises(TypeError, match="floating-point"):
        model(ModelInput(voxel_fields=torch.ones(1, 2, 8, 8, 8, dtype=torch.int64)))
    with pytest.raises(ValueError, match="levels"):
        UNet3DConfig(levels=1)


def test_point_voxelization_occupancy_features_and_bounds():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.02, 0.02, 0.02]])
    features = torch.tensor([[2.0], [4.0], [6.0]])
    volume = voxelize_points(
        points,
        features=features,
        resolution=(3, 4, 5),
        bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )
    assert volume.shape == (1, 2, 3, 4, 5)
    assert volume[0, 0].sum() == 2
    assert volume[0, 1, 0, 0, 0] == pytest.approx(4.0)
    adapted = point_cloud_to_voxel_grid(points, resolution=8)
    assert adapted.voxel_fields.shape == (1, 1, 8, 8, 8)
    assert adapted.representation == "GRID"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_forward_and_backward():
    model = _model().cuda()
    inputs = ModelInput(voxel_fields=torch.randn(2, 2, 16, 16, 16, device="cuda"))
    output = model(inputs).predictions
    output.mean().backward()
    assert output.is_cuda
