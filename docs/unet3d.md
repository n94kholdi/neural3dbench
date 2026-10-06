# 3D U-Net

3D U-Net extends the encoder-decoder U-Net architecture to structured volumes.
Its encoder learns increasingly broad spatial context with `Conv3d` blocks and
downsampling. The decoder restores resolution and concatenates same-resolution
encoder features through skip connections. A task-neutral `1x1x1` head returns
raw logits or continuous values.

## Input and output contract

The model consumes `ModelInput.voxel_fields` in channels-first format:

```text
[batch, input_channels, depth, height, width]
```

It returns `ModelOutput.predictions` with shape:

```text
[batch, output_channels, depth, height, width]
```

Use one output channel for a scalar field, multiple channels for vector or
multi-physics fields, and `num_classes` channels for segmentation. The model
does not apply sigmoid, softmax, domain masks, or physics losses. A geometry
mask can be concatenated as an input channel, applied by a loss, or used during
post-processing.

## Architecture and configuration

At every level, `ConvBlock3D` applies two convolution-normalization-activation
sequences. The default downsampler is `MaxPool3d`, and the default decoder uses
`ConvTranspose3d`. Odd spatial sizes are supported: decoder features are
interpolated to the encoder skip size when pooling creates a one-voxel mismatch.
The input itself is never silently cropped or resized.

```yaml
model:
  name: unet3d
  input_channels: 4
  output_channels: 3
  base_channels: 32
  levels: 4
  channel_multiplier: 2
  kernel_size: 3
  normalization: batch       # batch | instance | group | none
  activation: relu           # relu | leaky_relu | elu | gelu | silu
  downsampling: max_pool      # max_pool | strided_conv
  upsampling: transpose_conv  # transpose_conv | interpolate
```

`levels` includes the bottleneck resolution. The feature widths are
`base_channels * channel_multiplier**level`. Reduce `base_channels`, `levels`,
batch size, or voxel resolution when memory is constrained.

## Inference

```python
import torch

from models import ModelInput, create_model

model = create_model({
    "name": "unet3d",
    "input_channels": 4,
    "output_channels": 3,
    "base_channels": 16,
    "levels": 4,
    "normalization": "group",
})
volume = torch.randn(2, 4, 31, 47, 35)
prediction = model(ModelInput(voxel_fields=volume, representation="GRID")).predictions
assert prediction.shape == (2, 3, 31, 47, 35)
```

For segmentation, pass the raw output to cross-entropy or apply softmax only
for evaluation. For scalar or vector regression, use the appropriate external
data/physics objective.

## Voxelizing points and meshes

Voxelization stays separate from the neural network:

```python
from volumetric import mesh_to_voxel_grid, point_cloud_to_voxel_grid

inputs = point_cloud_to_voxel_grid(
    points,
    features=point_features,
    resolution=64,
    bounds=((-1, -1, -1), (1, 1, 1)),
)
output = model(inputs)

# Mesh vertices and selected node_data fields use the same GRID contract.
mesh_inputs = mesh_to_voxel_grid(
    mesh, resolution=(32, 64, 64), feature_names=("temperature",)
)
```

The first channel is occupancy by default. Feature values landing in the same
voxel are averaged. This utility voxelizes points or mesh vertices; it does not
rasterize watertight interiors or mesh faces. Supply a precomputed domain mask
when filled geometry is required.

## Training, visualization, and validation

The synthetic example uses a sphere mask plus XYZ coordinate channels to
predict the continuous field `x^2 + y^2 + z^2` inside the sphere:

```bash
python -m examples.train_unet3d_synthetic --epochs 10 --resolution 24
python -m examples.train_unet3d_synthetic --visualize
```

It uses the same `ModelInput`, `ModelOutput`, optimizer, and state-dict
checkpoint pattern as the other examples. Central or selected orthogonal slices
can also be plotted directly:

```python
from volumetric import plot_volume_slices

plot_volume_slices(
    input_volume,
    ground_truth=target,
    prediction=prediction,
    slice_indices=(15, 23, 17),  # z, y, x; omit for central slices
)
```

Run the focused tests with `pytest -q tests/test_unet3d.py` or the full suite
with `pytest -q`.

## Resolution and limitations

Dense 3D convolution is memory intensive. Doubling each axis from `64^3` to
`128^3` creates approximately eight times as many voxels, before accounting for
intermediate feature maps and gradients. Start with low resolution and small
channel counts.

3D U-Net provides strong local spatial modeling on dense structured grids, but
voxelizing complex geometries can consume substantial memory and lose geometric
precision relative to point- or mesh-based networks. PointNet/PointNet++ remain
appropriate for point clouds, while GCN/GAT/MeshGraphNet preserve graph or mesh
connectivity. This module intentionally contains no PDE, PINN, or task-specific
loss so it can be reused for supervised, surrogate, and physics-guided work.
