# Fourier Neural Operator (FNO3D)

FNO3D learns mappings between dense physical fields on regular 3D grids. A
typical mapping is a coefficient, source, material, or boundary field to a
pressure, temperature, velocity, or displacement field. It is an operator
model: training examples should contain varied input functions and their
corresponding output functions.

This implementation follows the Fourier Neural Operator paper and the authors'
original 3D implementation:

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- [Original FNO repository](https://github.com/zongyi-li/fourier_neural_operator)
- [NeuralOperator, the maintained official library](https://github.com/neuraloperator/neuraloperator)

## Repository mapping

| FNO component | Neural3DBench implementation |
| --- | --- |
| lifting map `P` | `models/fno/model.py` (`Conv3d`, kernel 1) |
| 3D spectral convolution | `models/fno/spectral.py` |
| Fourier operator block | `models/fno/components.py` |
| projection map `Q` | `models/fno/components.py` |
| model, config, input validation | `models/fno/model.py` |
| registry entry and public exports | `models/fno/__init__.py` |

## Input and output contract

The public layout is channel-first:

```text
ModelInput.voxel_fields: [B, C_in,  D, H, W]
ModelOutput.predictions:  [B, C_out, D, H, W]
representation:           "GRID"
```

`input_channels` counts only user-supplied physical fields. When
`append_coordinates=True`, FNO3D constructs three non-trainable channels in
`[0, 1]` and appends them internally in `D, H, W` order. The grid uses the
input device and dtype.

Material properties, parameters, and boundary-condition indicators that the
operator should learn from must be encoded as dense input channels. Dataset
normalization, denormalization, PDE residuals, and hard boundary enforcement
remain outside the model.

## Architecture and Fourier tensor flow

The default path is:

```text
physical fields (+ D/H/W coordinates)
  -> pointwise lifting P
  -> [spectral K(x) + pointwise W(x), activation] repeated
  -> final [K(x) + W(x)] without activation
  -> pointwise projection Q
```

For a channel-first latent tensor, one spectral layer computes:

```text
x:      [B, C_in,  D, H, W]
x_ft:   [B, C_in,  D, H, W//2 + 1]  complex
weight: [C_in, C_out, mD, mH, mW]    complex, four tensors
out_ft: [B, C_out, D, H, W//2 + 1]   complex
output: [B, C_out, D, H, W]
```

The channel contraction is
`einsum("bidhw,iodhw->bodhw", x_ft_region, weight)`. `rfftn` and `irfftn`
use PyTorch's default FFT normalization. The inverse receives the original
`(D, H, W)` explicitly, which preserves odd sizes.

The `D` and `H` axes are full complex FFT axes, while `W` is the one-sided
real-FFT axis. Four non-overlapping blocks are populated:

```text
D[:mD],   H[:mH],   W[:mW]  # positive/positive
D[-mD:],  H[:mH],   W[:mW]  # negative/positive
D[:mD],   H[-mH:],  W[:mW]  # positive/negative
D[-mD:],  H[-mH:],  W[:mW]  # negative/negative
```

All other coefficients in `out_ft` are zero. Therefore `mD <= D//2`,
`mH <= H//2`, and `mW <= W//2+1`. Invalid mode counts raise an error rather
than being clipped. Every block has its own learned complex parameter tensor.

## Configuration and factory use

```yaml
model:
  name: fno3d
  input_channels: 4
  output_channels: 1
  width: 32
  num_layers: 4
  modes: [12, 12, 12]
  projection_channels: 128
  activation: gelu
  append_coordinates: true
  normalization: none
  dropout: 0.0
  padding: [0, 0, 0]
```

```python
import torch
from models import ModelInput, create_model

model = create_model({
    "name": "fno3d",
    "input_channels": 4,
    "output_channels": 1,
    "width": 32,
    "num_layers": 4,
    "modes": [12, 12, 12],
})
x = torch.randn(2, 4, 32, 32, 32)
y = model(ModelInput(voxel_fields=x, representation="GRID")).predictions
```

The supported block normalization values are `none`, `batch`, and `instance`.
Dropout is applied only between the two pointwise projection layers.

## Padding and boundaries

FFT convolution represents periodic continuation. That assumption is most
natural for periodic problems; coordinate and boundary fields or the local
branch can help learn non-periodic behavior, but do not enforce a boundary
condition.

`padding` is either one integer or `[pad_D, pad_H, pad_W]`. FNO3D applies
constant-zero, one-sided padding to the upper end of each spatial axis after
lifting, using PyTorch order `(0, pad_W, 0, pad_H, 0, pad_D)`. After the final
operator block it crops `[..., :D, :H, :W]`. Zero padding exactly disables
this path. Mode validity is checked against the original input grid.

## Synthetic training, checkpoints, and visualization

The example learns a family of deterministic Helmholtz-like spectral smoothing
maps. Each sample contains a different source function and a dense scalar
coefficient channel, so this is operator learning rather than reconstruction
of one fixed volume.

```bash
python examples/train_fno3d_synthetic.py --epochs 5 --resolution 16
python examples/train_fno3d_synthetic.py --device cuda --visualize
```

The script saves `model_state_dict` and serialized `config`, reloads both, and
checks the restored prediction. Visualization uses
`volumetric.plot_volume_slices` and requires the `viz` optional dependency.

## Geocore voxel-grid integration

The sibling geocore project can produce the canonical interchange file from
filled geometry rather than merely marking mesh vertices. Its external
`VOXEL_GRID` representation is converted by `GridRepresentationAdapter` to
the internal `GRID` name expected by FNO3D; `voxel_fields` remains in
channel-first `[B,C,D,H,W]` order with `D,H,W = Z,Y,X`.

The smoke integration creates a rounded box, saves and reloads
`voxel_grid.npz`, validates four channels (`occupancy`, `sdf`, `boundary`, and
`source`), runs CPU forward/backward passes, and verifies a checkpoint round
trip:

```bash
.venv/bin/python neural3dbench/examples/geocore_fno3d_integration.py \
  --output-dir outputs/fno3d_geocore_smoke
```

The training example varies spheres, rounded boxes, and heat-source locations
on a fixed physical domain. It trains FNO3D against a small masked diffusion
reference operator and saves the canonical interchange input, checkpoint, and
validation tensors. A companion command renders middle-plane previews:

```bash
.venv/bin/python neural3dbench/examples/train_fno3d_geocore_heat.py \
  --device cpu --output-dir outputs/fno3d_geocore_heat
.venv/bin/python neural3dbench/examples/preview_fno3d_geocore_heat.py \
  --output-dir outputs/fno3d_geocore_heat
```

The saved checkpoint records channel order and `normalization_id`; both must
match at inference. The preview script requires the `viz` optional dependency.

## Scalability benchmark

Safe CPU defaults benchmark resolutions 16 and 32:

```bash
python examples/benchmark_fno3d.py
python examples/benchmark_fno3d.py --device cuda --resolutions 16 32 64
```

Output is CSV with resolution, parameter count, forward latency,
forward-and-backward latency, and peak allocated CUDA memory. CUDA is
synchronized around timing and peak statistics are reset after warm-up. Cubic
grids grow quickly: physical and Fourier activations scale with `D*H*W`, and
the retained complex weights scale with `C_in*C_out*mD*mH*mW` per block.

## Resolution generalization and limitations

The same learned low-frequency weights can be evaluated on another regular
resolution when every configured mode still fits. This architectural property
does not guarantee resolution-independent accuracy: the training distribution,
sampling, normalization, aliasing, coordinate convention, and whether the new
grid resolves the relevant physics all matter.

Standard FNO3D requires a dense rectangular grid. It does not directly process
arbitrary point clouds, unstructured meshes, changing topology, or sparse node
sets, and it does not silently voxelize them. Use the repository's graph,
MeshGraphNet, PointNet, or Transolver families for those representations.
Geo-FNO, NUFFT, graph neural operators, and mesh-aware Fourier variants are
separate future work.

## Reference-fidelity differences

- The original `fourier_3d.py` has channel-last experiment I/O and hard-coded
  channels, width, four blocks, projection size, and last-axis padding. This
  implementation exposes those choices and uses Neural3DBench's channel-first
  contract. It is not state-dict compatible with the original experiment.
- The original code creates coordinates by NumPy-backed tensors. Here they are
  created directly on the input device and dtype in deterministic `D,H,W`
  order. Numerical values remain linearly spaced in `[0,1]`.
- The original declares batch-normalization modules but its shown forward path
  does not use them. `normalization="none"` is therefore the default; enabled
  normalization is applied after the spectral/local sum and before activation.
- Padding defaults to zero and, when enabled, can cover all three physical
  axes. This is more general than the original experiment's hard-coded
  one-sided final-axis pad and changes boundary numerics when enabled.
- Complex initialization and scale, FFT normalization, four-corner indexing,
  spectral-plus-local ordering, GELU default, and omission of activation after
  the final operator block follow the authors' modern 3D code. The projection
  uses `1x1x1` convolutions, which are pointwise-equivalent to its linear layers.
