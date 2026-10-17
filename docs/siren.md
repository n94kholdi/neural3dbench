# SIREN

SIREN (Sinusoidal Representation Networks) is a coordinate-based implicit
neural representation. It maps each continuous coordinate, optionally
concatenated with pointwise conditioning values, directly to a physical-field
value. Its periodic activations and specialized initialization are particularly
useful for high-frequency signals and coordinate derivatives.

This implementation follows the [SIREN paper](https://arxiv.org/abs/2006.09661),
the [official project page](https://www.vincentsitzmann.com/siren/), and the
[official implementation](https://github.com/vsitzmann/siren). SIREN is not a
neural operator and does not exchange information among points. A canonical
model's weights encode one signal. Conditioning features extend that mapping,
but do not turn the default architecture into a geometry-to-geometry operator.

## Repository mapping

| Component | Implementation |
| --- | --- |
| sine activation and initialization | `models/siren/layers.py` |
| model, config, and input validation | `models/siren/model.py` |
| registry entry and public exports | `models/siren/__init__.py` |
| focused verification | `tests/test_siren.py` |
| continuous 3D field example | `examples/train_siren_synthetic.py` |

## Input and output contract

```text
ModelInput.coordinates:    [B, N, spatial_dim]       required
ModelInput.point_features: [B, N, input_dim]         required if input_dim > 0
ModelInput.point_mask:     [B, N] boolean            optional
ModelOutput.predictions:   [B, N, output_dim]
representation:            "POINT_CLOUD"
```

`POINT_CLOUD` describes the irregular-point data container only. SIREN applies
the same MLP independently at every point and performs no neighborhood query,
pooling, attention, or other point-cloud aggregation. Coordinates and features
are kept separate at the public boundary and concatenated immediately before
the first periodic layer. Consequently, scaling of every concatenated feature
matters.

A true mask entry denotes a valid point. Masked predictions are zero, and the
device-local mask is returned as `output.metadata["point_mask"]`. Use
`pointcloud.collate_point_clouds` for padded 3D batches and
`pointcloud.pointwise_regression_loss` with that mask so padding does not enter
the loss. Every sample must contain at least one valid point.

## Architecture and initialization

A sine layer with input `x`, weights `W`, bias `b`, and frequency `omega_0`
computes

```text
sin(omega_0 * (W x + b)).
```

The tensor flow is

```text
[coordinates, optional point features]
  -> first sine layer (first_omega_0)
  -> num_hidden_layers hidden sine layers (hidden_omega_0)
  -> linear output layer
  -> [B, N, output_dim]
```

`num_hidden_layers` means exactly the number of sine layers after the first
sine layer and before the output layer. `outermost_linear=True` is the
paper-standard default. Setting it to false explicitly applies
`sin(hidden_omega_0 * output)` after the final linear map.

Initialization is part of SIREN, not an interchangeable implementation detail.
For a linear layer with `fan_in` inputs, weights are sampled as follows:

```text
first layer:      Uniform(-1/fan_in, 1/fan_in)
later sine layer: Uniform(-sqrt(6/fan_in)/omega_0,
                          sqrt(6/fan_in)/omega_0)
final linear:     Uniform(-sqrt(6/fan_in)/hidden_omega_0,
                          sqrt(6/fan_in)/hidden_omega_0)
```

As in the official implementation, custom initialization changes weights only;
linear biases retain normal `torch.nn.Linear` initialization. Both omega values
default to `30.0`, but remain separate because the first layer controls the
input frequency scale while hidden-layer omega participates in stable signal
and gradient propagation.

## Configuration and factory use

```python
from models import ModelInput, SIRENConfig, create_model

config = SIRENConfig(
    spatial_dim=3,
    input_dim=0,
    output_dim=1,
    hidden_dim=256,
    num_hidden_layers=3,
    first_omega_0=30.0,
    hidden_omega_0=30.0,
    outermost_linear=True,
)
model = create_model(config)
prediction = model(ModelInput(coordinates=coordinates)).predictions
```

The equivalent serialized form is accepted by the common factory:

```python
model = create_model({
    "name": "siren",
    "spatial_dim": 3,
    "input_dim": 0,
    "output_dim": 1,
    "hidden_dim": 128,
    "num_hidden_layers": 3,
    "first_omega_0": 30.0,
    "hidden_omega_0": 30.0,
})
```

## Coordinates, frequency, and derivatives

Normalize coordinates in the data pipeline, commonly to `[-1, 1]`. SIREN does
not normalize coordinates, features, or targets internally. A coordinate
rescaling changes the effective represented frequencies, so `first_omega_0`
must be interpreted together with the coordinate domain. Physical field
normalization and denormalization also stay outside the model.

The forward path preserves the caller's coordinate autograd graph and does not
change `requires_grad`:

```python
coordinates = coordinates.requires_grad_(True)
prediction = model(ModelInput(coordinates=coordinates)).predictions
gradient = torch.autograd.grad(
    prediction.sum(), coordinates, create_graph=True
)[0]
laplacian_x = torch.autograd.grad(
    gradient[..., 0].sum(), coordinates, create_graph=True
)[0][..., 0]
```

PDE residuals, boundary losses, sample selection, and derivative combinations
belong in the training loop. A PINN is a physics-informed training formulation;
SIREN can be its differentiable function approximator, but SIREN alone is not a
PINN.

## Synthetic example and checkpoints

The CPU-friendly example fits one deterministic multi-frequency 3D scalar
field at random training coordinates and evaluates different held-out
coordinates:

```bash
python examples/train_siren_synthetic.py --device cpu
python examples/train_siren_synthetic.py --device cuda --steps 500
```

It prints initial and final training/evaluation MSE, then saves and reloads a
checkpoint containing `model_state_dict` and `config`. Loading uses
`create_model(checkpoint["config"])`, consistent with other examples.

## Comparison and limitations

- FNO3D maps dense grid functions to dense grid functions and shares
  information spectrally across a regular grid. SIREN maps individual
  continuous coordinates to values.
- Transolver exchanges information among irregular spatial points through
  learned physical slices. SIREN has no point-to-point interaction.
- PointNet and PointNet++ aggregate point sets for global or pointwise tasks.
  SIREN is invariant to point ordering only because every point is evaluated
  independently; it does not form a set representation.
- PINN describes physics-informed optimization. SIREN supplies a suitable
  differentiable field representation, while the training code supplies the
  governing equations and constraints.

The core model deliberately omits graph edges, voxelization, convolutions,
attention, Fourier features, normalization layers, residual connections,
dropout, latent modulation, and built-in PDE behavior. Unlike the official
meta-learning code, it uses ordinary `nn.Linear` modules and the repository's
`ModelInput`/`ModelOutput` interface; official state dictionaries are therefore
not directly compatible. The sine transformation, layer ordering, default
linear output, omega semantics, and weight bounds remain faithful to the
reference architecture.
