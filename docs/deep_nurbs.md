# Deep NURBS

Deep NURBS is a physics-informed PDE trial-function construction, not a generic
network for reconstructing CAD from point clouds. The reference is H. Saidaoui,
L. Espath, and R. Tempone, [“Deep NURBS—admissible physics-informed neural
networks”](https://doi.org/10.1007/s00366-024-02040-9), *Engineering with
Computers* 40 (2024), 4007–4021. The authors' [arXiv preprint
(2210.13900)](https://arxiv.org/abs/2210.13900) gives the method in full. The
paper says its code is available from the corresponding author by request; it
does not identify a public reference implementation.

## Reference method

The paper solves PDEs on complex 2D domains using an admissible NURBS field
multiplied by a coordinate MLP. For a scalar field, the trial solution is

```text
u(x; theta) = phi(x) * NN(x; theta) + zeta(x)
```

Here, `phi` is a NURBS field constructed to vanish on the Dirichlet boundary,
and `zeta` is a NURBS lifting field for nonzero Dirichlet data. The NURBS
surface map sends parametric coordinates to physical coordinates; the MLP is
evaluated on those physical coordinates. The paper uses a shallow, fully
connected MLP and the `max(0, x^3)` activation, and forms the PDE objective
using a weak-form energy. Uniform sampling in parametric space induces a
geometry-dependent physical-space sampling distribution.

This package implements the differentiable trial-function model and NURBS
surface evaluator. It intentionally leaves PDE definitions, energy/residual
losses, collocation sampling, and optimization to the caller. The documented
paper examples use 2D scalar elliptic PDEs; this implementation supports a
configurable output width but does not claim to reproduce the paper's training
experiments or TensorFlow code.

## Input and output contract

```text
ModelInput.coordinates: [B, N, 2]    required parametric coordinates [u, v]
ModelOutput.predictions: [B, N, output_dim]  phi * NN(physical_xy) + zeta
ModelOutput.latent:      [B, N, output_dim]  raw NN output before the ansatz
```

Parametric coordinates must be finite floating-point values in `[0, 1]` on
both axes. The active domain of each knot vector is exactly `[0, 1]`. The
surface's physical control points use `[Nu, Nv, 2]` layout with the first
control-net axis corresponding to `u`, and the second to `v`. `weights` has
shape `[Nu, Nv]`; `admissible_coefficients` has shape `[Nu, Nv]`; and
`lifting_coefficients` has shape `[Nu, Nv, output_dim]`. Surface coordinates
and both fields use the same tensor-product rational basis.

Clamped, nondecreasing knot vectors are required, each with length
`control_count + degree + 1`; knot multiplicity cannot exceed `degree + 1`.
The endpoints are repeated `degree + 1` times. All NURBS weights must be strictly
positive, which keeps the rational denominator positive. The admissible
coefficient net is required to be zero on its four outer edges, so its trace is
zero on the clamped surface boundary. The default geometry is a unit square,
with a degree-two basis and a nonzero interior admissible coefficient. The
caller remains responsible for choosing a non-folding, nonsingular surface
map when the PDE requires a bijective physical-domain parameterization.

`output.metadata` includes the original `parametric_coordinates`, evaluated
`physical_coordinates`, `admissible_field`, and `lifting_field`. The evaluator
is implemented in PyTorch operations and supports autograd with respect to
parametric coordinates and neural parameters. Geometry and NURBS coefficient
data are fixed configuration buffers, matching the paper's choice not to
optimize its control points. This implementation's 2D scope, configurable
hidden width/depth, PyTorch initialization, and `tanh` option are explicit
engineering choices; the paper's examples use one or two hidden layers, 50
neurons, cubic-ReLU activation, and TensorFlow defaults.

## Configuration and factory

```python
import torch

from models import DeepNURBSConfig, ModelInput, create_model

model = create_model(
    DeepNURBSConfig(hidden_dim=50, num_hidden_layers=1)
)
parametric_coordinates = torch.rand(2, 128, 2)
output = model(ModelInput(
    coordinates=parametric_coordinates,
    representation="PARAMETRIC_SURFACE",
))
print(output.predictions.shape)  # [2, 128, 1]
print(output.metadata["physical_coordinates"].shape)  # [2, 128, 2]
```

The same config can be supplied as a dictionary with `name="deep_nurbs"` to
`create_model`. For a custom geometry, provide knot vectors, a control-point
net, positive rational weights, and coefficient nets with the shapes above.
For nonhomogeneous Dirichlet data, choose `lifting_coefficients` to encode the
boundary trace; the model does not infer those coefficients or verify that
they match external boundary data.

The model guarantees the zero trace of the `phi * NN` term. It does not impose
PDE constraints, construct an admissible `phi` for a supplied boundary-value
problem, validate geometric bijectivity, or perform the paper's importance
sampling. Those tasks belong to problem setup and training code.

## Geocore integration and CPU training

A canonical geocore `PARAMETRIC_SURFACE` file contains both kinds of data the
model needs: fixed NURBS geometry for its configuration and sampled UV points
for runtime input. Load and split them explicitly:

```python
raw = gc.load("parametric_surface.npz")
config = DeepNURBSConfig.from_parametric_surface(raw, hidden_dim=48)
model_input = ParametricSurfaceRepresentationAdapter().adapt(raw)
model = create_model(config)
prediction = model(model_input).predictions
```

`examples/geocore/geocore_deep_nurbs_integration.py` creates a curved quarter-annulus,
validates/saves/reloads `parametric_surface.npz`, checks physical coordinates,
CPU gradients, and checkpoint restoration. The focused training example fits a
manufactured zero-Dirichlet field and saves metrics and visualization inputs:

```bash
python examples/geocore/train_deep_nurbs_geocore_field.py --device cpu
python examples/geocore/preview_deep_nurbs_geocore_field.py
pytest tests/test_deep_nurbs.py tests/test_geocore_deep_nurbs_integration.py
```

The manufactured field is a development check for the representation/model
boundary, not a comparable PDE benchmark or reproduction of the paper.
