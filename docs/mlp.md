# MLP

`MLP` is a pointwise multilayer perceptron for continuous coordinate fields.
It is the classical PINN baseline: every point is evaluated independently, so
coordinate derivatives of any order are available through autograd. It shares
SIREN's input contract and can replace SIREN in coordinate-based training
loops.

## Repository mapping

| Component | Implementation |
| --- | --- |
| model, config, and input validation | `models/mlp/model.py` |
| registry entry and public exports | `models/mlp/__init__.py` |
| focused verification | `tests/test_mlp.py` |

## Input and output contract

```text
ModelInput.coordinates:    [B, N, spatial_dim]       required
ModelInput.point_features: [B, N, input_dim]         required if input_dim > 0
ModelInput.point_mask:     [B, N] boolean            optional
ModelOutput.predictions:   [B, N, output_dim]
ModelOutput.latent:        [B, N, widths[-1]]
representation:            "POINT_CLOUD"
```

Coordinates and features are concatenated before the first hidden layer.
Masked predictions and latents are zero, and the mask is returned as
`output.metadata["point_mask"]`.

## Configuration

```python
from models import MLPConfig, ModelInput, create_model

model = create_model(MLPConfig(
    spatial_dim=3,
    input_dim=0,
    output_dim=1,
    widths=(64, 64, 64),
    activation="tanh",   # tanh, silu, relu, or gelu
    residual=True,
))
prediction = model(ModelInput(coordinates=coordinates)).predictions
```

`widths` lists the hidden layer widths. With `residual=True`, a hidden layer
adds its input to its output only when both have the same width; the first
layer never does when its input size differs. Each hidden layer applies
`activation(linear(x))`, followed by a linear output layer.

`relu` has zero second derivatives almost everywhere, so prefer `tanh`, `silu`,
or `gelu` for PDEs with second- or higher-order terms.

## References

- [Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear PDEs (arXiv 1711.10561)](https://arxiv.org/abs/1711.10561)
