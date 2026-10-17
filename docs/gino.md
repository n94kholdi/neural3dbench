# Geometry-Informed Neural Operator (GINO)

GINO maps fields on irregular 3D point sets through a regular latent grid. Its
architecture follows the Geometry-Informed Neural Operator described by Li et
al. (NeurIPS 2023): an input graph/kernel neural operator integrates point
features onto a latent grid, Fourier Neural Operator layers propagate global
information on that grid, and an output graph/kernel neural operator evaluates
the latent field at requested coordinates.

Primary reference: [Li et al., *Geometry-Informed Neural Operator for
Large-Scale 3D PDEs*, arXiv:2309.00583](https://arxiv.org/abs/2309.00583).
The implementation uses a compact-support learned displacement-conditioned
kernel and normalized weighted integration for both geometry transforms. This is a lightweight
PyTorch realization of the GNO-FNO-GNO design, not a reproduction of every
optional feature in the reference implementation.

## Input and output

Use `ModelInput` with the `POINT_CLOUD` representation:

- `coordinates`: source locations, `[B, N, 3]`.
- `point_features`: source field values, `[B, N, input_dim]`.
- `query_coordinates`: optional output locations, `[B, Q, 3]`.
- `point_mask`: optional boolean validity mask, `[B, N]`; false entries are
  excluded as sources. When query coordinates are omitted, the mask is also
  used as the query mask and padded predictions are zero.

Each batch item is integrated independently. Batch items may have different
valid point counts through padding and `point_mask`, but tensors in a batch
share padded `N` and, for explicit queries, padded `Q`. Coordinates must be
finite and normalized to `[0, 1]` on each axis; the latent grid spans that same
unit cube. Normalize physical coordinates before constructing `ModelInput`.
Coordinates outside the domain are rejected for valid sources and all explicit
queries. Point features must be floating-point and have the configured width.

Predictions have shape `[B, Q, output_dim]` at `query_coordinates`. If queries
are omitted, predictions are returned at source coordinates with shape
`[B, N, output_dim]`; invalid padded source rows are zero. The latent output is
`[B, width, D, H, W]`.

The compact-support kernel radius is measured in normalized coordinate units.
Pairwise distance and kernel computations are chunked over both query and
source points, with at most `kernel_max_pairs` coordinate pairs in the active
workspace. This bounds temporary memory, while computation still scales with
the source and target point counts.

## Configuration

`GINOConfig` exposes:

- `input_dim`, `output_dim`: source and prediction feature widths.
- `latent_resolution`: one integer or three `[D, H, W]` sizes.
- `width`, `modes`, `num_layers`: latent channels and Fourier operator setup.
- `kernel_radius`, `kernel_hidden_dim`: compact-support geometry kernel.
- `kernel_chunk_size`, `kernel_max_pairs`: chunk and pair-workspace limits.
- `activation`: `gelu`, `relu`, or `silu` between Fourier blocks.

Fourier modes must fit the selected latent resolution. Inputs must use the
point-cloud representation when a representation is specified. No additional
package dependency is required.

## Example

```python
import torch

from models import ModelInput, create_model

model = create_model(
    {
        "name": "gino",
        "input_dim": 1,
        "output_dim": 1,
        "latent_resolution": (8, 8, 8),
        "width": 16,
        "modes": (3, 3, 3),
        "num_layers": 2,
        "kernel_radius": 0.4,
    }
)
inputs = ModelInput(
    coordinates=torch.rand(2, 128, 3),
    point_features=torch.randn(2, 128, 1),
    query_coordinates=torch.rand(2, 64, 3),
    representation="POINT_CLOUD",
)
predictions = model(inputs).predictions  # [2, 64, 1]
```

The current implementation uses fixed cubic latent grids and compact-support
radial kernels; it does not model signed-distance fields or impose physical
boundary conditions. Kernel integration uses chunked pairwise distance
evaluation rather than a spatial index, so very large point sets remain
quadratic in arithmetic even though temporary pairwise memory is bounded.