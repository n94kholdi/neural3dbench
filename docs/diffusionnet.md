# DiffusionNet

Surface network built from learned heat diffusion (Sharp et al., ACM TOG 2022). It is robust to
how the surface is triangulated, so weights trained on one mesh resolution work on another.

Each of `num_blocks` blocks of width `width` does three things:

1. **Learned diffusion.** `x_t = Phi exp(-lambda t) Phi^T M x`, with one learned time `t` per
   channel. `Phi` and `lambda` are the lowest `k_eig` eigenpairs of the cotangent Laplacian, and
   `M` is the lumped vertex mass.
2. **Spatial gradient features.** Gradients of the diffused features in each vertex's tangent
   plane, then `tanh(Re(conj(g) . A g))` with a learned complex channel mixing `A`. This gives
   orientation-aware filters that do not depend on the arbitrary tangent frame.
3. **Per-vertex MLP** on `[x, x_t, gradient features]`, added back to `x` (residual).

For classification, the per-vertex outputs are averaged with mass weights (area-weighted mean)
over each mesh.

## Operators

`models.diffusionnet.compute_operators(vertices, faces, k_eig)` precomputes the operators once
per mesh on the CPU with SciPy (`pip install scipy`). It returns:

- the lumped mass;
- the Laplacian eigenbasis (shift-invert `eigsh`, or a dense solve for small meshes);
- least-squares tangent gradient operators, stored as COO arrays.

Meshes with fewer than `k_eig` vertices are zero-padded. Disconnected parts give several zero
eigenvalues; this is expected.

`batch_operators(list_of_ops)` concatenates them in `collate_meshes` order. Pass the result as
`ModelInput.metadata["diffusion_operators"]`. If it is missing, the model computes the operators
in the forward pass. That is correct but slow, so precompute for training.

## Input/output contract

| Field | Shape | Notes |
|---|---|---|
| `coordinates` | `[V, 3]` float | Vertices of all meshes, concatenated (`collate_meshes`). |
| `faces` | `[F, 3]` int | Triangles. |
| `batch` | `[V]` int | Sorted mesh id per vertex; optional for one mesh. |
| `node_features` | `[V, input_dim]` | Optional input channels; default is xyz (`input_dim=3`). |
| `metadata["diffusion_operators"]` | dict | `mass [V]`, `evals [B, K]`, `evecs [V, K]`, sparse `grad_x`, `grad_y [V, V]`. |

| Output | Shape | Notes |
|---|---|---|
| `predictions` | `[B, output_dim]` | `task="classification"`: mass-weighted mean logits. |
| `predictions` | `[V, output_dim]` | `task="segmentation"`: per-vertex logits. |
| `latent` | `[V, output_dim]` | Per-vertex outputs before pooling (classification). |

## Usage

```python
from models import collate_meshes, create_model
from models.diffusionnet import batch_operators, compute_operators

model = create_model({"name": "diffusionnet", "output_dim": 10})
inputs = collate_meshes(meshes)  # [(vertices, faces), ...]
inputs.metadata["diffusion_operators"] = batch_operators([compute_operators(v, f, 128) for v, f in meshes])
logits = model(inputs).predictions  # [B, 10]
```

```bash
python examples/train_mesh_classifier.py --model diffusionnet --epochs 30   # operators cached on first run
```

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `input_dim` | 3 | Input channels (xyz). |
| `output_dim` | 10 | Classes or per-vertex outputs. |
| `width` | 128 | Channels per block. |
| `num_blocks` | 4 | DiffusionNet blocks. |
| `k_eig` | 128 | Laplacian eigenvectors for spectral diffusion. |
| `dropout` | 0.5 | Dropout in the block MLPs. |
| `task` | classification | `classification` or `segmentation`. |
| `with_gradient_features` | true | Add tangent gradient features. |
| `with_gradient_rotations` | true | Complex (rotating) `A` in the gradient features. |

## Limits

- Uses the plain cotangent Laplacian, not the robust (intrinsic Delaunay) Laplacian of the
  reference code. Cotangents of degenerate triangles are clipped at `1e3`.
- xyz inputs are not rotation invariant. The paper uses heat-kernel signatures (HKS) for
  invariance; pass them as `node_features`.
- Only spectral diffusion is implemented (no implicit time stepping), which suits meshes up to a
  few tens of thousands of vertices.

## References

- N. Sharp, S. Attaiki, K. Crane, M. Ovsjanikov. *DiffusionNet: Discretization Agnostic Learning on Surfaces.* ACM TOG 2022. https://arxiv.org/abs/2012.00888
- Reference code: https://github.com/nmwsharp/diffusion-net
