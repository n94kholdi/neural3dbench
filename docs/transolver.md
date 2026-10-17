# Transolver

Transolver is a neural PDE solver for point- or mesh-node samples on irregular
domains. Unlike graph models, it does not require edges, and unlike 3D U-Net it
does not voxelize the domain. Its Physics-Attention layers learn a small set of
latent physical slices and model global interactions between those slices.

This implementation follows the original ICML 2024 paper and the official
THUML irregular-mesh implementation:

- [paper](https://arxiv.org/abs/2402.02366)
- [official repository](https://github.com/thuml/Transolver)
- [official Physics-Attention](https://github.com/thuml/Transolver/blob/main/Physics_Attention.py)

## Input and output

The model consumes the common `ModelInput` interface:

```python
inputs = ModelInput(
    coordinates=coordinates,      # [B, N, spatial_dim]
    point_features=features,      # [B, N, input_dim], optional
    point_mask=valid_points,      # [B, N] boolean, optional
)
output = model(inputs)
prediction = output.predictions   # [B, N, output_dim]
```

Coordinates and physical/geometric features remain separate at the data
boundary. `InputEmbedding` concatenates them exactly where the architecture
embeds its input. No graph connectivity is consumed. Outputs are raw continuous
values; task-specific activations, data losses, PDE residuals, and normalization
belong in the training/data pipeline.

`point_mask=True` denotes a real point. Padding is removed from slice
aggregation, zeroed after every residual block, and zeroed in the final
prediction. Use the existing `pointcloud.collate_point_clouds` helper for padded
3D point-cloud batches. Two-dimensional data can construct the same
`ModelInput` directly.

## Physics-Attention tensor flow

Let `B` be batch size, `N` points, `C` hidden width, `H` heads, `D=C/H` channels
per head, and `G` learned slices:

1. Two learned projections map `[B,N,C]` to assignment and value features
   `[B,H,N,D]`.
2. A slice projection produces logits `[B,H,N,G]`. Softmax is taken over `G`,
   so every valid point's assignment weights sum to one.
3. Weighted spatial aggregation produces tokens `[B,H,G,D]`; each token is
   divided by its total assignment mass over `N`.
4. Per-head Q/K/V projections and scaled dot-product attention operate among
   the `G` physical tokens, producing attention `[B,H,G,G]`.
5. The same slice weights broadcast the updated tokens back to points,
   producing `[B,H,N,D]`, which is concatenated to `[B,N,C]` and projected.

The cost is `O(NGC + G²C)`, linear in point count when `G` is fixed and much
smaller than `N`. The learned slices need not be spatially local: distant points
with similar latent physical states may share a slice.

Set `return_attention_weights=True` to expose each block's `slice_weights`,
`slice_tokens`, token `attention_weights`, and `output_tokens` under
`output.auxiliary["physics_attention"]`. Leave it disabled for normal training
to avoid retaining large diagnostic tensors.

## Configuration

```yaml
model:
  name: transolver
  spatial_dim: 3
  input_dim: 8
  output_dim: 4
  hidden_dim: 256
  num_layers: 8
  num_heads: 8
  num_slices: 32
  mlp_ratio: 4
  dropout: 0.0
  activation: gelu
  return_attention_weights: false
```

`input_dim` counts only `point_features`; coordinates are counted separately by
`spatial_dim`. `hidden_dim` must be divisible by `num_heads`. The original
experiments generally used eight layers, width 128 or 256, and 32 or 64 slices.
Smaller settings are useful for tests and development.

## Training, checkpoints, and visualization

The synthetic example uses the same direct PyTorch training/checkpoint style as
the repository's other examples and reports MSE and relative L2:

```bash
python examples/train_transolver_synthetic.py --epochs 100
python examples/train_transolver_synthetic.py --output-dim 3 --visualize
```

It learns a smooth field on random 2D points. With multiple outputs it also
generates additional smooth fields. The visualization reuses
`pointcloud.plot_point_cloud`, including ground truth, prediction, and absolute
error; multi-component fields can be visualized by component or magnitude using
the existing utility.

Checkpoints contain `model_state_dict` and the serializable config, matching the
other model examples. Input/output/coordinate normalization is intentionally
external: compute statistics on training data, transform `ModelInput` and
targets before the forward pass, then denormalize predictions for evaluation.
`pointcloud.pointwise_regression_loss` provides MSE, MAE, and relative L2 and
excludes padding when passed the same `point_mask` used by the model.

Run tests and the hardware benchmark with:

```bash
pytest tests/test_transolver.py
python examples/benchmark_transolver.py
python examples/benchmark_transolver.py --point-counts 1024 4096 --device cpu
```

The benchmark measures median forward time, forward/backward step time, and
peak allocated CUDA memory when available. It generates results on the current
hardware rather than storing potentially misleading numbers. Retaining
attention diagnostics increases memory use. Runtime is linear in `N` for fixed
`G`, while token attention is quadratic in `G`.

## Reference-fidelity implementation note

| Original component | Repository implementation |
| --- | --- |
| Concatenate geometry and observed quantities; MLP preprocessing | `models/transolver/components.py::InputEmbedding` |
| Irregular-mesh slicing, token attention, and deslicing | `models/transolver/physics_attention.py::PhysicsAttention` |
| Pre-norm attention/FFN residual layer | `models/transolver/block.py::TransolverBlock` |
| Learned placeholder, stacked layers, field projection | `models/transolver/model.py::Transolver` |
| Architecture configuration and validation | `models/transolver/model.py::TransolverConfig` |

Intentional differences from the official task-specific code:

- Padding masks are supported. Invalid points receive zero slice weight and
  cannot affect tokens; this extends the original fixed-size batches without
  changing unmasked computation.
- The last `LayerNorm + Linear` projection is a separate `OutputHead` instead
  of being embedded in the final block. The computation is equivalent, but
  official checkpoint parameter names are not directly compatible.
- PyTorch's built-in `trunc_normal_` replaces the official `timm` helper, and
  tensor rearrangement uses reshape/permute instead of adding `einops`.
- Only the paper's irregular-mesh linear projections are included. Structured
  1D/2D/3D convolutional variants are deliberately out of scope because this
  repository integration targets arbitrary points and mesh nodes.
- Optional timestep embeddings and task-specific unified positional grids from
  individual official experiments are not part of the core paper default.
  Coordinates are concatenated directly, which is the official unstructured
  behavior. Time or global parameters can be supplied as repeated point
  features by the data pipeline.

These differences preserve the core Physics-Attention algorithm. No ordinary
point-to-point self-attention is substituted for it.
