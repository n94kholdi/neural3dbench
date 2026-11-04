# DCP (Deep Closest Point)

Rigid point-cloud registration (Wang & Solomon, 2019; DCP-v2). Given a source and a target cloud,
DCP predicts the rotation `R` and translation `t` with `R @ x + t ≈ y`.

1. **Embedding.** Both clouds pass through the same DGCNN EdgeConv stack (k-NN in feature space).
   The layer outputs are concatenated and projected to `embedding_dim`.
2. **Transformer (v2).** `phi_X = F_X + T(F_Y, F_X)` and `phi_Y = F_Y + T(F_X, F_Y)`, so each cloud's
   features depend on the other cloud. With `use_transformer=False` this step is skipped (DCP-v1).
3. **Soft matching.** `scores = softmax(phi_X @ phi_Y^T / sqrt(d))`; every source point gets a soft
   correspondence `scores @ Y` in the target.
4. **SVD head.** Weighted Procrustes between source points and their correspondences gives `R`
   (reflections removed, `det R = +1`) and `t`.

## Input/output contract

| Field | Shape | Notes |
|---|---|---|
| `coordinates` | `[B, N, 3]` | Source cloud (required). |
| `query_coordinates` | `[B, M, 3]` | Target cloud (required). `M` may differ from `N`. |
| `point_mask` | `[B, N]` bool | Optional source padding mask. |

| Output | Shape | Notes |
|---|---|---|
| `predictions` | `[B, 4, 4]` | Homogeneous transform source to target. |
| `auxiliary["rotation"]` | `[B, 3, 3]` | `R`, orthonormal, `det = +1`. |
| `auxiliary["translation"]` | `[B, 3]` | `t`. |
| `auxiliary["correspondences"]` | `[B, N, 3]` | Soft match of each source point in the target. |
| `auxiliary["scores"]` | `[B, N, M]` | Matching probabilities (rows sum to 1). |

## Usage

```python
from models import ModelInput, create_model

model = create_model({"name": "dcp"})
output = model(ModelInput(coordinates=source, query_coordinates=target))
losses = model.registration_loss(output, gt_rotation, gt_translation)  # [B, 3, 3], [B, 3]
losses["total"].backward()
```

`registration_loss` is the paper loss `||R^T R_gt - I||^2 + ||t - t_gt||^2`; add weight decay in the
optimizer. The paper trains on ModelNet40 with 1024 points, rotations up to 45 degrees per axis and
translations in `[-0.5, 0.5]`.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `neighbor_count` | 20 | k for EdgeConv k-NN. |
| `edge_channels` | `(64, 64, 128, 256)` | EdgeConv widths. |
| `embedding_dim` | 512 | Per-point embedding width. |
| `num_heads`, `ff_dim`, `num_layers` | 4, 1024, 1 | Transformer size. |
| `dropout` | 0.0 | Transformer dropout. |
| `use_transformer` | true | false gives DCP-v1. |

## Limits

- Assumes near-full overlap; no outlier rejection. Partial scans need PRNet/RPM-Net-style variants.
- Features are not rotation invariant; train on the rotation range you expect at test time.
- `N x M` score matrix and feature k-NN are quadratic; use about 1k-2k points per cloud.
- The dashboard has no paired-pose dataset, so it does not train DCP.

## References

- Y. Wang, J. Solomon. *Deep Closest Point: Learning Representations for Point Cloud Registration.* ICCV 2019. https://arxiv.org/abs/1905.03304
- Reference code: https://github.com/WangYueFt/dcp
