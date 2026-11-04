# FoldingNet

FoldingNet is a point-cloud autoencoder that decodes by folding a fixed 2D grid onto the
shape surface. It follows Yang et al., *FoldingNet: Point Cloud Auto-encoder via Deep Grid
Deformation* (CVPR 2018). An encoder pools the cloud into one codeword. Two MLP folding
steps then map each grid point, together with the codeword, to a 3D point.

## Input contract

```python
inputs = ModelInput(
    coordinates=points,          # [B, N, 3], required XYZ (partial or complete cloud)
    point_features=features,     # [B, N, input_dim - 3], optional
    point_mask=valid_points,     # [B, N] bool, optional for padded batches
    representation="POINT_CLOUD",
)
```

Padded points are left out of the max-pool through `point_mask`.

## Output contract

```python
output = model(inputs)
output.predictions              # [B, M, 3] folded point set, M = grid_size**2
output.auxiliary["first_fold"]  # [B, M, 3] output of the first folding step
output.latent                   # [B, latent_dim] codeword
```

M does not depend on N, and the output is a new point set. Train with a Chamfer loss
against the target cloud. For an autoencoder, the target is the input itself. For
completion, the target is the complete shape.

## Configuration

| key | default | meaning |
| --- | --- | --- |
| `input_dim` | 3 | XYZ plus optional point features |
| `latent_dim` | 512 | codeword width |
| `hidden_dim` | 512 | folding MLP width |
| `grid_size` | 32 | grid side; output has `grid_size**2` points (paper: 45) |
| `grid_scale` | 0.3 | grid spans `[-grid_scale, grid_scale]^2` |

```python
from models import ModelInput, create_model

model = create_model({"name": "foldingnet", "grid_size": 45})  # paper setting, 2025 points
points = model(ModelInput(coordinates=partial)).predictions
```

## Differences from the paper

The encoder is a PointNet-style shared MLP (3, 64, 128, `latent_dim`) with max pooling. The
paper uses local covariance features and k-NN graph max-pooling layers. This version is
simpler, but it captures less local detail.

## Limits

- One folded sheet struggles with complex topology, such as holes or disconnected parts.
- PCN's coarse-to-fine decoder is usually more accurate for completion.

## References

- [FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation](https://arxiv.org/abs/1712.07262), Yang et al., CVPR 2018.
