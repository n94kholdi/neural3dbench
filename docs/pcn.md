# PCN

PCN (Point Completion Network) completes a partial point cloud into a dense, complete one.
It follows Yuan et al., *PCN: Point Completion Network* (3DV 2018). Two stacked PointNets
encode the partial cloud into one global feature. An MLP decodes that feature into coarse
points. A folding step then grows a small 2D patch around each coarse point.

## Input contract

```python
inputs = ModelInput(
    coordinates=partial,         # [B, N, 3], required XYZ of the partial scan
    point_features=features,     # [B, N, input_dim - 3], optional
    point_mask=valid_points,     # [B, N] bool, optional for padded batches
    representation="POINT_CLOUD",
)
```

N may differ between batches. Padded points are left out of both max-pools through
`point_mask`, so they do not change the output.

## Output contract

```python
output = model(inputs)
output.predictions          # [B, M, 3] dense completion, M = num_coarse * grid_size**2
output.auxiliary["coarse"]  # [B, num_coarse, 3] coarse completion
output.latent               # [B, latent_dim] global feature
```

M does not depend on N, and the output is a new point set. It is not a value per input
point. Train with a Chamfer or EMD loss against the complete shape. The paper also
applies the loss to `auxiliary["coarse"]`. The dashboard trains on the dense output only.

## Configuration

| key | default | meaning |
| --- | --- | --- |
| `input_dim` | 3 | XYZ plus optional point features |
| `latent_dim` | 1024 | global feature width |
| `hidden_dim` | 512 | folding MLP width |
| `num_coarse` | 256 | coarse points (the paper uses 1024) |
| `grid_size` | 2 | patch side; each coarse point becomes `grid_size**2` points (paper: 4) |
| `grid_scale` | 0.05 | half-width of the 2D patch |

```python
from models import ModelInput, create_model

model = create_model({"name": "pcn", "num_coarse": 1024, "grid_size": 4})  # paper setting, 16384 points
dense = model(ModelInput(coordinates=partial)).predictions
```

The layers have no batch normalisation, as in the paper.

## Limits

- The decoder sees only the global feature, so thin parts and sharp edges blur.
- The output size is fixed by the config.

## References

- [PCN: Point Completion Network](https://arxiv.org/abs/1808.00671), Yuan et al., 3DV 2018.
- [Reference implementation](https://github.com/wentaoyuan/pcn)
