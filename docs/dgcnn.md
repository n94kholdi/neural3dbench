# DGCNN

Dynamic Graph CNN (Wang et al., ACM TOG 2019) for point cloud classification and per-point
segmentation. Each EdgeConv layer rebuilds a k-nearest-neighbour graph in the current feature
space and applies a shared MLP to `(x_i, x_j - x_i)` for every edge, followed by a max over the
neighbours. Because the graph is recomputed per layer, points that are far apart in xyz but similar
in feature space can exchange information.

## Input/output contract

| Field | Shape | Notes |
| --- | --- | --- |
| `coordinates` | `[B, N, 3]` | Required xyz. |
| `point_features` | `[B, N, input_dim - 3]` | Optional extra channels (normals, colors). |
| `point_mask` | `[B, N]` bool | Optional; padded points are excluded from k-NN and pooling. |
| `predictions` (`task="classification"`) | `[B, output_dim]` | Max + mean pooled global feature, MLP head with dropout. |
| `predictions` (`task="segmentation"`) | `[B, N, output_dim]` | Per-point logits or regression values. |

```python
from models import ModelInput, ModelRegistry

model = ModelRegistry.create("dgcnn", {"task": "segmentation", "output_dim": 4})
logits = model(ModelInput(coordinates=points)).predictions  # [B, N, 4]
```

## Configuration

| Key | Default | Paper |
| --- | --- | --- |
| `neighbor_count` | 20 | k = 20 (k = 40 with 2048 points) |
| `edge_channels` | `(64, 64, 128, 256)` | same |
| `embedding_dim` | 1024 | same |
| `dropout` | 0.5 | keep probability 0.5 in the last two FC layers |

## Cost and limits

The k-NN search builds an `N x N` distance matrix in every EdgeConv layer: time and memory grow
as `O(L * N^2)`. Above a few thousand points per sample, sub-sample or switch to PointNet++ /
PointNeXt (ball query on sampled centroids). The model is permutation invariant (checked
numerically) but not translation or rotation invariant; train with augmentation when poses vary.

## Paper results and recipe

ModelNet40 92.9% overall accuracy (1024 points), ShapeNetPart 85.2% mIoU, S3DIS 56.1% mIoU.
Training: SGD, learning rate 0.1, momentum 0.9, cosine annealing to 0.001, batch 32; augmentation
by random scaling and point perturbation. The advisor's `recommend_training_setup` uses this recipe.

## References

- [Dynamic Graph CNN for Learning on Point Clouds (arXiv 1801.07829)](https://arxiv.org/abs/1801.07829)
- [Official implementation](https://github.com/WangYueFt/dgcnn)
