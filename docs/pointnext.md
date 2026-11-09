# PointNeXt

PointNeXt (Qian et al., NeurIPS 2022) revisits PointNet++. It keeps set abstraction (farthest
point sampling, ball query, shared MLP, max pooling) and adds two things: a modern training
recipe, and inverted residual MLP blocks (InvResMLP) that make the network deeper without losing
accuracy. The paper shows the recipe alone lifts PointNet++ from 77.9% to 86.1% on ScanObjectNN.

## Input/output contract

| Field | Shape | Notes |
| --- | --- | --- |
| `coordinates` | `[B, N, 3]` | Required xyz. |
| `point_features` | `[B, N, input_dim - 3]` | Optional extra channels. |
| `point_mask` | `[B, N]` bool | Optional, for padded batches. |
| `predictions` (`task="classification"`) | `[B, output_dim]` | Pooled global feature, MLP head with dropout. |
| `predictions` (`task="segmentation"`) | `[B, N, output_dim]` | Feature-propagation decoder back to every point. |

```python
from models import ModelInput, ModelRegistry

model = ModelRegistry.create("pointnext", {"task": "classification", "output_dim": 40})
logits = model(ModelInput(coordinates=points)).predictions  # [B, 40]
```

## Configuration

`width` sets the first stage's channels; each later stage doubles them. `blocks` has one entry per
stage (`len(sample_counts) + 1`) and counts the stage's stem / downsampling layer, so
`blocks=(1, 1, 1, 1, 1)` (the default) means no InvResMLP and is PointNeXt-S. Raise the entries to
build the larger variants. `radius` grows by `radius_scaling` per stage, and `expansion` is the
InvResMLP bottleneck ratio.

## Cost and limits

Linear in the number of points after sampling: `sample_counts` fixes the centroids per stage, and
each ball query takes `neighbor_count` neighbours. Permutation invariant (checked numerically); not
translation or rotation invariant.

## Paper results and recipe

ScanObjectNN 87.7% OA; ModelNet40 93.2% OA (PointNeXt-S); ShapeNetPart 86.7% instance mIoU
(PointNeXt-S); S3DIS 70.5% mIoU on Area 5 and 74.9% with 6-fold cross-validation (PointNeXt-XL).
ScanObjectNN recipe: AdamW, learning rate 0.002, weight decay 0.05, cosine decay, 250 epochs,
batch 32, label smoothing 0.3; augmentation by rotation, scaling, resampling and height appending.
ModelNet40 uses the same recipe for 600 epochs.

## References

- [PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies (arXiv 2206.04670)](https://arxiv.org/abs/2206.04670)
- [Official implementation (OpenPoints)](https://github.com/guochengqian/PointNeXt)
