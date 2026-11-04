# VoteNet

VoteNet detects oriented 3D boxes directly in point clouds, typically indoor scans.
It follows Qi et al., *Deep Hough Voting for 3D Object Detection in Point Clouds* (ICCV 2019).
A PointNet++ backbone with four set-abstraction levels and two feature-propagation levels
produces seed points with features. Each seed votes for an object centre with an offset and a
feature residual. The votes are sampled and grouped into clusters, and a shared MLP turns each
cluster into a box proposal.

## Input contract

```python
inputs = ModelInput(
    coordinates=points,          # [B, N, 3], required XYZ in metres, z up
    point_features=features,     # [B, N, input_dim - 3], optional (e.g. height or RGB)
    point_mask=valid_points,     # [B, N] bool, optional for padded batches
    representation="POINT_CLOUD",
)
```

## Output contract

```python
output = model(inputs)
output.predictions                    # [B, P, 7] boxes (x, y, z, l, w, h, yaw), z = box centre
output.auxiliary["objectness"]        # [B, P] probability that the proposal is an object
output.auxiliary["class_scores"]      # [B, P, num_classes]
output.auxiliary["proposal_centers"]  # [B, P, 3] vote-cluster centres
output.auxiliary["seeds"]             # [B, S, 3] seed points (S = sample_counts[1])
output.auxiliary["votes"]             # [B, S, 3] voted centres
output.latent                         # [B, P, 128] proposal features
```

P = `num_proposals` (256 by default). The raw heads are also in `auxiliary`:
`objectness_logits`, `center_offset`, `heading_logits`, `heading_residual`, `size_logits`,
`size_residual` `[B, P, num_classes, 3]` and `class_logits`. To get detections, keep the
proposals whose objectness passes a threshold and run 3D NMS.

## Training

```python
losses = model.detection_loss(output, gt_boxes, gt_labels)
losses["loss"].backward()   # also returns "vote", "objectness", "box" and "semantic"
```

- `gt_boxes`: `[B, M, 7]` as `(x, y, z, l, w, h, yaw)`.
- `gt_labels`: `[B, M]` class ids. Use `-1` for padding slots.

The loss follows the paper:

- **Vote loss.** Each seed inside a box regresses its vote to that box's centre with an L1 loss.
- **Objectness.** A proposal is positive within 0.3 m of a box centre and negative beyond 0.6 m.
  Proposals in between are ignored. Cross-entropy uses class weights 0.2 / 0.8.
- **Box loss.** On positive proposals: a smooth-L1 centre loss, heading bin and residual, and
  size cluster and residual. Each class is its own size cluster.
- **Semantic loss.** Cross-entropy on the class of positive proposals.

The weights are vote 1, objectness 0.5, box 1 and semantic 0.1.

## Configuration

| key | default | meaning |
| --- | --- | --- |
| `input_dim` | 3 | XYZ plus optional point features |
| `mean_sizes` | `((1, 1, 1),)` | one `(l, w, h)` per class; set to your dataset's class means |
| `num_heading_bins` | 12 | heading classes over 2 pi |
| `sample_counts` | `(2048, 1024, 512, 256)` | points kept at each set-abstraction level |
| `radii` | `(0.2, 0.4, 0.8, 1.2)` | ball radii in metres |
| `neighbor_counts` | `(64, 32, 16, 16)` | points per ball |
| `num_proposals` | 256 | vote clusters |
| `proposal_radius` | 0.3 | vote grouping radius |

```python
from models import ModelInput, create_model

model = create_model({"name": "votenet", "mean_sizes": [(2.0, 1.0, 0.8), (0.6, 0.6, 0.9)]})
output = model(ModelInput(coordinates=scan))
```

## Differences from the paper

- One vote per seed. The paper's vote loss lets a seed inside several boxes match the closest
  of up to three votes.
- The centre loss is a smooth-L1 to the assigned box, not the paper's Chamfer loss between
  predicted and ground-truth centres.
- Post-processing (objectness threshold and 3D NMS) is left to the caller.

## Limits

- The ball radii are absolute, so coordinates must be in metres or rescaled to match.
- Farthest-point sampling is sequential and slow on large outdoor sweeps.
  Use PointPillars for LiDAR scenes.

## References

- [Deep Hough Voting for 3D Object Detection in Point Clouds](https://arxiv.org/abs/1904.09664), Qi et al., ICCV 2019.
- [Reference implementation](https://github.com/facebookresearch/votenet)
