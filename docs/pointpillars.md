# PointPillars

PointPillars detects oriented 3D boxes in LiDAR scenes. It follows Lang et al.,
*PointPillars: Fast Encoders for Object Detection from Point Clouds* (CVPR 2019).
Points are grouped into vertical pillars on an x/y grid. A small PointNet encodes each
pillar, and the pillar features are scattered into a bird's-eye-view (BEV) pseudo-image.
A 2D CNN with three strided blocks and upsampling processes that image. An SSD-style
head predicts class scores, box deltas and a heading direction for every anchor.

## Input contract

```python
inputs = ModelInput(
    coordinates=points,          # [B, N, 3], required XYZ in metres (x forward, z up)
    point_features=intensity,    # [B, N, input_dim - 3], e.g. [B, N, 1] LiDAR intensity
    point_mask=valid_points,     # [B, N] bool, optional for padded batches
    representation="POINT_CLOUD",
)
```

Points outside `point_cloud_range` and masked points are ignored. N may differ between batches.

## Output contract

```python
output = model(inputs)
output.predictions                    # [B, A, 7] decoded boxes (x, y, z, l, w, h, yaw), z = box centre
output.auxiliary["scores"]            # [B, A, num_classes] sigmoid class scores
output.auxiliary["class_logits"]      # [B, A, num_classes]
output.auxiliary["box_deltas"]        # [B, A, 7] anchor-relative deltas
output.auxiliary["direction_logits"]  # [B, A, 2] heading direction (yaw in [0, pi) or [pi, 2 pi))
output.auxiliary["anchors"]           # [A, 7]
output.latent                         # [B, sum(upsample_filters), H / s, W / s] BEV features
```

`A = (H / s) * (W / s) * num_classes * 2`. H x W is the pillar grid, s the backbone output
stride (2 at defaults), and each class gets anchors at headings 0 and pi/2.
At defaults, A = 248 * 216 * 6 = 321,408.

The output is one box per anchor, not a final detection list. To get detections, keep the boxes
whose score passes a threshold and run NMS, for example `torchvision.ops.nms` on BEV rectangles
or a rotated NMS.

## Training

```python
losses = model.detection_loss(output, gt_boxes, gt_labels)
losses["loss"].backward()   # also returns the "cls", "box" and "dir" parts
```

- `gt_boxes`: `[B, M, 7]` as `(x, y, z, l, w, h, yaw)`, with z at the box centre.
- `gt_labels`: `[B, M]` class ids in `range(num_classes)`. Use `-1` for padding slots.

Anchors are matched to boxes of the same class by BEV IoU. Each box is first snapped to the
nearest axis-aligned rectangle, as in SECOND/OpenPCDet. Anchors at or above `positive_iou` are
positive, and so is the best anchor of each box. Anchors below `negative_iou` are negative.
The loss has three parts, weighted 1, 2 and 0.2 as in the paper:

- focal loss for the class scores,
- smooth-L1 loss on the SECOND box coding, using sin(Δyaw) for the heading,
- cross-entropy for the heading direction.

## Configuration

| key | default | meaning |
| --- | --- | --- |
| `input_dim` | 4 | XYZ plus point features (intensity) |
| `point_cloud_range` | `(0, -39.68, -3, 69.12, 39.68, 1)` | x/y/z min then max, in metres |
| `pillar_size` | `(0.16, 0.16)` | pillar footprint; grid = 432 x 496 |
| `pillar_channels` | 64 | pillar feature width |
| `layer_nums` / `layer_strides` / `num_filters` | `(3, 5, 5)` / `(2, 2, 2)` / `(64, 128, 256)` | backbone blocks |
| `upsample_strides` / `upsample_filters` | `(1, 2, 4)` / `(128, 128, 128)` | upsampling to a common stride |
| `anchor_sizes` | car, pedestrian, cyclist `(l, w, h)` | one class per entry |
| `anchor_bottom_heights` | `(-1.78, -0.6, -0.6)` | anchor bottom z per class |
| `positive_iou` / `negative_iou` | `(0.6, 0.5, 0.5)` / `(0.45, 0.35, 0.35)` | matching thresholds per class |

The pillar grid must be divisible by the product of `layer_strides`.

```python
from models import ModelInput, create_model

model = create_model({"name": "pointpillars"})  # KITTI car / pedestrian / cyclist
output = model(ModelInput(coordinates=xyz, point_features=intensity))
```

## Differences from the paper

- There is no cap on points per pillar or on the number of pillars. All points in range are
  encoded and max-pooled with `scatter_reduce`.
- Anchors are matched with a nearest-axis BEV IoU, not a rotated IoU.
- Post-processing (score threshold and NMS) is left to the caller.

## Limits

- The metric range and pillar size are fixed at build time. Points outside the range are dropped.
- Vertical structure is reduced to pillar features, so stacked or small objects suffer.

## References

- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784), Lang et al., CVPR 2019.
- [SECOND: Sparsely Embedded Convolutional Detection](https://www.mdpi.com/1424-8220/18/10/3337), Yan et al., 2018 (box coding, direction head).
- [OpenPCDet reference implementation](https://github.com/open-mmlab/OpenPCDet)
