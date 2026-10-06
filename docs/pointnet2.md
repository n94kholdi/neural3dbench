# PointNet++

PointNet++ extends PointNet with hierarchical local feature learning. This
implementation follows its single-scale grouping architecture using standard
PyTorch operations and the repository's existing point-cloud contract.

## Architecture

The encoder contains two local set-abstraction levels followed by global set
abstraction:

```text
point cloud
  → farthest-point sampling
  → radius neighborhood grouping
  → local shared MLP and max pooling
  → coarser sampling and grouping
  → global PointNet aggregation
```

For point-wise prediction, three-neighbor inverse-distance interpolation and
shared MLPs propagate coarse features back to every original point. The
classification mode reuses the same classification head as PointNet.

Sampling starts deterministically with the point furthest from the valid-cloud
mean. Padded batches are supported through `ModelInput.point_mask`; invalid
points do not participate in sampling, neighborhoods, or pooling, and their
point-wise outputs are zero. If a cloud contains fewer points than a configured
sampling level, all available tensor positions are used and validity masks
prevent padded points from becoming centroids.

## Configuration

Create the model through the common registry using `pointnet2` or its
`pointnet++` alias:

```python
model = create_model({
    "name": "pointnet2",
    "input_dim": 6,             # XYZ plus three optional attributes
    "output_dim": 4,
    "task": "segmentation",    # or classification
    "sample_counts": [512, 128],
    "radii": [0.2, 0.4],
    "neighbor_counts": [32, 64],
    "abstraction_channels": [
        [64, 64, 128],
        [128, 128, 256],
    ],
    "global_dim": 1024,
    "dropout": 0.3,
})
```

Input and output formats match PointNet:

- `coordinates`: `[B, N, 3]`
- optional `point_features`: `[B, N, input_dim - 3]`
- optional boolean `point_mask`: `[B, N]`
- classification output: `[B, output_dim]`
- point-wise output: `[B, N, output_dim]`

The existing normalization, random sampling, padded batching, mesh extraction,
and visualization functions in `pointcloud` work for both architectures.
Sampled coordinates and masks for all abstraction levels are available in
`output.auxiliary` for inspection or visualization.

## Synthetic validation

```bash
python examples/train_pointnet_synthetic.py --model pointnet2 --task classification
python examples/train_pointnet_synthetic.py --model pointnet2 --task segmentation
pytest tests/test_pointnet2.py
```

The example automatically chooses smaller sampling levels for its compact
synthetic clouds. Production settings should select radii after normalizing the
domain and considering point density.

## Use with physics-informed models

PointNet++ provides local, multiscale geometry features useful for irregular or
changing domains. Its farthest-point sampling and radius membership are
discrete operations, however. For PDEs that require smooth coordinate
derivatives, use PointNet++ as a geometry encoder and feed its features into a
smooth coordinate/Fourier decoder; calculate PDE derivatives through that
decoder with respect to the original physical coordinates. Mesh-based models
remain preferable when explicit connectivity, oriented fluxes, or strict local
conservation are central to the problem.
