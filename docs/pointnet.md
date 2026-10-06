# PointNet

PointNet is a neural network for unordered point sets based on Qi et al.,
*PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*.
It applies the same MLP to every point, uses learned input and feature transforms,
and obtains a permutation-invariant global representation through max pooling.

## Input contract

PointNet uses the common `ModelInput` interface:

```python
inputs = ModelInput(
    coordinates=points,          # [B, N, 3], required XYZ
    point_features=features,     # [B, N, input_dim - 3], optional
    point_mask=valid_points,     # [B, N] bool, optional for padded batches
    representation="POINT_CLOUD",
)
```

Coordinates stay separate from normals, materials, boundary conditions, or
other per-point attributes. The input T-Net transforms XYZ only; optional
features are concatenated afterward. PointNet needs neither edges nor mesh
connectivity. `PointCloudRepresentationAdapter` accepts the aliases
`points`/`features`, and `pointcloud.mesh_to_point_cloud` extracts mesh vertices
and selected node fields without modifying or discarding the source mesh.

`pointcloud.normalize_points`, `sample_points`, and `collate_point_clouds`
provide centering/unit-radius scaling, random fixed-count sampling, and padded
variable-length batching. Padded entries are excluded from global max pooling
with `point_mask`, and padded point-wise predictions are zero.

## Architecture and tasks

The encoder follows the standard pipeline: XYZ input T-Net, shared MLP, 64-D
feature T-Net, shared MLP, and symmetric max aggregation. The resulting global
feature is independent of point order. Heads are separate modules, so future
tasks and PointNet++ can reuse or replace components without changing the model
factory or training code.

Classification returns `[B, output_dim]`:

```python
model = create_model({
    "name": "pointnet",
    "input_dim": 3,
    "output_dim": 3,
    "task": "classification",
})
logits = model(ModelInput(coordinates=points)).predictions
```

Segmentation is the configuration name for any point-wise prediction task. It
concatenates the global representation with each 64-D local feature and returns
`[B, N, output_dim]`:

```python
model = create_model({
    "name": "pointnet",
    "input_dim": 6,  # XYZ plus three additional features
    "output_dim": 2,
    "task": "segmentation",
})
field = model(ModelInput(
    coordinates=points,
    point_features=normals,
)).predictions
```

Configuration options are `input_dim` (XYZ plus feature channels), `output_dim`,
`task`, `global_dim` (default 1024), `dropout`, `input_transform`, and
`feature_transform`. Learned transforms are returned in `output.auxiliary`.
`feature_transform_regularizer` implements the paper's optional orthogonality
penalty for a training objective.

## Synthetic validation and visualization

Train sphere/cube/cylinder classification or the point-wise field
`f(x, y, z) = x² + y² + z²`:

```bash
python examples/train_pointnet_synthetic.py --task classification --epochs 10
python examples/train_pointnet_synthetic.py --task segmentation --epochs 10
```

Add `--visualize` to show the input/class or ground truth, prediction, and
absolute error. Plotting is optional and installed with `pip install -e
".[viz]"`. The reusable `pointcloud.plot_point_cloud` also accepts point labels
and scalar or vector fields (vectors are colored by magnitude).

Run PointNet tests alone with `pytest tests/test_pointnet.py`, or use `pytest`
for the full compatibility suite.

## Limitation

PointNet processes points independently before global aggregation. It therefore
does not explicitly model local mesh connectivity or neighborhood message
passing like GCN, GAT, or MeshGraphNet. That simplicity provides permutation
invariance, but can miss fine local structure that neighborhood-aware models
such as PointNet++ are designed to capture.
