Use one representation-based `.npz` file for each input sample. Do not include a model name; include `schema_version` and `representation`.

## Common metadata

Every file should contain:

```text
schema_version       string
representation       string
coordinate_system    string
units                string
normalization_id     string
```

`normalization_id` identifies the preprocessing used during model training.

## 1. MeshGraphNet — mesh graph

File: `mesh_graph.npz`

```text
representation = "MESH_GRAPH"

coordinates      float32 [N, 3]    recommended
node_features    float32 [N, Fn]   required
edge_index       int64   [2, E]    required
edge_features    float32 [E, Fe]   required
batch            int64   [N]       optional
node_feature_names string [Fn]
edge_feature_names string [Fe]
```

Requirements:

- `edge_index[0]` contains senders; `edge_index[1]` contains receivers.
- Undirected mesh edges must appear in both directions.
- Typical edge features are `dx, dy, dz, distance`.
- For one mesh, `batch` is optional; all nodes belong to batch zero.

This representation can also serve GCN and GAT after compatibility validation.

## 2. PointNet — point cloud

File: `point_cloud.npz`

```text
representation = "POINT_CLOUD"

coordinates         float32 [B, N, 3]  required
point_features      float32 [B, N, F]  optional
point_mask          bool    [B, N]     optional
query_coordinates   float32 [B, Q, 3]  optional (GINO output locations)
point_feature_names string  [F]
```

Requirements:

- For one geometry, use `B = 1`.
- `point_mask` is only necessary when clouds are padded.
- Coordinates must use the same centering/scaling used during training.
- For GINO, source and query coordinates must both be normalized to `[0,1]`
	on each axis using the same coordinate frame. Geocore supports
	`normalization="unit_box_01"` for this.
- `query_coordinates` supplies output locations for source-to-query models
	such as GINO. When omitted, GINO predicts at the source coordinates.
- `3 + F` must equal PointNet’s configured `input_dim`.
- Faces and connectivity are not included.

This representation may also serve PointNet++, Transolver, GINO, and SIREN, subject to their feature requirements.

## 3. FNO3D — structured voxel grid

File: `voxel_grid.npz`

```text
representation = "VOXEL_GRID"

voxel_fields     float32 [B, C, D, H, W]  required
channel_names    string  [C]
origin           float32 [3]               recommended
spacing          float32 [3]               recommended
bounds           float32 [2, 3]            recommended
```

Requirements:

- For one geometry, use `B = 1`.
- Spatial tensor order is `D,H,W = Z,Y,X`.
- `C` must equal FNO’s configured `input_channels`.
- Channels can represent occupancy, signed distance, material, source terms, or boundary masks.
- Channel order and normalization must match training.
- Geometry must be rasterized as surfaces, filled volume, or SDF according to the trained model—not merely as occupied vertices.

This representation can also serve 3D U-Net.

In neural3dbench, load the canonical mapping and pass it through
`GridRepresentationAdapter`. The adapter retains `voxel_fields` and metadata
while translating the interchange name `VOXEL_GRID` to the models' internal
representation name `GRID`. See `examples/geocore_fno3d_integration.py` and
`examples/train_fno3d_geocore_heat.py` for validated and trained paths.

## 4. Deep NURBS — parametric surface

File: `parametric_surface.npz`

```text
representation = "PARAMETRIC_SURFACE"

degree                    int64   [2]
knot_vector_u             float32 [Ku]
knot_vector_v             float32 [Kv]
control_points            float32 [Nu, Nv, 2]
weights                   float32 [Nu, Nv]
admissible_coefficients   float32 [Nu, Nv]
lifting_coefficients      float32 [Nu, Nv, output_dim]
parametric_coordinates    float32 [B, N, 2]
```

Requirements:

- UV coordinates must be within `[0,1]`.
- `Ku = Nu + degree_u + 1`.
- `Kv = Nv + degree_v + 1`.
- Weights must be positive.
- Knot vectors must be clamped and nondecreasing.
- The admissible coefficients must be zero on all four boundaries.

Current limitation: Deep NURBS accepts only physical control points shaped `[Nu,Nv,2]`. It represents a 2D physical domain, not a general 3D NURBS surface. Geometry data is model configuration rather than normal runtime `ModelInput`. Use `DeepNURBSConfig.from_parametric_surface(payload)` for the geometry and `ParametricSurfaceRepresentationAdapter().adapt(payload)` for the sampled runtime coordinates. See `examples/geocore_deep_nurbs_integration.py` for the validated save/reload path.

In summary:

| Representation file | Primary model | Main required arrays |
|---|---|---|
| `mesh_graph.npz` | MeshGraphNet | node features, edges, edge features |
| `point_cloud.npz` | PointNet, GINO | XYZ coordinates; optional query coordinates for GINO |
| `voxel_grid.npz` | FNO3D | dense voxel fields |
| `parametric_surface.npz` | Deep NURBS | NURBS definition and UV samples |
