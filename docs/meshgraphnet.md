# MeshGraphNet

MeshGraphNet implements the encode-process-decode architecture from Pfaff et
al., *Learning Mesh-Based Simulation with Graph Networks* (ICLR 2021). The
implementation was compared with DeepMind's original `core_model.py` and the
maintained NVIDIA PhysicsNeMo MeshGraphNet implementation.

```text
raw node and edge features
           |
  separate node/edge encoders
           |
 processor blocks x K
   (edge update, sum, node update)
           |
       node decoder
           |
  per-node physical predictions
```

## Architecture

The encoder uses separate MLPs to map node and edge attributes to a shared
latent width. Every processor step has independent parameters. For directed
edge `s -> r`, a block computes

```text
delta_e_sr = edge_mlp([h_s, h_r, e_sr])
m_r        = sum_(s -> r) delta_e_sr
delta_h_r  = node_mlp([h_r, m_r])
e_sr'      = e_sr + delta_e_sr
h_r'       = h_r + delta_h_r
```

The edge update is calculated before the node update, and the node MLP consumes
the newly calculated edge messages. Sum aggregation and residual updates match
the canonical model. Encoders and processor MLPs use LayerNorm by default; the
decoder does not. `mlp_layers` counts hidden layers, each followed by the
configured activation, before a final linear output layer.

## Input and output contract

`MeshGraphNet` requires:

- `node_features`: floating tensor `[N, node_input_dim]`
- `edge_index`: integer tensor `[2, E]`, with rows `[sender, receiver]`
- `edge_features`: floating tensor `[E, edge_input_dim]`

It accepts variable graph sizes and disconnected graphs concatenated into one
batch. An optional `[N]` `batch` vector is passed through in output metadata;
there is no global pooling. `output.predictions` has shape `[N, output_dim]`,
`output.latent` contains final node latents, and the final edge latents are in
`output.auxiliary["edge_latent"]`.

Undirected meshes should provide both edge directions. Typical 3D geometric
edge features are `[dx, dy, dz, distance]`, where displacement is consistently
oriented from sender to receiver. Node features may freely combine static data
(coordinates, node/boundary type, material properties) and dynamic state
(velocity, displacement, pressure, or temperature).

## Graph construction and simulation semantics

Mesh parsing and graph construction belong in a dataset or
`RepresentationAdapter`, not in `MeshGraphNet.forward`. The synthetic example
shows how to form bidirectional k-nearest-neighbor edges and relative-position
edge attributes. Real triangle, tetrahedral, FEM, or CFD adapters can produce
the same `ModelInput` contract.

The model makes one task-neutral prediction. It does not decide whether that
prediction means a next state, delta, acceleration, or another target. Rollout
logic applies repeated predictions externally. Likewise, dataset running-stat
normalization and training-time noise injection remain outside the deterministic
network; LayerNorm inside an MLP is not a substitute for data normalization.

The differentiable `ModelOutput.predictions` can be used unchanged with
supervised data losses, physics/PDE and boundary-condition losses, or a weighted
hybrid objective. MeshGraphNet itself contains no PDE, optimizer, backward pass,
or NumPy conversion.

## Configuration and usage

```python
from models import MeshGraphNetConfig, ModelInput, create_model

config = MeshGraphNetConfig(
    node_input_dim=6,
    edge_input_dim=4,
    output_dim=3,
    latent_dim=128,
    processor_steps=15,
    mlp_hidden_dim=128,
    mlp_layers=2,
    activation="relu",
    normalization="layer_norm",
    residual=True,
)
model = create_model(config)
output = model(ModelInput(
    node_features=node_features,
    edge_index=edge_index,
    edge_features=edge_features,
    batch=batch,
    representation="GRAPH",
))
prediction = output.predictions
```

Run the graph-dependent one-step diffusion example with:

```bash
python examples/train_meshgraphnet_synthetic.py --epochs 30
```

The official `cylinder_flow` and `flag_simple` datasets can later be adapted by
reproducing their task-specific node features, edge construction, normalizers,
targets, and external rollout/update rules. No external dataset is required by
this implementation.

## References and intentional scope

- [Pfaff et al., ICLR 2021](https://openreview.net/forum?id=roNqYL0_XP)
- [Official DeepMind MeshGraphNets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
- [Original `core_model.py`](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/core_model.py)
- [NVIDIA PhysicsNeMo MeshGraphNet](https://docs.nvidia.com/physicsnemo/latest/api/physicsnemo.models.meshgraphnet.html)

The generic project model supports one edge set rather than DeepMind's named
multi-edge-set container. This is sufficient for the standard mesh-edge base
model requested here and keeps the established `ModelInput.edge_index` /
`edge_features` contract. World edges, multiscale processing, attention, and
other advanced variants are intentionally excluded. Optional deterministic 2D
refinement is described below. Efficient
PyTorch `index_add_` performs aggregation because the project has no PyG
dependency; no Python loop runs over edges.

## Optional adaptive mesh rollout

Adaptive meshing is external to `MeshGraphNet`; fixed mode remains the default
and does not compute indicators or copy/rebuild a mesh:

```text
fixed:     mesh -> graph -> MeshGraphNet -> task state update -> next step

adaptive:  mesh -> graph -> MeshGraphNet -> task state update
                                                    |
                 next step <- rebuilt graph <- state transfer <- remesher
                                                    ^              ^
                                             correspondence   local variation
```

This is a **simplified initial adaptive implementation**, not an exact
reproduction of the paper's learned anisotropic remesher. Its components are:

- `TriangularMesh`: 2D vertices, triangle cells, explicit tagged boundary
  edges, per-node data, per-cell material/region data, and refinement levels.
- `LocalVariationCriterion`: maximum edge-wise gradient magnitude of a
  configured scalar or vector state field in each triangle.
- `TriangleCentroidRemesher`: replaces selected triangles with three valid
  child triangles. It creates no hanging edge nodes, leaves boundary edges
  unchanged, and makes children inherit parent cell data. It supports
  refinement only; coarsening, edge flips, and anisotropic sizing are not
  exposed as configuration.
- `BarycentricStateTransfer`: averages continuous values from parent vertices.
  Categorical values select a parent rather than being averaged. Boundary
  metadata comes from geometry, and new interior nodes are not boundaries.
- `TriangleMeshGraphBuilder`: recreates bidirectional connectivity and
  `[dx, dy, distance]` features. Stale graph edges are never reused.
- `MeshAdaptationController` owns scheduling and composition;
  `MeshSimulationRunner` uses the same model and input/output contracts in both
  modes.

Configuration exposes only implemented behavior:

```python
from mesh import MeshAdaptationConfig, MeshMode

fixed = MeshAdaptationConfig()
adaptive = MeshAdaptationConfig(
    mode=MeshMode.ADAPTIVE,
    adapt_every=5,
    criterion_field="temperature",
    refinement_threshold=2.0,
    max_nodes=20_000,
    max_elements=40_000,
    max_refinement_level=3,
)
```

Run the localized-Gaussian example in both modes:

```bash
python examples/adaptive_meshgraphnet_synthetic.py --mesh-mode fixed
python examples/adaptive_meshgraphnet_synthetic.py --mesh-mode adaptive
```

Topology decisions are discrete and outside autograd. Model inference remains
unchanged and differentiable; barycentric transfer uses PyTorch operations. A
future learned criterion can implement `AdaptationCriterion` without placing
remeshing in `forward()`.

### Relationship to the reference method

Pfaff et al. predict a symmetric positive-definite sizing tensor at every node
with a separate MeshGraphNet. A generic triangular remesher uses it to split
invalid edges, flip edges with an anisotropic Delaunay criterion, collapse
edges that do not create invalid neighbors, and flip again. Dynamics and sizing
models are supervised separately; rollout updates dynamics, predicts sizing,
then remeshes. The paper also estimates sizing targets from consecutive meshes.

DeepMind's public repository includes the core model, fixed-mesh demonstration
rollouts, adaptive trajectory datasets, and `*_dynamic_sizing` datasets with
pre-remesh meshes and sizing-field targets. It does not publish the paper's
local remesher. We reproduce the separation among dynamics, criterion,
geometric remeshing, transfer, and graph reconstruction, while beginning with
deterministic isotropic refinement.

Optional PyTorch Geometric conversion is available through
`TriangleMeshGraphBuilder.to_pyg_data()` when the `pyg` extra is installed. The
established `ModelInput` path remains the dependency-free native interface.
