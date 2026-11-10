# Neural3DBench

Neural3DBench is a PyTorch-native framework for implementing and reproducibly
benchmarking neural networks for 3D geometry and physics simulations.

> [!NOTE]
> The project is under active development. Its model API is usable, while the
> common datasets, benchmark protocols, and published leaderboards are still
> being developed.

## Supported models

Models are grouped by input representation (advisor names in code font), then by task family.
Rows marked **To do** are planned additions.

<table>
<thead><tr><th>Representation</th><th>Task</th><th>Model</th><th>Status</th></tr></thead>
<tbody>
<tr><td rowspan="22"><b>Point cloud</b><br><code>point_cloud</code></td><td rowspan="8">Backbones: classification and segmentation</td><td><a href="docs/pointnet.md">PointNet</a></td><td>Implemented</td></tr>
<tr><td><a href="docs/pointnet2.md">PointNet++</a></td><td>Implemented</td></tr>
<tr><td><a href="docs/pointnext.md">PointNeXt</a></td><td>Implemented</td></tr>
<tr><td><a href="docs/dgcnn.md">DGCNN</a></td><td>Implemented</td></tr>
<tr><td><a href="https://arxiv.org/abs/2312.10035">Point Transformer V3 (PTv3)</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/1904.08889">KPConv</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2202.07123">PointMLP</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2104.12229">Vector Neurons (VN-DGCNN / VN-PointNet)</a></td><td>To do</td></tr>
<tr><td rowspan="2">Instance segmentation</td><td><a href="https://arxiv.org/abs/2004.01658">PointGroup</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2210.03105">Mask3D</a></td><td>To do</td></tr>
<tr><td rowspan="5">Shape completion and generation</td><td><a href="docs/pcn.md">PCN</a></td><td>Implemented</td></tr>
<tr><td><a href="docs/foldingnet.md">FoldingNet</a></td><td>Implemented</td></tr>
<tr><td><a href="https://arxiv.org/abs/1906.12320">PointFlow</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2104.03670">Point-Voxel Diffusion (PVD)</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2210.06978">LION</a></td><td>To do</td></tr>
<tr><td rowspan="2">Object detection</td><td><a href="docs/pointpillars.md">PointPillars</a></td><td>Implemented</td></tr>
<tr><td><a href="docs/votenet.md">VoteNet</a></td><td>Implemented</td></tr>
<tr><td rowspan="5">Registration and scene flow</td><td><a href="docs/dcp.md">Deep Closest Point (DCP)</a></td><td>Implemented</td></tr>
<tr><td><a href="https://arxiv.org/abs/2003.13479">RPM-Net</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/1910.12240">PRNet</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2202.06688">GeoTransformer</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/1806.01411">FlowNet3D</a></td><td>To do</td></tr>
<tr><td rowspan="14"><b>Mesh and graph</b><br><code>mesh_graph</code></td><td rowspan="4">General graph networks</td><td><a href="docs/gcn.md">Graph Convolutional Network (GCN)</a></td><td>Implemented</td></tr>
<tr><td><a href="docs/gat.md">Graph Attention Network (GAT)</a></td><td>Implemented</td></tr>
<tr><td><a href="https://arxiv.org/abs/2102.09844">EGNN</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2006.10503">SE(3)-Transformer</a></td><td>To do</td></tr>
<tr><td rowspan="4">Mesh surface networks</td><td><a href="docs/meshnet.md">MeshNet</a></td><td>Implemented</td></tr>
<tr><td><a href="docs/diffusionnet.md">DiffusionNet</a></td><td>Implemented</td></tr>
<tr><td><a href="docs/meshwalker.md">MeshWalker</a></td><td>Implemented</td></tr>
<tr><td><a href="https://arxiv.org/abs/1809.05910">MeshCNN</a></td><td>To do</td></tr>
<tr><td rowspan="6">Physics surrogates on meshes</td><td><a href="docs/meshgraphnet.md">MeshGraphNet</a></td><td>Implemented</td></tr>
<tr><td>Adaptive MeshGraphNet</td><td>Experimental</td></tr>
<tr><td><a href="docs/transolver.md">Transolver</a></td><td>Implemented</td></tr>
<tr><td><a href="https://arxiv.org/abs/2210.02573">Multiscale MeshGraphNet (BSMS-GNN)</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2411.17164">X-MeshGraphNet</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2302.14376">GNOT</a></td><td>To do</td></tr>
<tr><td rowspan="8"><b>Voxel grid</b><br><code>voxel_grid</code></td><td rowspan="3">Dense and sparse CNNs</td><td><a href="docs/unet3d.md">3D U-Net</a></td><td>Implemented</td></tr>
<tr><td><a href="https://arxiv.org/abs/1904.08755">MinkowskiNet</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/1907.03739">PVCNN</a></td><td>To do</td></tr>
<tr><td rowspan="2">Object detection</td><td><a href="https://www.mdpi.com/1424-8220/18/10/3337">SECOND</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2006.11275">CenterPoint</a></td><td>To do</td></tr>
<tr><td rowspan="3">Neural operators on grids</td><td><a href="docs/fno.md">Fourier Neural Operator (FNO3D)</a></td><td>Implemented</td></tr>
<tr><td><a href="https://arxiv.org/abs/2204.11127">U-NO</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2111.13802">F-FNO</a></td><td>To do</td></tr>
<tr><td rowspan="10"><b>Continuous coordinates</b><br><code>continuous_coords</code></td><td rowspan="6">Implicit neural fields</td><td><a href="docs/siren.md">SIREN</a></td><td>Implemented</td></tr>
<tr><td><a href="docs/mlp.md">MLP</a></td><td>Implemented</td></tr>
<tr><td><a href="docs/deep_nurbs.md">Deep NURBS</a></td><td>Implemented</td></tr>
<tr><td><a href="https://arxiv.org/abs/1901.05103">DeepSDF</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2003.04618">Convolutional Occupancy Networks</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2201.05989">Instant-NGP hash encoding</a></td><td>To do</td></tr>
<tr><td rowspan="4">Neural operators on irregular geometry</td><td><a href="docs/gino.md">Geometry-Informed Neural Operator (GINO)</a></td><td>Implemented</td></tr>
<tr><td><a href="https://arxiv.org/abs/1910.03193">DeepONet</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2207.05209">Geo-FNO</a></td><td>To do</td></tr>
<tr><td><a href="https://arxiv.org/abs/2105.14995">Galerkin Transformer</a></td><td>To do</td></tr>
</tbody>
</table>

## Tools

- **neural3d-advisor**: an MCP server that ranks registered models for a problem
  description, explains their trade-offs and estimates memory. It returns `no_fit`
  instead of a wrong model when no registered network covers the task. See
  [tools/mcp_advisor/README.md](tools/mcp_advisor/README.md).
- **Dashboard**: a Streamlit app that combines the advisor, 3D data preview, real
  dataset download (ShapeNetPart, ModelNet10/40) and background MLflow training.
  See [tools/dashboard/README.md](tools/dashboard/README.md).

## Installation

Neural3DBench requires Python 3.10 or newer. For local development:

```bash
git clone https://github.com/n94kholdi/neural3dbench.git
cd neural3dbench
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,pyg]"
```

The `pyg` extra is optional and installs PyTorch Geometric support.

## Architecture overview

The core idea is a stable abstraction layer between raw data, representation adapters, and model implementations:

raw simulation/problem data
↓
RepresentationAdapter
↓
ModelInput
↓
BaseNetwork
↓
ModelOutput

The framework intentionally avoids hard-coding assumptions about graphs, point clouds, grids, or PINN coordinate inputs into the shared model API.

## Core package

- `models.BaseNetwork`: abstract model contract
- `models.ModelInput`: flexible input container for heterogeneous data
- `models.ModelOutput`: common output abstraction
- `models.ModelRegistry`: registry/factory for model creation
- `models.RepresentationAdapter`: interface for representation conversion

## Quick start

```python
import torch

from models import GCNConfig, ModelInput, ModelRegistry

model = ModelRegistry.create(
    "gcn",
    GCNConfig(input_dim=3, hidden_dim=64, output_dim=1),
)
output = model(ModelInput(
    node_features=torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]),
    edge_index=torch.tensor([[0, 1], [1, 0]]),
))
print(output.predictions)
```

New architectures can be added by implementing a model subclass, config, and model registry entry without changing the common training flow.

PointNet supports both global classification and point-wise prediction from
unordered `[B, N, 3]` point clouds, with optional per-point features. See
[the PointNet documentation](docs/pointnet.md) for configuration, mesh-to-point
conversion, visualization, and synthetic examples.

PointNet++ adds farthest-point sampling, radius neighborhoods, hierarchical set
abstraction, and feature propagation while keeping the same input and task
interfaces. See [the PointNet++ documentation](docs/pointnet2.md).

3D U-Net provides dense volumetric segmentation and scalar/vector field
prediction for `[B, C, D, H, W]` voxel grids. It includes separate point/mesh
voxelization helpers and orthogonal slice visualization. See
[the 3D U-Net documentation](docs/unet3d.md).

FNO3D learns dense operators between physical fields on regular 3D grids using
truncated spectral convolutions plus pointwise local branches. It preserves
`[B, C, D, H, W]` resolution and can append normalized grid coordinates. It is
not an irregular point-cloud or mesh model. See [the FNO documentation](docs/fno.md)
for mode indexing, boundary assumptions, training, and benchmarking.

MeshGraphNet also has an optional external adaptive-triangle rollout pipeline.
See [the MeshGraphNet documentation](docs/meshgraphnet.md#optional-adaptive-mesh-rollout)
and `examples/adaptive_meshgraphnet_synthetic.py`. Fixed mesh mode remains the
default and both modes share the same neural architecture.

Transolver predicts point-wise physical fields on irregular domains without
requiring mesh edges. Its Physics-Attention layers aggregate points into learned
physical-state slices, attend among those slices, and map information back to
points. See [the Transolver documentation](docs/transolver.md) for tensor shapes,
configuration, reference-fidelity notes, synthetic training, and benchmarking.

SIREN represents a continuous field with a coordinate-to-value sine network.
It evaluates irregular points independently, supports coordinate derivatives,
and can serve as the function approximator in physics-informed training; it is
not itself a PINN or a neural operator. See [the SIREN documentation](docs/siren.md)
for initialization, tensor contracts, derivatives, masking, optional coordinate
scaling, frequency settings, and the synthetic 3D field example.

MLP is the standard pointwise coordinate baseline with `tanh`, `silu`, `relu`,
or `gelu` activations, configurable widths, optional point features, and
optional residual connections. It shares SIREN's input contract, so it can
replace SIREN in coordinate-based training loops. See
[the MLP documentation](docs/mlp.md).

Deep NURBS combines a NURBS admissible field with a shallow coordinate MLP to
represent PDE trial solutions while strongly imposing homogeneous Dirichlet
conditions. It also evaluates the configured NURBS surface to map parametric
coordinates to physical coordinates. The model does not include a PDE loss or
training loop. See [the Deep NURBS documentation](docs/deep_nurbs.md) for the
paper, geometry contract, configuration details, and the geocore
`PARAMETRIC_SURFACE` CPU training example.

Run the test suite with:

```bash
pytest
```

## Benchmarking roadmap

The public benchmark suite will standardize datasets and splits, training
budgets, evaluation metrics, random seeds, and machine-readable result files.
Until that protocol is published, results from the example scripts should be
treated as development checks rather than comparable benchmark scores.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the local
development and pull-request workflow.

## License

Neural3DBench is licensed under the [Apache License 2.0](LICENSE).
