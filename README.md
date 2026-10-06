# Neural3DBench

Neural3DBench is a PyTorch-native framework for implementing and reproducibly
benchmarking neural networks for 3D geometry and physics simulations.

> [!NOTE]
> The project is under active development. Its model API is usable, while the
> common datasets, benchmark protocols, and published leaderboards are still
> being developed.

## Supported models

| Model | Representation | Status |
| --- | --- | --- |
| Graph Convolutional Network (GCN) | Graph | Implemented |
| Graph Attention Network (GAT) | Graph | Implemented |
| MeshGraphNet | Mesh graph | Implemented |
| Adaptive MeshGraphNet | Adaptive mesh graph | Experimental |
| PointNet | Point cloud | Implemented |
| PointNet++ | Hierarchical point cloud | Implemented |
| 3D U-Net | Structured voxel grid | Implemented |

## Installation

Neural3DBench requires Python 3.10 or newer. For local development:

```bash
git clone https://github.com/YOUR_USERNAME/neural3dbench.git
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

MeshGraphNet also has an optional external adaptive-triangle rollout pipeline.
See [the MeshGraphNet documentation](docs/meshgraphnet.md#optional-adaptive-mesh-rollout)
and `examples/adaptive_meshgraphnet_synthetic.py`. Fixed mesh mode remains the
default and both modes share the same neural architecture.

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
