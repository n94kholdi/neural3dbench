# 3D Neural Networks

This project provides a lightweight, PyTorch-native foundation for physics-informed and geometry-aware neural models.

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

## Example

```python
from models import ModelInput, ModelRegistry

ModelRegistry.register("dummy_graph", DummyGraphModel)

model = ModelRegistry.create("dummy_graph", {"name": "dummy_graph", "hidden_dim": 64})
output = model(ModelInput(
    coordinates=[[0.0, 0.0], [1.0, 0.0]],
    node_features=[[1.0], [2.0]],
    edge_index=[[0, 1], [1, 0]],
))
```

New architectures can be added by implementing a model subclass, config, and model registry entry without changing the common training flow.
