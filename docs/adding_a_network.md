# Adding a new network

This project keeps the model interface stable while allowing architecture-specific implementations to vary.

## 1. Start from the common interface

Every network should inherit from `BaseNetwork` and implement:

- `required_inputs`
- `forward(inputs: ModelInput) -> ModelOutput`
- `validate_inputs(inputs)` through the base implementation

The base class does not know about graph edges, voxel grids, Fourier modes, or attention heads. Those are part of each concrete model.

## 2. Model input structure

Use `ModelInput` for the network contract. It is intentionally flexible and contains optional fields such as:

- coordinates
- node_features
- edge_index
- edge_features
- point_features
- grid
- voxel_fields
- material_properties
- physical_parameters
- time
- global_features
- batch
- metadata

A GCN model can require `coordinates`, `node_features`, and `edge_index`; a PINN may only require `coordinates` and `physical_parameters`.

## 3. Model outputs

All models return `ModelOutput` so generic training code can access `output.predictions` and `output.main_prediction` regardless of architecture.

Additional latent states or auxiliary outputs can be attached via `latent` and `auxiliary`.

## 4. Configuration object

Each architecture should define its own config subclass of `BaseModelConfig`.

Example:

```python
from models import BaseModelConfig


class GCNConfig(BaseModelConfig):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 3, **kwargs):
        super().__init__(name="gcn", **kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
```

This keeps architecture-specific parameters isolated and serializable.

## 5. Registration

Register a model so it participates in the factory.

```python
from models import BaseNetwork, ModelRegistry

class GCN(BaseNetwork):
    @property
    def required_inputs(self):
        return {"coordinates", "node_features", "edge_index"}

    def forward(self, inputs):
        self.validate_inputs(inputs)
        return ModelOutput(predictions=inputs.coordinates)


ModelRegistry.register("gcn", GCN)
```

## 6. Representation adapters

Raw data should be converted to `ModelInput` before it reaches the model. This keeps dataset and simulation code separate from the network implementation.

```python
from models import GraphRepresentationAdapter

adapter = GraphRepresentationAdapter()
model_input = adapter.adapt({
    "coordinates": coords,
    "node_features": features,
    "edge_index": edges,
})
```

## 7. Capabilities and compatibility

Declare architecture capabilities with the `capabilities` property or a lightweight model metadata object. This helps validate compatibility between a dataset, representation adapter, and network.

## 8. Testing checklist

When adding a model, include tests for:

- registration
- config creation
- validation for missing required inputs
- optional input acceptance
- model output access
- a minimal forward pass

The repository contains a small infrastructure test suite covering these conditions.
