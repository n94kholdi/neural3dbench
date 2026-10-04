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

## 9. Example: Graph Convolutional Network (GCN)

The project includes a graph package for architectures that operate on node features and connectivity, with the shared graph validation logic living in `models/graph/base.py` and the model-specific implementation in `models/graph/gcn.py`.

The GCN expects a graph input contract like:

```python
inputs = ModelInput(
    node_features=torch.randn(N, F),
    edge_index=torch.tensor([[src_0, src_1, ...], [dst_0, dst_1, ...]]),
)
```

Input contract:

- `node_features`: shape `[N, F]`
- `edge_index`: shape `[2, E]`
- optional: `coordinates`, `batch`, `edge_features` as metadata or future pipeline inputs

Output contract:

```python
output = model(inputs)
node_predictions = output.predictions
# shape: [N, output_dim]
```

The model is intentionally node-centric. It does not assume that arbitrary edge features are part of the convolution. For future architectures such as MeshGraphNet or GAT, those signals will be handled explicitly by the model itself.

Example usage:

```python
from models import ModelInput, create_model

model = create_model({
    "name": "gcn",
    "input_dim": 4,
    "output_dim": 2,
    "hidden_dim": 64,
    "num_layers": 3,
    "activation": "gelu",
    "dropout": 0.1,
    "normalization": "layer_norm",
    "residual": True,
})

inputs = ModelInput(
    node_features=node_features,
    edge_index=edge_index,
)

output = model(inputs)
predictions = output.predictions
```

Example configuration:

```yaml
model:
  name: gcn
  input_dim: 8
  output_dim: 3
  hidden_dim: 128
  num_layers: 4
  activation: gelu
  dropout: 0.0
  normalization: layer_norm
  residual: true
  add_self_loops: true
```

The GCN inserts self-loops explicitly when enabled and preserves the user-provided edge direction unless the caller supplies a bidirectional edge list. This keeps the graph topology intentional and avoids surprising implicit rewrites.
