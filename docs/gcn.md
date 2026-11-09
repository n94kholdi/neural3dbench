# GCN

Graph Convolutional Network (Kipf and Welling, ICLR 2017). Each layer averages neighbour
features with the symmetric normalisation `D^-1/2 (A + I) D^-1/2` and applies a shared linear map.
This repository uses it as a simple, isotropic baseline on mesh graphs.

## Input/output contract

| Field | Shape | Notes |
| --- | --- | --- |
| `node_features` | `[N, input_dim]` | Required. For meshes, e.g. vertex positions and boundary flags. |
| `edge_index` | `[2, E]` | Required, directed edges; add both directions for an undirected mesh. |
| `batch` | `[N]` | Optional sample id per node for block-diagonal batches. |
| `predictions` | `[N, output_dim]` | One value or logit vector per node. There is no graph-level head. |

```python
from models import ModelInput, ModelRegistry

model = ModelRegistry.create("gcn", {"input_dim": 3, "output_dim": 1})
field = model(ModelInput(node_features=x, edge_index=edges)).predictions  # [N, 1]
```

## Limits

- Isotropic: every neighbour gets the same weight, so directional effects (flow, anisotropic
  stress) are smoothed out. MeshGraphNet uses edge features and is the better surrogate model.
- The receptive field is `num_layers` hops. On fine meshes the hop count covers a small physical
  distance, and deep GCNs over-smooth.
- The paper evaluates node classification on citation graphs (Cora 81.5% accuracy), not meshes,
  so no 3D results exist to compare with.

## References

- [Semi-Supervised Classification with Graph Convolutional Networks (arXiv 1609.02907)](https://arxiv.org/abs/1609.02907)
