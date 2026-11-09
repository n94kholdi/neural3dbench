# GAT

Graph Attention Network (Velickovic et al., ICLR 2018). Each layer weights neighbours with
learned attention coefficients (softmax over the neighbourhood), with several heads that are
concatenated or averaged. This makes aggregation anisotropic, unlike GCN.

## Input/output contract

| Field | Shape | Notes |
| --- | --- | --- |
| `node_features` | `[N, input_dim]` | Required. |
| `edge_index` | `[2, E]` | Required, directed edges. |
| `batch` | `[N]` | Optional sample id per node. |
| `predictions` | `[N, output_dim]` | Per-node outputs; no graph-level head. |

```python
from models import ModelInput, ModelRegistry

model = ModelRegistry.create("gat", {"input_dim": 3, "output_dim": 2, "num_heads": 4})
logits = model(ModelInput(node_features=x, edge_index=edges)).predictions  # [N, 2]
```

Set `return_attention_weights=True` to inspect the attention per edge.

## Limits

- Attention uses node features only; geometric edge features (relative positions, lengths) are
  not built in. For physics on meshes, MeshGraphNet is the stronger choice.
- Receptive field of `num_layers` hops, as in GCN.
- The paper evaluates node classification on citation and protein graphs (Cora 83.0% accuracy),
  not meshes.

## References

- [Graph Attention Networks (arXiv 1710.10903)](https://arxiv.org/abs/1710.10903)
