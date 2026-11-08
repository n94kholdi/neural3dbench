# MeshWalker

Shape classifier that reads random walks over the mesh (Lahav and Tal, ACM TOG 2020). It needs
only vertices and edges, so the mesh does not need to be manifold.

1. **Walks.** Each walk starts at a random vertex and steps to a random unvisited edge neighbour.
   When it has none, it jumps to a random unvisited vertex of the same mesh. When every vertex has
   been visited, revisits are allowed. Walks are generated on the CPU in every forward pass.
2. **Step embedding.** The xyz displacement of every step (`walk_length - 1` steps) passes through
   FC layers `fc_dims`, each with instance normalization and ReLU.
3. **Stacked GRUs** of widths `gru_dims` read the sequence. The last step's state gives the logits.
4. **Ensembling.** Logits are averaged over `num_walks` walks per mesh in training and
   `num_walks_eval` walks in eval mode.

## Input/output contract

| Field | Shape | Notes |
|---|---|---|
| `coordinates` | `[V, 3]` float | Vertices of all meshes, concatenated (`collate_meshes`). |
| `faces` | `[F, 3]` int | Triangles; only their edges are used. |
| `batch` | `[V]` int | Sorted mesh id per vertex; optional for one mesh. |

| Output | Shape | Notes |
|---|---|---|
| `predictions` | `[B, output_dim]` | Class logits averaged over walks. |
| `auxiliary["walks"]` | `[B * walks, walk_length]` | Global vertex ids of each walk, mesh-major. |

## Usage

```python
from models import collate_meshes, create_model

model = create_model({"name": "meshwalker", "output_dim": 10})
logits = model(collate_meshes(meshes)).predictions  # [B, 10]
```

Walks follow `torch.manual_seed`. The default GRUs (1024, 1024, 512) are slow on a CPU. For a
quicker run, use smaller ones:

```bash
python examples/train_mesh_classifier.py --model meshwalker --config '{"gru_dims": [256, 256]}'
```

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `output_dim` | 10 | Number of classes. |
| `walk_length` | 200 | Vertices per walk. |
| `num_walks` | 1 | Walks per mesh in training. |
| `num_walks_eval` | 8 | Walks per mesh in eval mode (the paper uses up to 32). |
| `fc_dims` | (128, 256) | Step embedding widths. |
| `gru_dims` | (1024, 1024, 512) | Stacked GRU widths. |
| `dropout` | 0.0 | Dropout before each GRU. |

## Limits

- Predictions are stochastic; raise `num_walks_eval` for stable evaluation.
- Not rotation invariant. The paper trains with random rotations; the training script does not
  add augmentation.
- The paper tunes walk length per dataset (relative to vertex count). Here it is a fixed config value.
- Jumps produce large displacement steps, as in the paper.

## References

- A. Lahav, A. Tal. *MeshWalker: Deep Mesh Understanding by Random Walks.* ACM TOG (SIGGRAPH Asia) 2020. https://arxiv.org/abs/2006.05353
- Reference code: https://github.com/AlonLahav/MeshWalker
