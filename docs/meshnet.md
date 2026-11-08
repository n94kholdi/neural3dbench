# MeshNet

Face-based shape classifier for triangle meshes (Feng et al., AAAI 2019). Each face is one input
unit, so the mesh does not need to be manifold or watertight.

1. **Spatial descriptor.** An MLP on each face center gives 64 channels.
2. **Structural descriptor.**
   - *Face rotate convolution*: a shared MLP over the three corner-vector pairs, averaged. The
     result does not depend on which corner comes first.
   - *Face kernel correlation*: `num_kernel` learned kernels, each made of 4 unit normals. They are
     compared with the face normal and its 3 neighbours' normals through a Gaussian of width `sigma`.
   - Both are concatenated with the normal and projected to 131 channels.
3. **Mesh convolution** (two layers, 64/131 → 256/256 → 512/512). Spatial and structural features
   are combined. Structural features are aggregated with a max over the 3 faces that share an edge.
4. **Fusion and pooling.** The multi-level features are fused to 1024 channels. A max over the
   faces of each mesh feeds a 512-256-`output_dim` classifier with dropout.

Boundary edges have no neighbour face, so the face uses itself there. On a non-manifold edge,
faces are paired in sorted order.

## Input/output contract

Meshes are concatenated like graphs. `models.collate_meshes([(vertices, faces), ...])` builds the input.

| Field | Shape | Notes |
|---|---|---|
| `coordinates` | `[V, 3]` float | Vertices of all meshes in the batch. |
| `faces` | `[F, 3]` int | Triangles, indices into `coordinates`. |
| `batch` | `[V]` int | Sorted mesh id per vertex; optional for one mesh. |

| Output | Shape | Notes |
|---|---|---|
| `predictions` | `[B, output_dim]` | Class logits. |
| `latent` | `[B, 1024]` | Pooled mesh feature. |
| `metadata["face_batch"]` | `[F]` | Mesh id per face. |

## Usage

```python
from models import collate_meshes, create_model

model = create_model({"name": "meshnet", "output_dim": 10})
logits = model(collate_meshes([(vertices_a, faces_a), (vertices_b, faces_b)])).predictions  # [2, 10]
```

Train on ModelNet10 meshes (official split, decimated to 2048 faces):

```bash
python examples/train_mesh_classifier.py --model meshnet --epochs 30
```

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `output_dim` | 10 | Number of classes. |
| `num_kernel` | 64 | Face kernel correlation kernels. |
| `sigma` | 0.2 | Gaussian width of the kernel correlation. |
| `dropout` | 0.5 | Classifier dropout. |

## Limits

- Classification only; there is no per-face segmentation head.
- Not rotation invariant: it uses absolute face centers and normals.
- Activations grow by about 19k floats per face in training. The paper simplifies meshes to 1024
  faces; the training script caps them at `--max-faces` (2048 by default).
- BatchNorm needs at least 2 meshes per training batch.

## References

- Y. Feng, Y. Feng, H. You, X. Zhao, Y. Gao. *MeshNet: Mesh Neural Network for 3D Shape Representation.* AAAI 2019. https://arxiv.org/abs/1811.11424
- Reference code: https://github.com/iMoonLab/MeshNet
