"""End-to-end geocore -> ModelInput -> MeshGraphNet smoke test.

Run from any directory with:

    python neural3dbench/examples/geocore_meshgraphnet_integration.py

The model is randomly initialized, so this example verifies interoperability,
tensor shapes, inference, and gradients rather than prediction accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


# Keep this example runnable from a source checkout without installing either
# sibling package. Installed packages continue to work because these are their
# source roots.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
NEURAL3DBENCH_ROOT = WORKSPACE_ROOT / "neural3dbench"
GEOCORE_ROOT = WORKSPACE_ROOT / "geometry-py" / "geocore"
for source_root in (NEURAL3DBENCH_ROOT, GEOCORE_ROOT):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import geocore as gc  # noqa: E402
from models import (  # noqa: E402
    GraphRepresentationAdapter,
    MeshGraphNetConfig,
    create_model,
)


METADATA_KEYS = (
    "schema_version",
    "representation",
    "coordinate_system",
    "units",
    "normalization_id",
    "node_feature_names",
    "edge_feature_names",
)


def tetrahedron() -> gc.Mesh:
    """Return a small closed triangle mesh with outward-facing triangles."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )
    return gc.Mesh(vertices, faces)


def scalar_or_list(value: np.ndarray):
    """Convert NPZ metadata arrays to ordinary Python values."""
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Geometry Core creates and validates the canonical ModelInput.md data.
    geometry = tetrahedron()
    graph = geometry.output("MESH_GRAPH")
    assert graph.validate() == "MESH_GRAPH"
    npz_path = Path(graph.save(str(output_dir) + "/"))

    # 2. Reload the actual interchange file, not the in-memory builder result.
    raw = gc.load(str(npz_path))
    assert str(np.asarray(raw["representation"])) == "MESH_GRAPH"

    # 3. The adapter maps external MESH_GRAPH data to neural3dbench's internal
    # GRAPH representation. Keep schema and feature-name information as model
    # metadata, and preserve the optional graph batch vector explicitly.
    metadata = {
        key: scalar_or_list(raw[key])
        for key in METADATA_KEYS
        if key in raw
    }
    model_input = GraphRepresentationAdapter().adapt(
        {**raw, "metadata": metadata},
        batch=raw.get("batch"),
    )

    node_count, node_input_dim = model_input.node_features.shape
    edge_count = model_input.edge_index.shape[1]
    edge_input_dim = model_input.edge_features.shape[1]

    # Feature widths come from the file, preventing configuration drift when
    # geocore feature selections change.
    config = MeshGraphNetConfig(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        output_dim=3,
        latent_dim=16,
        processor_steps=2,
        mlp_hidden_dim=32,
        mlp_layers=1,
    )
    torch.manual_seed(0)
    model = create_model(config)

    # 4. Exercise validation, inference, output metadata, and autograd.
    output = model(model_input)
    assert output.predictions.shape == (node_count, config.output_dim)
    assert output.latent.shape == (node_count, config.latent_dim)
    assert output.auxiliary["edge_latent"].shape == (
        edge_count,
        config.latent_dim,
    )
    assert torch.isfinite(output.predictions).all()
    assert torch.equal(output.metadata["batch"], model_input.batch)

    output.predictions.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    print("PASS: geocore and MeshGraphNet work together")
    print(f"  interchange file: {npz_path}")
    print(f"  nodes / directed edges: {node_count} / {edge_count}")
    print(f"  node features: {metadata['node_feature_names']}")
    print(f"  edge features: {metadata['edge_feature_names']}")
    print(f"  prediction shape: {tuple(output.predictions.shape)}")
    print("  forward and backward passes: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep mesh_graph.npz in this directory (default: temporary directory).",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        run(args.output_dir.resolve())
        return

    with tempfile.TemporaryDirectory(prefix="geocore_meshgraphnet_") as temp_dir:
        run(Path(temp_dir))


if __name__ == "__main__":
    main()
