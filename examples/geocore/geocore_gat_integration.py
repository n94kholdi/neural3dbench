"""End-to-end geocore -> ModelInput -> GAT smoke test.

Run from the workspace root with:

    .venv/bin/python neural3dbench/examples/geocore/geocore_gat_integration.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
NEURAL3DBENCH_ROOT = WORKSPACE_ROOT / "neural3dbench"
GEOCORE_ROOT = WORKSPACE_ROOT / "geometry-py" / "geocore"
for source_root in (NEURAL3DBENCH_ROOT, GEOCORE_ROOT):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import geocore as gc  # noqa: E402
from models import GATConfig, GraphRepresentationAdapter, create_model  # noqa: E402


METADATA_KEYS = (
    "schema_version",
    "representation",
    "coordinate_system",
    "units",
    "normalization_id",
    "node_feature_names",
    "edge_feature_names",
)


def make_mesh_graph(nx: int = 8, ny: int = 6) -> dict[str, np.ndarray]:
    """Create a normalized, bidirectional geocore grid graph for node fields."""
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must both be at least 2")

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    vertices = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(nx * ny)))
    faces = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            lower_left = row * nx + column
            lower_right = lower_left + 1
            upper_left = lower_left + nx
            upper_right = upper_left + 1
            faces.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )

    mesh = gc.Mesh(vertices, np.asarray(faces, dtype=np.int64))
    graph = dict(mesh.output("MESH_GRAPH", normalization="unit_cube"))
    graph["node_features"] = graph["coordinates"].astype(np.float32, copy=True)
    graph["node_feature_names"] = np.asarray(["x", "y", "z"], dtype=np.str_)
    graph.pop("faces", None)
    gc.validate(graph)
    return graph


def metadata_from(raw: dict) -> dict:
    metadata = {}
    for key in METADATA_KEYS:
        if key not in raw:
            continue
        value = np.asarray(raw[key])
        metadata[key] = value.item() if value.ndim == 0 else value.tolist()
    return metadata


def adapt_graph(raw: dict):
    """Use the shared graph adapter without changing geocore edge ordering."""
    return GraphRepresentationAdapter().adapt(
        {**raw, "metadata": metadata_from(raw)},
        batch=raw.get("batch"),
    )


def assert_bidirectional(edge_index: np.ndarray) -> None:
    directed = set(map(tuple, np.asarray(edge_index).T.tolist()))
    assert all(
        (receiver, sender) in directed
        for sender, receiver in directed
        if sender != receiver
    ), "MESH_GRAPH must contain both directions for every mesh edge"


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = make_mesh_graph()
    graph_path = Path(gc.save(str(output_dir / "gat_mesh_graph.npz"), graph))

    raw = gc.load(str(graph_path))
    assert str(np.asarray(raw["representation"])) == "MESH_GRAPH"
    assert_bidirectional(raw["edge_index"])
    model_input = adapt_graph(raw)

    node_count = model_input.node_features.shape[0]
    torch.manual_seed(0)
    model = create_model(
        GATConfig(
            input_dim=model_input.node_features.shape[1],
            output_dim=1,
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            attention_dropout=0.0,
        )
    )
    output = model(model_input)
    assert output.predictions.shape == (node_count, 1)
    assert torch.isfinite(output.predictions).all()
    output.predictions.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    print("PASS: geocore and GAT work together")
    print(f"  interchange file: {graph_path}")
    print(f"  nodes / directed edges: {node_count} / {model_input.edge_index.shape[1]}")
    print(f"  node features: {metadata_from(raw)['node_feature_names']}")
    print(f"  prediction shape: {tuple(output.predictions.shape)}")
    print("  forward and backward passes: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        run(args.output_dir.resolve())
        return
    with tempfile.TemporaryDirectory(prefix="geocore_gat_") as temp_dir:
        run(Path(temp_dir))


if __name__ == "__main__":
    main()