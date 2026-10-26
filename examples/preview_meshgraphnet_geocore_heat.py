"""Render the geocore mesh graph and MeshGraphNet heat-rollout results.

Example:

    python preview_meshgraphnet_geocore_heat.py \
        --output-dir ../../outputs/meshgraphnet_geocore_heat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402


def render_mesh_graph(graph: dict[str, np.ndarray], path: Path) -> None:
    coordinates = graph["coordinates"]
    xy = coordinates[:, :2]
    edge_index = graph["edge_index"]
    unique_edges = edge_index[:, edge_index[0] < edge_index[1]].T
    segments = xy[unique_edges]

    names = graph["node_feature_names"].tolist()
    features = graph["node_features"]
    hot = features[:, names.index("hot_boundary")].astype(bool)
    cold = features[:, names.index("cold_boundary")].astype(bool)
    interior = ~(hot | cold)

    figure, axis = plt.subplots(figsize=(10, 5), dpi=160)
    axis.add_collection(LineCollection(segments, color="#596579", linewidth=0.7, alpha=0.8))
    axis.scatter(*xy[interior].T, s=14, color="#e7edf5", edgecolor="#344054", linewidth=0.4, label="interior")
    axis.scatter(*xy[hot].T, s=32, color="#e53935", edgecolor="white", linewidth=0.6, label="hot boundary: T=1")
    axis.scatter(*xy[cold].T, s=32, color="#1e88e5", edgecolor="white", linewidth=0.6, label="cold boundary: T=0")
    axis.autoscale()
    axis.set_aspect("equal")
    axis.set_xlabel("normalized x")
    axis.set_ylabel("normalized y")
    axis.set_title(
        f"geocore MESH_GRAPH — {len(coordinates)} nodes, "
        f"{edge_index.shape[1]} directed edges"
    )
    axis.legend(loc="upper center", ncol=3, frameon=False)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def render_heat_rollout(result: dict[str, np.ndarray], path: Path) -> None:
    xy = result["coordinates"][:, :2]
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1])
    initial = result["initial_temperature"]
    reference = result["reference_temperature"]
    predicted = result["predicted_temperature"]
    error = np.abs(predicted - reference)

    fields = (
        ("Initial temperature", initial, "inferno", 0.0, 1.0),
        ("Reference after 12 steps", reference, "inferno", 0.0, 1.0),
        ("MeshGraphNet after 12 steps", predicted, "inferno", 0.0, 1.0),
        ("Absolute error", error, "magma", 0.0, max(float(error.max()), 1e-6)),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=160, constrained_layout=True)
    for axis, (title, values, cmap, vmin, vmax) in zip(axes.flat, fields):
        image = axis.tripcolor(
            triangulation,
            values,
            shading="gouraud",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.triplot(triangulation, color="white", linewidth=0.15, alpha=0.35)
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.set_xlabel("normalized x")
        axis.set_ylabel("normalized y")
        figure.colorbar(image, ax=axis, shrink=0.85)

    mse = float(np.mean((predicted - reference) ** 2))
    mae = float(error.mean())
    figure.suptitle(
        f"Heat diffusion rollout — MSE={mse:.6f}, MAE={mae:.6f}",
        fontsize=14,
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/meshgraphnet_geocore_heat"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    graph_path = output_dir / "mesh_graph.npz"
    result_path = output_dir / "rollout_result.npz"
    if not graph_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(
            f"Expected {graph_path} and {result_path}; run training first."
        )

    mesh_preview = output_dir / "mesh_graph_preview.png"
    rollout_preview = output_dir / "meshgraphnet_heat_preview.png"
    render_mesh_graph(load_npz(graph_path), mesh_preview)
    render_heat_rollout(load_npz(result_path), rollout_preview)
    print(f"Mesh graph preview: {mesh_preview}")
    print(f"MeshGraphNet preview: {rollout_preview}")


if __name__ == "__main__":
    main()
