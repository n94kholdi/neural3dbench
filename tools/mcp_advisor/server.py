"""neural3d-advisor MCP server (stdio).

Run from the repository root with ``python -m tools.mcp_advisor`` or by path with
``python /abs/path/to/tools/mcp_advisor/server.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x

from tools.mcp_advisor.advisor import Advisor

advisor = Advisor()
server = _Server(
    "neural3d-advisor",
    instructions=(
        "Decision support for Neural3DBench 3D architectures. Models are discovered live from "
        "models.ModelRegistry, docs/ and tools/mcp_advisor/catalog.yaml. Representations: point_cloud, "
        "mesh_graph, voxel_grid, continuous_coords. Tasks: surrogate_simulation, pde_operator, "
        "segmentation_classification, field_fitting. Memory figures are heuristics."
    ),
)


@server.tool()
def list_supported_models(representation: str | None = None, task: str | None = None) -> list[dict]:
    """List registered and catalogued models, optionally filtered.

    representation: point_cloud | mesh_graph | voxel_grid | continuous_coords
    task: surrogate_simulation | pde_operator | segmentation_classification | field_fitting
    """
    return advisor.list_supported_models(representation=representation, task=task)


@server.tool()
def get_model_dossier(model_name: str) -> dict:
    """Inductive biases, merits, demerits, tensor contracts, complexity, docs and examples for one model."""
    return advisor.get_model_dossier(model_name)


@server.tool()
def recommend_architecture(
    problem_description: str,
    input_format: str,
    max_elements_per_sample: int,
    dynamic_topology: bool = False,
    target_metric: str = "accuracy",
) -> dict:
    """Rank architectures for a problem with rationale and pitfalls.

    input_format: point_cloud | mesh_graph | voxel_grid | continuous_coords
    max_elements_per_sample: points, nodes or voxels per sample.
    target_metric: accuracy | speed
    """
    return advisor.recommend_architecture(
        problem_description,
        input_format,
        max_elements_per_sample,
        dynamic_topology=dynamic_topology,
        target_metric=target_metric,
    )


@server.tool()
def estimate_resource_footprint(
    model_name: str, num_elements: int, batch_size: int = 1, precision: str = "fp32"
) -> dict:
    """Heuristic GPU memory (training/inference) and hardware tier for a workload.

    precision: fp32 | fp16 | bf16 | amp
    """
    return advisor.estimate_resource_footprint(model_name, num_elements, batch_size, precision)


@server.tool()
def refresh_model_catalog() -> dict:
    """Re-scan ModelRegistry, docs/ and catalog.yaml without restarting the server."""
    return advisor.refresh()


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
