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

from tools.mcp_advisor import guardrails, problem_schema, results
from tools.mcp_advisor.advisor import Advisor

advisor = Advisor()
server = _Server(
    "neural3d-advisor",
    instructions=(
        "Decision support for Neural3DBench 3D architectures. Models are discovered live from "
        "models.ModelRegistry, docs/ and tools/mcp_advisor/catalog.yaml. Representations: point_cloud, "
        "mesh_graph, voxel_grid, continuous_coords. Tasks: surrogate_simulation, pde_operator, "
        "segmentation_classification, field_fitting, shape_completion, object_detection, registration. "
        "Instance segmentation, generation and scene flow are not covered: recommend_architecture then returns "
        "verdict no_fit with networks to add; say so instead of proposing a model. "
        "To define a problem, call get_problem_questions for the task, ask the user, then "
        "compose_problem_description and pass its description to recommend_architecture. "
        "Memory and time figures are heuristics. "
        "Before training, run audit_problem_assumptions and check_symmetry_and_invariance. "
        "get_benchmark_results returns measured metrics of past training runs; cite them when available."
    ),
)


@server.tool()
def list_supported_models(representation: str | None = None, task: str | None = None) -> list[dict]:
    """List registered and catalogued models, optionally filtered.

    representation: point_cloud | mesh_graph | voxel_grid | continuous_coords
    task: surrogate_simulation | pde_operator | segmentation_classification | field_fitting
          | shape_completion | object_detection | registration
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
    required_output: str | None = None,
) -> dict:
    """Rank architectures for a problem with rationale, pitfalls and paper evidence.

    input_format: point_cloud | mesh_graph | voxel_grid | continuous_coords
    max_elements_per_sample: points, nodes or voxels per sample.
    target_metric: accuracy | speed
    required_output: shape_label | point_label | point_field | point_set | boxes | rigid_pose | deformation_field
                     (compose_problem_description returns it); models whose heads cannot give it rank last.
    Returns ``verdict``: fit | weak_fit | no_fit. On no_fit, ``missing_capability``, ``reason`` and
    ``suggested_networks`` name what to add; the ranked list is then only the closest registered models.
    """
    return advisor.recommend_architecture(
        problem_description,
        input_format,
        max_elements_per_sample,
        dynamic_topology=dynamic_topology,
        target_metric=target_metric,
        required_output=required_output,
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
def audit_problem_assumptions(
    problem_description: str,
    model_name: str | None = None,
    boundary_conditions: str = "none",
    topology_changes: bool = False,
    geometry_varies: bool = False,
    evaluate_at_unseen_resolution: bool = False,
) -> dict:
    """Flag unhandled boundary conditions, topology/geometry changes and resolution transfer.

    boundary_conditions: none | dirichlet | neumann | mixed
    Without model_name every model is audited and the error-free ones are listed.
    """
    return guardrails.audit_problem_assumptions(
        advisor, problem_description, model_name, boundary_conditions,
        topology_changes, geometry_varies, evaluate_at_unseen_resolution,
    )


@server.tool()
def check_symmetry_and_invariance(
    model_name: str,
    required_symmetries: list[str],
    target_kind: str = "scalar",
    uses_absolute_coordinates: bool = True,
) -> dict:
    """Compare required symmetries with those the architecture guarantees.

    required_symmetries: subset of permutation | translation | rotation
    target_kind: scalar | vector | tensor (vectors/tensors need equivariance, not invariance)
    """
    return guardrails.check_symmetry_and_invariance(
        advisor, model_name, required_symmetries, target_kind, uses_absolute_coordinates
    )


@server.tool()
def recommend_representation_pipeline(
    raw_format: str, target_model: str, watertight: bool | None = None, voxel_resolution: int = 64
) -> dict:
    """Conversion steps from a raw format (step, stl, msh, xyz, npy, nurbs, ...) to the model's ModelInput."""
    return guardrails.recommend_representation_pipeline(
        advisor, raw_format, target_model, watertight, voxel_resolution
    )


@server.tool()
def calculate_effective_receptive_field(
    model_name: str,
    depth: int | None = None,
    element_spacing: float | None = None,
    interaction_length: float | None = None,
) -> dict:
    """Reach of one forward pass vs the physical coupling length.

    depth: layers/steps (default from the model config); element_spacing: mean edge length or voxel
    size; interaction_length: distance the physics couples over, same units.
    """
    return guardrails.calculate_effective_receptive_field(
        advisor, model_name, depth, element_spacing, interaction_length
    )


@server.tool()
def recommend_batching_strategy(
    model_name: str,
    num_elements: int,
    vram_budget_gb: float,
    target_effective_batch: int = 16,
    variable_size: bool = True,
) -> dict:
    """Micro-batch, gradient accumulation, precision and collation (graph block-diagonal vs point padding)."""
    return guardrails.recommend_batching_strategy(
        advisor, model_name, num_elements, vram_budget_gb, target_effective_batch, variable_size
    )


@server.tool()
def estimate_training_budget(
    model_name: str,
    num_elements: int,
    dataset_size: int,
    epochs: int,
    target_gpu: str = "A100",
    precision: str = "fp32",
) -> dict:
    """Rough wall-clock training time. target_gpu: H100 | A100 | RTX4090 | V100 | T4"""
    return guardrails.estimate_training_budget(
        advisor, model_name, num_elements, dataset_size, epochs, target_gpu, precision
    )


@server.tool()
def generate_benchmark_config(
    model_name: str,
    dataset_name: str,
    num_elements: int,
    hardware_target: str = "A100",
    dataset_size: int = 1000,
    epochs: int = 100,
) -> dict:
    """YAML config and Python launcher (ModelRegistry.create) with batch, precision and optimizer filled in."""
    return guardrails.generate_benchmark_config(
        advisor, model_name, dataset_name, num_elements, hardware_target, dataset_size, epochs
    )


@server.tool()
def recommend_training_setup(
    model_name: str,
    num_elements: int,
    vram_gb: float = 8.0,
    required_symmetries: list[str] | None = None,
    target_kind: str = "scalar",
    max_epochs: int = 300,
) -> dict:
    """Optimizer, learning rate, LR schedule, epochs, batch, augmentation and model config from the model's
    paper recipe (cited), adapted to the problem's VRAM and required symmetries. Returns the setup, a reason
    per setting, and warnings for what the paper does not state or the implementation does differently."""
    return guardrails.recommend_training_setup(
        advisor, model_name, num_elements, vram_gb, required_symmetries, target_kind, max_epochs
    )


@server.tool()
def get_problem_questions(task: str) -> dict:
    """Application domains and follow-up questions that matter for a task, to define the problem precisely.

    task: surrogate_simulation | pde_operator | segmentation_classification | field_fitting
          | shape_completion | object_detection | registration
    Each question has a key, label, kind (choice | flag) and, for choices, option labels.
    Ask the user these, then pass the answers to compose_problem_description.
    """
    return problem_schema.problem_questions(task)


@server.tool()
def compose_problem_description(
    task: str, answers: dict | None = None, domains: list[str] | None = None, extra: str = ""
) -> dict:
    """Turn answers from get_problem_questions into recommend_architecture and pre-flight inputs.

    answers: question key -> option label (choice) or true/false (flag).
    Returns description (for problem_description), dynamic_topology, and assumptions
    (target_kind, geometry_varies, topology_changes, evaluate_at_unseen_resolution, add_symmetries).
    """
    return problem_schema.compose_problem(task, answers, domains, extra)


@server.tool()
def suggest_ablation_matrix(primary_model: str, max_contenders: int = 4) -> dict:
    """Baseline contenders that isolate the primary model's architectural contribution."""
    return guardrails.suggest_ablation_matrix(advisor, primary_model, max_contenders)


@server.tool()
def get_benchmark_results(
    model_name: str | None = None, dataset: str | None = None, tracking_uri: str = "./mlruns"
) -> dict:
    """Measured results of dashboard training runs (MLflow): final validation metrics per run and the
    best model per dataset by its task's primary metric (mIoU, accuracy, MAE or Chamfer).

    Prefer these numbers over heuristics when the user asks which model works best on a dataset.
    tracking_uri: MLflow store; a relative path resolves from the server's working directory.
    """
    return results.benchmark_results(tracking_uri, model_name, dataset)


@server.tool()
def refresh_model_catalog() -> dict:
    """Re-scan ModelRegistry, docs/ and catalog.yaml without restarting the server."""
    return advisor.refresh()


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
