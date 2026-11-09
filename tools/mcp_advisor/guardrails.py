"""Problem-formulation, data-pipeline and training guardrails built on ``Advisor``.

Every function is rule based: it reads catalog.yaml metadata (symmetries, receptive_field,
resolution_transfer, boundary_conditions, memory_model) and the live registry defaults, and
names only converters that exist in this repository. Numbers are heuristics.
"""

from __future__ import annotations

import math
from typing import Any

import yaml

from tools.mcp_advisor.advisor import GB, Advisor

SYMMETRIES = ("permutation", "translation", "rotation")
TARGET_KINDS = ("scalar", "vector", "tensor")
BOUNDARY_CONDITIONS = ("none", "dirichlet", "neumann", "mixed")

# name: (VRAM GB, memory bandwidth bytes/s, native bf16)
GPUS = {
    "H100": (80, 3.35e12, True),
    "A100": (80, 2.0e12, True),
    "RTX4090": (24, 1.0e12, True),
    "V100": (32, 0.9e12, False),
    "T4": (16, 0.32e12, False),
}
# Training step is treated as memory-bound: activations are written in forward, read and
# written again in backward (~3 passes). The low efficiency also absorbs compute-bound conv/MLP
# layers and kernel-launch overhead. shortcut: not calibrated on real GPUs; fit it to timed pilot runs.
ACTIVATION_PASSES = 3
BANDWIDTH_EFFICIENCY = 0.15

ADAPTERS = {
    "point_cloud": "models.representations.PointCloudRepresentationAdapter",
    "mesh_graph": "models.representations.GraphRepresentationAdapter",
    "voxel_grid": "models.representations.GridRepresentationAdapter",
    "continuous_coords": "models.representations.CoordinateRepresentationAdapter",
}
RAW_FORMATS = {
    "cad": ("step", "stp", "iges", "igs", "brep", "cad"),
    "surface_mesh": ("stl", "obj", "ply", "off", "surface_mesh"),
    "volume_mesh": ("msh", "vtk", "vtu", "inp", "cgns", "volume_mesh"),
    "point_cloud": ("xyz", "pts", "pcd", "las", "point_cloud"),
    "voxel_grid": ("npy", "nii", "vdb", "voxel", "voxel_grid"),
    "nurbs": ("nurbs",),
}


def _defaults(model: dict[str, Any]) -> dict[str, Any]:
    return (model["registry"] or {}).get("config_defaults", {})


def _per_sample_gb(advisor: Advisor, name: str, num_elements: int, precision: str) -> tuple[float, float]:
    """(fixed GB, GB per extra sample) of the training peak, from two footprint evaluations."""
    one = advisor.estimate_resource_footprint(name, num_elements, 1, precision)["memory_gb"]["training_peak"]
    two = advisor.estimate_resource_footprint(name, num_elements, 2, precision)["memory_gb"]["training_peak"]
    return one - (two - one), two - one


def _gpu(name: str) -> tuple[str, tuple[float, float, bool]]:
    key = name.upper().replace(" ", "").replace("-", "")
    if key not in GPUS:
        raise ValueError(f"Unknown GPU '{name}'. Use one of {sorted(GPUS)}.")
    return key, GPUS[key]


# ---------------------------------------------------------------- problem formulation


def audit_problem_assumptions(
    advisor: Advisor,
    problem_description: str,
    model_name: str | None = None,
    boundary_conditions: str = "none",
    topology_changes: bool = False,
    geometry_varies: bool = False,
    evaluate_at_unseen_resolution: bool = False,
) -> dict[str, Any]:
    """Check boundary conditions, topology changes, geometry variation and resolution transfer."""
    bc = boundary_conditions.lower()
    if bc not in BOUNDARY_CONDITIONS:
        raise ValueError(f"boundary_conditions must be one of {BOUNDARY_CONDITIONS}.")
    text = problem_description.lower()
    names = [advisor.resolve(model_name)["name"]] if model_name else list(advisor.state["models"])

    audits = {}
    for name in names:
        meta = advisor.state["models"][name]["metadata"]
        findings: list[dict[str, str]] = []

        def add(severity: str, check: str, message: str) -> None:
            findings.append({"severity": severity, "check": check, "message": message})

        if bc != "none":
            hard = meta.get("boundary_conditions") == "hard_dirichlet"
            if hard and bc == "dirichlet":
                add("ok", "boundary_conditions", "Dirichlet BCs are satisfied exactly by the ansatz.")
            elif hard:
                add("warning", "boundary_conditions",
                    f"Only Dirichlet BCs are exact; {bc} (Neumann) parts must be penalised in the loss.")
            else:
                add("warning", "boundary_conditions",
                    "BCs are not enforced by the architecture. Feed them as inputs (boundary_mask for grids, "
                    "node-type/boundary_tags features for graphs) or penalise them in the loss, and report "
                    "boundary error separately - a low global MSE can hide BC violations.")

        dyn = meta["dynamic_topology"]
        if topology_changes and dyn is False:
            add("error", "topology", "Assumes one fixed geometry/topology; it cannot follow remeshing, "
                "cracks or changing vertex counts.")
        elif topology_changes and dyn == "partial":
            add("warning", "topology", "Topology changes force re-voxelisation at a fixed resolution every step.")

        if geometry_varies and dyn is False:
            add("error", "geometry", "Fitted per geometry (per-instance); it does not generalise across shapes "
                "without conditioning. Use an operator/graph model for families of geometries.")

        transfer = meta.get("resolution_transfer")
        if evaluate_at_unseen_resolution and transfer is False:
            add("error", "resolution", "Weights are tied to the training discretisation (hop count or kernel "
                "size in elements); predictions change when mesh/grid resolution changes.")
        elif evaluate_at_unseen_resolution and transfer == "partial":
            add("warning", "resolution", "Generalises to new meshes, but receptive field in physical units "
                "shifts with element density; validate on the target resolution.")

        if name == "fno3d" and bc != "none" and "periodic" not in text:
            add("warning", "periodicity", "FFT layers assume periodicity; set padding > 0 for non-periodic BCs.")
        if name == "fno3d" and any(w in text for w in ("shock", "discontinu", "crack", "contact")):
            add("warning", "spectral", "Truncated Fourier modes ring (Gibbs) at shocks/discontinuities.")
        if any(w in text for w in ("velocity", "stress", "displacement", "force", "vector", "tensor", "strain")):
            add("info", "symmetry", "Vector/tensor targets: run check_symmetry_and_invariance before training.")

        audits[name] = findings

    passing = sorted(n for n, f in audits.items() if not any(x["severity"] == "error" for x in f))
    return {
        "inputs": {
            "boundary_conditions": bc,
            "topology_changes": topology_changes,
            "geometry_varies": geometry_varies,
            "evaluate_at_unseen_resolution": evaluate_at_unseen_resolution,
        },
        "audits": audits if model_name else {n: f for n, f in audits.items() if f},
        "models_without_errors": passing,
    }


def check_symmetry_and_invariance(
    advisor: Advisor,
    model_name: str,
    required_symmetries: list[str],
    target_kind: str = "scalar",
    uses_absolute_coordinates: bool = True,
) -> dict[str, Any]:
    """Compare the symmetries the physics needs with those the architecture guarantees."""
    target_kind = target_kind.lower()
    if target_kind not in TARGET_KINDS:
        raise ValueError(f"target_kind must be one of {TARGET_KINDS}.")
    required = [s.lower() for s in required_symmetries]
    unknown = set(required) - set(SYMMETRIES)
    if unknown:
        raise ValueError(f"Unknown symmetries {sorted(unknown)}; use {SYMMETRIES}.")

    model = advisor.resolve(model_name)
    meta = model["metadata"]
    has = set(meta.get("symmetries", []))
    grid = "voxel_grid" in meta["representations"]
    verdicts = []
    for sym in required:
        guaranteed = sym in has
        note = ""
        if sym == "permutation":
            note = ("Not applicable: grid ordering is fixed." if grid
                    else "Guaranteed." if guaranteed else "Shuffle element order during training.")
        elif sym == "translation":
            if guaranteed and uses_absolute_coordinates:
                guaranteed = False
                note = ("Broken: absolute coordinates are model inputs. Centre each sample "
                        "(pointcloud.normalize_points) or feed only relative features.")
            elif guaranteed:
                note = "Guaranteed with relative features only."
            else:
                note = "Centre each sample on its centroid and avoid absolute positions as features."
            if model["name"] == "fno3d" and _defaults(model).get("append_coordinates"):
                guaranteed, note = False, "Broken by append_coordinates=True; disable it or accept a fixed frame."
        elif sym == "rotation":
            if target_kind == "scalar":
                note = ("Invariance needed. Use rotation-invariant inputs (distances, norms, angles), a canonical "
                        "frame (PCA - resolve sign ambiguity), or random SO(3) augmentation.")
            else:
                rule = "v' = R v" if target_kind == "vector" else "T' = R T R^T"
                note = (f"Equivariance needed, not invariance: outputs must rotate ({rule}). No model in "
                        "Neural3DBench is SO(3)-equivariant. Options: augment with random rotations applied to "
                        "inputs AND targets, predict in local edge/body frames and rotate back, or use an external "
                        "equivariant network (EGNN, e3nn, SE(3)-Transformer).")
            if model["name"] == "pointnet":
                note += " The T-Net only learns approximate alignment; it is not a guarantee."
        verdicts.append({"symmetry": sym, "guaranteed_by_architecture": guaranteed, "action": note})

    warnings = []
    if "rotation" in required:
        warnings.append("Require rotation symmetry only if the physics has no preferred direction; gravity, "
                        "inflow direction or a loading axis break it and should be inputs instead.")
    if target_kind != "scalar" and "rotation" not in required:
        warnings.append(f"{target_kind} targets are frame-dependent: keep every sample in one consistent "
                        "coordinate frame, or rotations in test data will be extrapolation.")
    return {
        "model": model["name"],
        "architecture_symmetries": sorted(has),
        "target_kind": target_kind,
        "verdicts": verdicts,
        "all_guaranteed": all(v["guaranteed_by_architecture"] for v in verdicts),
        "warnings": warnings,
    }


# ---------------------------------------------------------------- data pipeline


def _raw_category(raw_format: str) -> str:
    key = raw_format.lower().lstrip(".").replace("-", "_").replace(" ", "_")
    for category, names in RAW_FORMATS.items():
        if key in names:
            return category
    accepted = sorted(n for names in RAW_FORMATS.values() for n in names)
    raise ValueError(f"Unknown raw_format '{raw_format}'. Use one of {accepted}.")


def recommend_representation_pipeline(
    advisor: Advisor,
    raw_format: str,
    target_model: str,
    watertight: bool | None = None,
    voxel_resolution: int = 64,
) -> dict[str, Any]:
    """Conversion steps from a raw file format to the target model's ModelInput."""
    source = _raw_category(raw_format)
    model = advisor.resolve(target_model)
    meta = model["metadata"]
    reps = meta["representations"]
    target = source if source in reps else reps[0]
    steps: list[str] = []
    warnings: list[str] = []

    if model["name"] == "deep_nurbs":
        if source in ("cad", "nurbs"):
            steps.append("Extract one NURBS patch (degree, knot_vectors, control_points, weights) from the CAD "
                         "file with an external kernel (OpenCASCADE, build123d) into DeepNURBSConfig.")
        else:
            warnings.append("Deep NURBS needs the NURBS patch itself; a mesh/point/voxel source cannot provide it.")
        steps.append("Sample parametric (u, v) collocation points and wrap them with "
                     "models.representations.ParametricSurfaceRepresentationAdapter.")
        warnings.append("Single 2D parametric patch only; multi-patch CAD is unsupported.")
        return {"source": source, "target_representation": "continuous_coords", "model": model["name"],
                "steps": steps, "warnings": warnings}

    mesh = source
    if source in ("cad", "nurbs"):
        steps.append("Tessellate the CAD with an external mesher (gmsh, OpenCASCADE, build123d); Neural3DBench "
                     "has no CAD reader. Mesh the volume (gmsh -3) if the target field lives inside the solid.")
        mesh = "surface_mesh"

    if mesh in ("surface_mesh", "volume_mesh"):
        steps.append("Load vertices/faces/cells and node fields with meshio or trimesh.")
        if target == "mesh_graph":
            steps.append("2D triangles: mesh.TriangularMesh + mesh.TriangleMeshGraphBuilder(node_feature_fields) "
                         "returns ModelInput(GRAPH) with edge features. 3D: TriangularMesh is 2D only - build a "
                         "bidirectional edge_index [2, E] from face/cell edges and use GraphRepresentationAdapter.")
            warnings.append("Count connected components: disjoint parts exchange no messages. Add radius or "
                            "world edges if physics couples them (e.g. contact).")
        elif target == "point_cloud":
            steps.append("pointcloud.mesh_to_point_cloud(mesh, feature_names=..., num_points=...) - takes "
                         "vertices only, so point density follows mesh refinement (no area-weighted sampling).")
        elif target == "voxel_grid":
            steps.append(f"volumetric.mesh_to_voxel_grid(mesh, resolution={voxel_resolution}, feature_names=...).")
            warnings.append("mesh_to_voxel_grid rasterises vertices only: the result is a sparse shell with no "
                            "inside/outside fill and no SDF. Neural3DBench has no solid or narrow-band SDF "
                            "voxeliser - compute occupancy/SDF externally and pass it as boundary_mask via "
                            "GridRepresentationAdapter.")
            if not watertight:
                warnings.append("Mesh is not known to be watertight: inside/outside and SDF sign are undefined. "
                                "Repair it first, or avoid voxelisation with a point/mesh model (transolver, gino).")
    elif source == "point_cloud" and target == "voxel_grid":
        steps.append(f"volumetric.point_cloud_to_voxel_grid(points, resolution={voxel_resolution}, features=...).")
    elif source == "point_cloud" and target == "mesh_graph":
        steps.append("No connectivity in the source: reconstruct a surface (external) or build a k-NN/radius "
                     "edge_index, then GraphRepresentationAdapter.")
        warnings.append("k-NN edges are not mesh edges; MeshGraphNet edge features lose their meaning across gaps.")
    elif source == "voxel_grid" and target == "point_cloud":
        steps.append("Flatten occupied voxel centres [N, 3] and their fields [N, F]; wrap with "
                     "PointCloudRepresentationAdapter.")
    elif source == "voxel_grid" and target == "mesh_graph":
        steps.append("Build a 6-neighbour grid edge_index over occupied voxels; wrap with GraphRepresentationAdapter.")

    if target == "continuous_coords":
        steps.append("Sample interior and boundary collocation coordinates; wrap with "
                     "CoordinateRepresentationAdapter.")
    if target == "point_cloud":
        steps.append("Normalise coordinates (pointcloud.normalize_points) so absolute radii/kernel sizes in the "
                     "config match the data scale; batch with pointcloud.collate_point_clouds.")
    if target == "voxel_grid":
        est = advisor.estimate_resource_footprint(model["name"], voxel_resolution**3)
        warnings.append(f"{voxel_resolution}^3 = {voxel_resolution**3:,} voxels: ~{est['memory_gb']['training_peak']} "
                        "GB training peak per sample; memory grows 8x per doubling of resolution. Curved "
                        "boundaries are staircased at this resolution.")
    steps.append(f"Final container: {ADAPTERS[target]} -> ModelInput.")
    return {"source": source, "target_representation": target, "model": model["name"],
            "native_representations": reps, "steps": steps, "warnings": warnings}


def calculate_effective_receptive_field(
    advisor: Advisor,
    model_name: str,
    depth: int | None = None,
    element_spacing: float | None = None,
    interaction_length: float | None = None,
) -> dict[str, Any]:
    """Reach of one forward pass, versus the physical length information must travel.

    element_spacing: mean edge length / voxel size in physical units.
    interaction_length: distance over which the physics couples (e.g. body length for a shock).
    """
    model = advisor.resolve(model_name)
    meta = model["metadata"]
    rf = meta.get("receptive_field", {"kind": "unknown"})
    kind = rf["kind"]
    defaults = _defaults(model)
    if depth is None and rf.get("depth_key"):
        depth = defaults.get(rf["depth_key"])

    result: dict[str, Any] = {"model": model["name"], "kind": kind, "depth": depth, "note": rf.get("note", "")}
    reach = None  # in elements
    if kind == "hops":
        reach = depth
        result["reach_elements"] = f"{depth} hops"
    elif kind == "conv_grid":
        k = defaults.get(rf.get("kernel_key", ""), 3)
        levels = depth
        # Two convs per encoder level at stride 2^l, pooling between levels, two convs per decoder level.
        reach = (1 + 2 * (k - 1) * (2**levels - 1) + (2 ** (levels - 1) - 1)
                 + 2 * (k - 1) * (2 ** (levels - 1) - 1))
        result["reach_elements"] = f"{reach} voxels (theoretical; effective field is roughly half)"
        reach = reach / 2
    elif kind == "global":
        result["reach_elements"] = "whole sample in one forward pass"
    elif kind == "pointwise":
        result["reach_elements"] = ("none: queries are independent; spatial coupling comes only from PDE "
                                    "residual losses or the fitted field itself")
    else:
        result["reach_elements"] = "unknown (no receptive_field in catalog.yaml)"

    if reach is not None and element_spacing and interaction_length:
        needed = math.ceil(interaction_length / element_spacing)
        result["reach_physical"] = round(reach * element_spacing, 6)
        result["elements_to_cover_interaction"] = needed
        if reach < needed:
            fix = (f"raise {rf.get('depth_key')} to >= {needed}" if kind == "hops"
                   else "add levels or coarsen the grid")
            result["warning"] = (f"Information travels ~{reach:g} elements but the physics couples over {needed}. "
                                 f"Either {fix}, coarsen/multiscale the mesh, or switch to a global model "
                                 "(transolver, fno3d, gino).")
            if kind == "hops" and needed > 30:
                result["warning"] += " Deep message passing oversmooths and costs memory linear in depth."
        else:
            result["ok"] = "Receptive field covers the interaction length."
    return result


# ---------------------------------------------------------------- training feasibility


def recommend_batching_strategy(
    advisor: Advisor,
    model_name: str,
    num_elements: int,
    vram_budget_gb: float,
    target_effective_batch: int = 16,
    variable_size: bool = True,
) -> dict[str, Any]:
    """Micro-batch size, gradient accumulation, precision and collation for a VRAM budget."""
    if vram_budget_gb <= 0 or target_effective_batch <= 0:
        raise ValueError("vram_budget_gb and target_effective_batch must be positive.")
    model = advisor.resolve(model_name)
    meta = model["metadata"]
    name = model["name"]
    rep = meta["representations"][0]
    pde_points = meta["element"] == "query point"

    options = {}
    for precision in ("fp32", "bf16"):
        fixed, per = _per_sample_gb(advisor, name, num_elements, precision)
        options[precision] = max(0, int((vram_budget_gb - fixed) // per)) if per > 0 else target_effective_batch
    # PDE residuals need fp32; otherwise switch to bf16 only when fp32 cannot fit a useful micro-batch.
    precision = "fp32" if pde_points or options["fp32"] >= min(target_effective_batch, 4) else "bf16"
    micro = min(options[precision], target_effective_batch)

    result: dict[str, Any] = {
        "model": name,
        "max_micro_batch": options,
        "precision": precision,
        "micro_batch": micro,
        "notes": [],
    }
    notes = result["notes"]
    if micro == 0:
        n = num_elements
        while n > 1000 and sum(_per_sample_gb(advisor, name, n, "bf16")) > vram_budget_gb:
            n //= 2
        result["gradient_accumulation_steps"] = None
        notes.append(f"One sample of {num_elements:,} {meta['element']}s does not fit in {vram_budget_gb} GB even "
                     f"in bf16. Sub-sample/crop to ~{n:,} elements, use activation checkpointing, or shard.")
    else:
        result["gradient_accumulation_steps"] = math.ceil(target_effective_batch / micro)

    collate = {
        "mesh_graph": "Disjoint block-diagonal batching: concatenate nodes, offset edge_index by the running "
                      "node count, and set ModelInput.batch to the sample id per node. No padding is needed.",
        "point_cloud": "pointcloud.collate_point_clouds(samples, num_points=...) pads to a common size with "
                       "point_mask.",
        "voxel_grid": "Stack fixed-resolution grids as [B, C, D, H, W]; mark geometry with boundary_mask.",
        "continuous_coords": "Batch collocation points, not samples: draw random chunks of query points per step.",
    }[rep]
    result["collation"] = collate
    if rep == "point_cloud" and variable_size:
        notes.append("Variable sizes: padding wastes memory and compute on masked points. Bucket samples by size "
                     "or resample to a fixed num_points.")
        if meta["memory_model"].get("per_element_sq"):
            notes.append("Padded points still enter the N x N k-NN distance matrix; padding cost is quadratic.")
    if precision == "bf16" or options["bf16"] > options["fp32"]:
        if name == "fno3d":
            notes.append("cuFFT in half precision needs power-of-two grid sizes; keep spectral weights and FFTs "
                         "in fp32 (autocast region excluding SpectralConv3d) if sizes are not powers of two.")
        if pde_points:
            notes.append("Keep fp32 when PDE residuals use second derivatives; bf16 loses the precision they need.")
    if micro and micro < 4 and _defaults(model).get("normalization") == "batch":
        notes.append(f"micro_batch={micro} with BatchNorm gives noisy statistics; switch normalization or freeze "
                     "BN statistics. Gradient accumulation does not fix BN batch statistics.")
    return result


def estimate_training_budget(
    advisor: Advisor,
    model_name: str,
    num_elements: int,
    dataset_size: int,
    epochs: int,
    target_gpu: str = "A100",
    precision: str = "fp32",
) -> dict[str, Any]:
    """Rough wall-clock time for a training run, treating each step as memory-bandwidth bound."""
    if dataset_size <= 0 or epochs <= 0:
        raise ValueError("dataset_size and epochs must be positive.")
    gpu, (vram, bandwidth, has_bf16) = _gpu(target_gpu)
    model = advisor.resolve(model_name)
    est = advisor.estimate_resource_footprint(model["name"], num_elements, 1, precision)
    activation_bytes = est["memory_gb"]["activations_training"] * GB
    params = est["parameter_count"] or 0

    batching = recommend_batching_strategy(advisor, model["name"], num_elements, vram)
    micro = max(1, batching["micro_batch"])
    steps_per_epoch = math.ceil(dataset_size / micro)
    effective_bw = bandwidth * BANDWIDTH_EFFICIENCY
    per_sample_s = ACTIVATION_PASSES * activation_bytes / effective_bw
    per_step_optimizer_s = params * 16 / effective_bw
    epoch_s = dataset_size * per_sample_s + steps_per_epoch * per_step_optimizer_s
    hours = epochs * epoch_s / 3600

    warnings = []
    if est["memory_gb"]["training_peak"] > vram:
        warnings.append(f"One sample needs ~{est['memory_gb']['training_peak']} GB; {gpu} has {vram} GB.")
    if precision in ("bf16", "amp") and not has_bf16:
        warnings.append(f"{gpu} has no native bf16; use fp16 autocast with a GradScaler.")
    pinn = model["metadata"]["element"] == "query point" and "pde_operator" in model["metadata"]["tasks"]
    if pinn:
        warnings.append("PINN residuals with second derivatives take roughly 3x longer per step.")
    if hours > 48:
        warnings.append("Multi-day run: benchmark a 1-epoch pilot and a smaller baseline first.")
    return {
        "model": model["name"],
        "gpu": gpu,
        "micro_batch": micro,
        "steps_per_epoch": steps_per_epoch,
        "seconds_per_epoch": round(epoch_s, 2),
        "total_hours": round(hours, 2),
        "plausible_range_hours": [round(hours * 0.5, 2), round(hours * 4, 2)],
        "pinn_residuals": pinn,
        "assumptions": (f"Memory-bound steps: {ACTIVATION_PASSES} activation passes at {BANDWIDTH_EFFICIENCY:.0%} "
                        f"of {bandwidth / 1e12:.2f} TB/s; Adam update 16 B/param per step; data loading, "
                        "evaluation and k-NN/FPS kernels ignored. Calibrate with a timed pilot epoch."),
        "warnings": warnings,
    }


# ---------------------------------------------------------------- benchmark workflow


DASHBOARD_OPTIMIZERS = ("adam", "adamw", "sgd")


def lr_schedule_spec(scheduler: dict[str, Any] | None, scale: float = 1.0) -> str:
    """Paper scheduler (catalog ``paper_recipe.scheduler``, in paper epochs) as the trainer's schedule string,
    with epoch counts multiplied by ``scale``: ``none`` | ``cosine:<eta_min>`` | ``step:<every>:<gamma>``
    | ``multistep:<e1>,<e2>:<gamma>``."""
    kind = (scheduler or {}).get("kind")
    if kind == "cosine":
        return f"cosine:{scheduler.get('eta_min', 0.0)}"
    if kind == "step":
        return f"step:{max(1, round(scheduler['every_epochs'] * scale))}:{scheduler['gamma']}"
    if kind == "multistep":
        milestones = sorted({max(1, round(m * scale)) for m in scheduler["milestones"]})
        return f"multistep:{','.join(map(str, milestones))}:{scheduler['gamma']}"
    return "none"


def recommend_training_setup(
    advisor: Advisor,
    model_name: str,
    num_elements: int,
    vram_gb: float = 8.0,
    required_symmetries: list[str] | None = None,
    target_kind: str = "scalar",
    max_epochs: int = 300,
) -> dict[str, Any]:
    """Training settings from the model's paper recipe (catalog ``evidence.paper_recipe``), adapted to the
    problem: VRAM caps the batch, required rotation symmetry adds augmentation or invariant inputs.
    Every setting comes with the reason it was chosen; ``warnings`` list what the paper does not state."""
    if num_elements <= 0 or vram_gb <= 0 or max_epochs <= 0:
        raise ValueError("num_elements, vram_gb and max_epochs must be positive.")
    model = advisor.resolve(model_name)
    meta, name = model["metadata"], model["name"]
    evidence = meta.get("evidence") or {}
    recipe = evidence.get("paper_recipe") or {}
    paper = evidence.get("paper") or {}
    cite = f"{paper.get('title', name)} ({paper.get('venue', 'paper')})"
    reasons: list[str] = []
    warnings: list[str] = []
    missing = [key for key in ("optimizer", "lr", "batch_size", "epochs") if recipe.get(key) is None]
    if missing:
        warnings.append(f"The paper does not state {', '.join(missing)}; defaults are used for those.")

    optimizer = recipe.get("optimizer") if recipe.get("optimizer") in DASHBOARD_OPTIMIZERS else "adam"
    lr = float(recipe.get("lr") or 1e-3)
    momentum = float(recipe.get("momentum") or (0.9 if optimizer == "sgd" else 0.0))
    weight_decay = float(recipe.get("weight_decay") or 0.0)
    reasons.append(f"Optimizer {optimizer}, learning rate {lr:g}"
                   + (f", momentum {momentum:g}" if optimizer == "sgd" else "")
                   + (f", weight decay {weight_decay:g}" if weight_decay else "")
                   + (f": as in {cite}." if recipe.get("optimizer") and recipe.get("lr") else
                      ": Adam at 1e-3 because the paper does not state its optimizer or learning rate."))

    paper_epochs = recipe.get("epochs")
    epochs = min(int(paper_epochs), max_epochs) if paper_epochs else min(100, max_epochs)
    if paper_epochs and epochs < paper_epochs:
        reasons.append(f"{epochs} epochs: the paper trains {paper_epochs}; capped at {max_epochs}, schedule scaled to match.")
    elif paper_epochs:
        reasons.append(f"{epochs} epochs, as in the paper.")
    else:
        reasons.append(f"{epochs} epochs: the paper gives no epoch count"
                       + (f" ({recipe['steps']:,} iterations)." if recipe.get("steps") else "."))
    schedule = lr_schedule_spec(recipe.get("scheduler"), epochs / paper_epochs if paper_epochs else 1.0)
    if schedule != "none":
        reasons.append(f"LR schedule {schedule}: the paper's '{recipe.get('schedule')}'.")
    elif recipe.get("schedule"):
        warnings.append(f"The paper's schedule '{recipe['schedule']}' is iteration-based or cyclic; "
                        "the dashboard keeps the learning rate constant.")

    paper_batch = recipe.get("batch_size")
    batching = recommend_batching_strategy(advisor, name, num_elements, vram_gb, int(paper_batch or 8))
    batch = max(1, min(int(paper_batch or 8), batching["micro_batch"] or 1))
    if batching["micro_batch"] == 0:
        warnings.append(f"One sample of {num_elements:,} {meta['element']}s may not fit in {vram_gb:g} GB; "
                        "lower the points per sample.")
    if paper_batch and batch < paper_batch:
        reasons.append(f"Batch {batch}: the paper uses {paper_batch}, but the memory estimate fits only {batch} "
                       f"samples of {num_elements:,} elements in {vram_gb:g} GB.")
    else:
        reasons.append(f"Batch {batch}" + (": as in the paper." if paper_batch else
                                           ": the paper does not state one; 8 fits the memory estimate."))
    if batch < 4 and _defaults(model).get("normalization") == "batch":
        warnings.append(f"Batch {batch} with BatchNorm gives noisy statistics.")
    if recipe.get("points") and recipe["points"] != num_elements:
        reasons.append(f"{num_elements:,} points per sample from your problem (the paper used {recipe['points']:,}).")

    model_config: dict[str, Any] = {}
    augment_rotation = False
    if "rotation" in (required_symmetries or []) and "rotation" not in meta.get("symmetries", []):
        if name == "diffusionnet":
            model_config["input_features"] = "hks"
            reasons.append("input_features='hks': you need rotation invariance; heat-kernel signatures are "
                           "rotation invariant and the paper reports equal or better accuracy with them "
                           "(SHREC'11 99.7%).")
        elif target_kind in ("vector", "tensor"):
            warnings.append("Rotation is required but the target is a vector/tensor: rotating only the inputs "
                            "would corrupt the labels. Use an equivariant model or rotate targets too.")
        else:
            augment_rotation = True
            reasons.append(f"Random rotation augmentation: you need rotation invariance and {meta['display_name']} "
                           "is not rotation invariant (checked numerically).")
    other_aug = [a for a in recipe.get("augmentation") or [] if "rotation" not in a]
    if other_aug:
        warnings.append("The paper also uses " + "; ".join(other_aug) + ". The dashboard does not apply these.")
    if evidence.get("implementation_notes"):
        warnings.append("Implementation vs paper: " + evidence["implementation_notes"])

    return {
        "model": name,
        "paper": paper,
        "paper_recipe": recipe,
        "setup": {
            "optimizer": optimizer, "learning_rate": lr, "momentum": momentum, "weight_decay": weight_decay,
            "lr_schedule": schedule, "epochs": epochs, "batch_size": batch, "num_points": num_elements,
            "augment_rotation": augment_rotation, "model_config": model_config,
        },
        "reasons": reasons,
        "warnings": warnings,
        "benchmark_datasets": evidence.get("benchmark_datasets", []),
        "checked": evidence.get("checked", "not checked against the paper"),
    }


def generate_benchmark_config(
    advisor: Advisor,
    model_name: str,
    dataset_name: str,
    num_elements: int,
    hardware_target: str = "A100",
    dataset_size: int = 1000,
    epochs: int = 100,
) -> dict[str, Any]:
    """Execution-ready YAML config and Python launcher using ModelRegistry.create()."""
    gpu, (vram, _, _) = _gpu(hardware_target)
    model = advisor.resolve(model_name)
    info = model["registry"] or {}
    registry_name = model["registry_name"] or model["name"]
    batching = recommend_batching_strategy(advisor, model["name"], num_elements, vram)
    budget = estimate_training_budget(advisor, model["name"], num_elements, dataset_size, epochs, gpu,
                                      batching["precision"])
    model_config = {k: v for k, v in _defaults(model).items() if k not in ("metadata", "name", "model_type")}
    recipe = recommend_training_setup(advisor, model["name"], num_elements, vram, max_epochs=epochs)
    setup = recipe["setup"]

    config = {
        "model": {"name": registry_name, "config": model_config},
        "data": {
            "dataset": dataset_name,
            "representation": model["metadata"]["representations"][0],
            "required_inputs": info.get("required_inputs", []),
            "elements_per_sample": num_elements,
            "normalize_inputs": "standardize per channel with training-set statistics",
            "normalize_targets": "standardize per channel; invert before computing physical errors",
        },
        "training": {
            "hardware": gpu,
            "precision": batching["precision"],
            "micro_batch": max(1, batching["micro_batch"]),
            "gradient_accumulation_steps": batching["gradient_accumulation_steps"] or 1,
            "epochs": epochs,
            "optimizer": {"name": setup["optimizer"], "lr": setup["learning_rate"], "momentum": setup["momentum"],
                          "weight_decay": setup["weight_decay"]},
            "lr_schedule": setup["lr_schedule"],  # from the paper; see recommend_training_setup
            "paper": recipe["paper"].get("url"),
            "grad_clip_norm": 1.0,
            "seed": 0,
        },
    }
    config_yaml = yaml.safe_dump(config, sort_keys=False)
    python = f'''"""Launcher for {registry_name} on {dataset_name} (generated by neural3d-advisor)."""
import torch
import yaml

from models import ModelRegistry

cfg = yaml.safe_load(open("benchmark_{registry_name}.yaml"))
model = ModelRegistry.create(cfg["model"]["name"], cfg["model"]["config"]).cuda()
train = cfg["training"]
opt = train["optimizer"]
optimizer = {{"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}}[opt["name"]](
    model.parameters(), lr=opt["lr"], weight_decay=opt["weight_decay"],
    **({{"momentum": opt["momentum"]}} if opt["name"] == "sgd" else {{}}))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train["epochs"])  # paper schedule: train["lr_schedule"]
dtype = torch.bfloat16 if train["precision"] == "bf16" else None
loader = ...  # DataLoader yielding (ModelInput, target); see {model["examples"][0] if model["examples"] else "examples/"}

for epoch in range(train["epochs"]):
    for step, (inputs, target) in enumerate(loader):
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            prediction = model(inputs).predictions
            loss = torch.nn.functional.mse_loss(prediction.float(), target)
        (loss / train["gradient_accumulation_steps"]).backward()
        if (step + 1) % train["gradient_accumulation_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    scheduler.step()
'''
    return {
        "yaml": config_yaml,
        "python": python,
        "estimated_hours": budget["total_hours"],
        "notes": batching["notes"] + budget["warnings"],
        "examples": model["examples"],
    }


def suggest_ablation_matrix(advisor: Advisor, primary_model: str, max_contenders: int = 4) -> dict[str, Any]:
    """Baselines that isolate the primary model's architectural contribution."""
    primary = advisor.resolve(primary_model)
    pmeta = primary["metadata"]
    p_rf = pmeta.get("receptive_field", {}).get("kind")
    p_reps = set(pmeta["representations"])
    pool = [
        m for n, m in advisor.state["models"].items()
        if n != primary["name"] and set(m["metadata"]["tasks"]) & set(pmeta["tasks"])
    ]

    def best(candidates: list[dict], key) -> dict | None:
        return max(candidates, key=key) if candidates else None

    same_rep = [m for m in pool if p_reps & set(m["metadata"]["representations"])]
    simpler = [m for m in same_rep if m["metadata"]["expressivity"] < pmeta["expressivity"]]
    roles = [
        ("simplest baseline (same input)", best(simpler, lambda m: -m["metadata"]["expressivity"]),
         "Shows the gain over a cheap, low-capacity model on identical inputs."),
        ("same input, different interaction mechanism",
         best([m for m in same_rep if m["metadata"].get("receptive_field", {}).get("kind") != p_rf],
              lambda m: m["metadata"]["expressivity"]),
         "Isolates the interaction mechanism (global vs local vs hop-based) at equal data."),
        ("different representation", best([m for m in pool if not p_reps & set(m["metadata"]["representations"])],
                                          lambda m: m["metadata"]["expressivity"]),
         "Tests whether the representation (and its conversion loss) matters."),
        ("same interaction mechanism, different model",
         best([m for m in pool if m["metadata"].get("receptive_field", {}).get("kind") == p_rf],
              lambda m: m["metadata"]["expressivity"]),
         "Separates the specific design from its receptive-field family."),
    ]
    matrix, seen = [], set()
    for role, model, purpose in roles:
        if model is None or model["name"] in seen or len(matrix) >= max_contenders:
            continue
        seen.add(model["name"])
        meta = model["metadata"]
        matrix.append({
            "model": model["name"],
            "role": role,
            "isolates": purpose,
            "contrast": (f"{p_rf} on {sorted(p_reps)} vs {meta.get('receptive_field', {}).get('kind')} on "
                         f"{meta['representations']}"),
            "key_bias": meta["inductive_biases"][0],
        })
    return {
        "primary": primary["name"],
        "contenders": matrix,
        "protocol": "Same splits, target normalisation, epochs and tuning budget for every row; report mean and "
                    "std over >= 3 seeds, plus parameter count and training peak memory.",
    }
