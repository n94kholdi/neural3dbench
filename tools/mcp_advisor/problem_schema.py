"""Task-specific problem questions and their translation into advisor inputs.

Each task asks only the questions that matter for it: a segmentation problem is asked about label
granularity and scene scale, a simulation about time behaviour and solution character.
Every option adds words the advisor matches (catalog.yaml ``keywords``, advisor task keywords,
guardrails text checks) and may set pre-flight assumptions (target_kind, geometry_varies,
topology_changes, evaluate_at_unseen_resolution, extra required symmetries).
tests/test_problem_schema.py fails if an option stops matching any advisor keyword.
"""

from __future__ import annotations

from typing import Any

# Application domains, grouped. Words avoid advisor.MISSING_CAPABILITIES keywords (diffusion, generat,
# instance...), which would force a "no_fit" verdict.
DOMAINS = {
    "Autonomy · Autonomous driving (LiDAR)": "autonomous driving lidar outdoor kitti pedestrian",
    "Autonomy · Robot grasping & manipulation": "pose rigid partial",
    "Autonomy · SLAM, mapping & localization": "scan matching lidar outdoor",
    "Autonomy · Drones & aerial mapping": "outdoor lidar dense",
    "Scenes · Indoor scenes, AR / VR": "indoor scannet sun rgb-d furniture",
    "Scenes · 3D scanning & digitization": "scan partial occlusion surface",
    "Scenes · Buildings, BIM & digital twins": "indoor cad geometry",
    "Scenes · Remote sensing, forestry & geospatial": "outdoor lidar dense hierarchical",
    "Design · CAD & mechanical parts": "cad nurbs mesh",
    "Design · Manufacturing inspection & metrology": "rigid cad surface robust",
    "Design · 3D printing / additive manufacturing": "mesh triangle cad",
    "Design · Architecture & interior design": "furniture indoor cad",
    "Graphics · Games, VFX & 3D assets": "mesh triangle polygon soup shape",
    "Graphics · Humans, faces & avatars": "deformation mesh surface",
    "Graphics · Shape retrieval & 3D asset catalogs": "shape modelnet",
    "Graphics · Cultural heritage": "scan surface robust",
    "Medicine · Medical imaging (CT / MRI)": "medical ct volumetric voxel",
    "Medicine · Anatomy & surgical planning meshes": "medical mesh surface",
    "Science · Molecules & proteins": "graph node unordered",
    "Science · Particle & astronomy point data": "unordered knn",
    "Science · Geoscience & seismic volumes": "volumetric grid dense",
    "Science · Climate & weather grids": "grid spectral global",
    "Physics · Fluid dynamics / CFD": "cfd fluid pressure",
    "Physics · Aerodynamics (cars, aircraft)": "aerodynamics airfoil car pressure",
    "Physics · Structural & solid mechanics": "structural elasticity deformation",
    "Physics · Heat transfer": "heat",
    "Physics · Porous media & subsurface flow": "darcy",
    "Physics · Cloth & soft bodies": "cloth deformation",
    "Physics · Crash, fracture & contact": "contact topology remeshing adaptive",
    "Physics · Electromagnetics & acoustics": "spectral",
}

_ALL_GROUPS = ("Autonomy", "Scenes", "Design", "Graphics", "Medicine", "Science", "Physics")
DOMAIN_GROUPS_BY_TASK = {
    "surrogate_simulation": ("Physics", "Science", "Medicine", "Design"),
    "pde_operator": ("Physics", "Science"),
    "field_fitting": ("Physics", "Graphics", "Design", "Scenes", "Medicine"),
    "segmentation_classification": ("Autonomy", "Scenes", "Design", "Graphics", "Medicine", "Science"),
    "shape_completion": ("Autonomy", "Scenes", "Design", "Graphics", "Medicine"),
    "object_detection": ("Autonomy", "Scenes"),
    "registration": ("Autonomy", "Scenes", "Design", "Graphics", "Medicine"),
}


def _choice(key: str, label: str, options: dict[str, Any], help: str = "") -> dict[str, Any]:
    """Single-choice question. Option values: keywords, or (keywords, assumptions to set)."""
    opts = {name: {"keywords": v, "sets": {}} if isinstance(v, str) else {"keywords": v[0], "sets": v[1]}
            for name, v in options.items()}
    return {"key": key, "label": label, "kind": "choice", "help": help, "options": opts}


def _flag(key: str, label: str, keywords: str = "", sets: dict | None = None, help: str = "") -> dict[str, Any]:
    """Yes/no question; keywords and assumptions apply only when checked."""
    return {"key": key, "label": label, "kind": "flag", "help": help, "keywords": keywords, "sets": sets or {}}


_ROTATION = {"add_symmetries": ["rotation"]}
_FIELD_TARGET = _choice("target", "What the model predicts", {
    "Scalar field (pressure, temperature)": ("", {"target_kind": "scalar"}),
    "Vector field (velocity, displacement)": ("velocity displacement", {"target_kind": "vector"}),
    "Tensor field (stress, strain)": ("stress strain", {"target_kind": "tensor"}),
}, "Vector/tensor targets need rotation equivariance, not invariance.")
_SOLUTION = _choice("solution", "Solution character", {
    "Not sure": "",
    "Smooth": "smooth",
    "Sharp gradients / fine detail": "high-frequency",
    "Shocks, cracks or contact": "shock discontinuity contact",
}, "Shocks and cracks make spectral models (FNO) ring.")
_GEOMETRY = _choice("geometry", "Geometry across samples", {
    "One fixed geometry": "",
    "Parametric family of shapes": ("parametric", {"geometry_varies": True}),
    "Arbitrary irregular shapes": ("irregular geometry", {"geometry_varies": True}),
})
_INTERACTION = _choice("interaction", "Interaction range", {
    "Not sure": "",
    "Mostly local": "local",
    "Long-range / global coupling": "global",
}, "Global coupling favours attention/operator models over local kNN ones.")
_TOPOLOGY = _flag("topology_changes", "Mesh connectivity changes between samples / steps",
                  sets={"topology_changes": True}, help="Remeshing, fracture, adaptive refinement.")
_PERIODIC = _flag("periodic", "Periodic domain", "periodic")
_ORIENTATION = _choice("orientation", "Object orientation", {
    "Pre-aligned (canonical pose)": "",
    "Arbitrary rotations": ("", _ROTATION),
}, "Arbitrary rotations add rotation to the required symmetries.")

QUESTIONS: dict[str, list[dict[str, Any]]] = {
    "surrogate_simulation": [
        _FIELD_TARGET,
        _choice("time", "Time behaviour", {"Steady state (one solve)": "",
                                           "Transient (rollout over time)": "transient rollout"}),
        _SOLUTION, _GEOMETRY, _INTERACTION, _TOPOLOGY, _PERIODIC,
    ],
    "pde_operator": [
        _FIELD_TARGET,
        _choice("time", "Time behaviour", {"Single solution field": "",
                                           "Time evolution": "transient"}),
        _SOLUTION, _GEOMETRY, _INTERACTION, _PERIODIC,
        _flag("resolution", "Evaluate at a finer / other resolution", "super-resolution",
              {"evaluate_at_unseen_resolution": True}),
    ],
    "field_fitting": [
        _choice("fitted", "What is fitted", {
            "Shape surface (SDF / occupancy)": "sdf implicit surface",
            "PDE solution (PINN)": "pinn derivative",
            "Signal / neural field (image, radiance)": "neural field high-frequency",
        }),
        _choice("detail", "Level of detail", {"Smooth": "smooth", "Fine, high-frequency": "high-frequency"}),
        _choice("domain_geometry", "Domain geometry", {
            "Simple box / analytic domain": "coordinate",
            "Complex CAD / spline geometry": "nurbs cad isogeometric",
        }),
        _flag("derivatives", "Need derivatives (normals, gradients, PDE residuals)", "derivative continuous"),
        _flag("many_shapes", "Fit many shapes with one network", sets={"geometry_varies": True}),
    ],
    "segmentation_classification": [
        _choice("granularity", "What is labelled", {
            "One label per object / shape": ("classification", {"required_output": "shape_label"}),
            "Part label per point (object parts)": ("segmentation part", {"required_output": "point_label"}),
            "Semantic label per point (whole scene)": ("segmentation dense", {"required_output": "point_label"}),
            "Separate id per object instance": "instance",
        }, "Instance ids are not covered by any registered model yet; the advisor says so."),
        _choice("scale", "Scene scale", {
            "Single objects": "shape",
            "Indoor rooms": "indoor scannet",
            "Outdoor / large-scale LiDAR": "outdoor lidar",
        }),
        _choice("detail", "What decides the label", {
            "Not sure": "",
            "Fine local geometry (edges, thin parts)": "local geometry knn edge",
            "Overall shape": "global",
        }),
        _choice("quality", "Data quality", {
            "Clean (synthetic / CAD)": "",
            "Noisy or partial real scans": "robust partial occlusion",
        }),
        _ORIENTATION,
    ],
    "shape_completion": [
        _choice("input", "Input", {
            "Partial scan (occluded regions)": "partial scan occlusion",
            "Sparse / low-density points": "upsample densify",
        }),
        _choice("output", "Output", {"Coarse complete shape": "",
                                     "Dense detailed shape": "coarse-to-fine dense"}),
        _choice("supervision", "Training data", {
            "Complete ground-truth shapes available": "",
            "No complete shapes (self-supervised)": "unsupervised autoencoder",
        }),
        _ORIENTATION,
    ],
    "object_detection": [
        _choice("scene", "Scene", {
            "Outdoor driving": "outdoor autonomous driving kitti",
            "Indoor rooms": "indoor scannet votes",
        }),
        _choice("sensor", "Sensor", {"LiDAR": "lidar", "RGB-D / depth camera": "sun rgb-d"}),
        _choice("objects", "Objects", {
            "Vehicles, pedestrians, cyclists": "car pedestrian",
            "Furniture & household objects": "furniture",
        }),
        _flag("realtime", "Real-time on the vehicle / robot", "bev"),
    ],
    "registration": [
        _choice("transform", "Transformation", {
            "Rigid (rotation + translation)": ("rigid", {"required_output": "rigid_pose"}),
            "Non-rigid / deformable": ("", {"required_output": "deformation_field"}),
        }),
        _choice("overlap", "Overlap between the two clouds", {"Full": "", "Partial": "partial"}),
        _choice("init", "Initial alignment", {
            "Rough alignment known": "icp",
            "Arbitrary pose": ("correspondence global", _ROTATION),
        }),
        _choice("data", "Data", {"Object scans": "scan", "LiDAR sweeps / maps": "scan matching lidar outdoor"}),
        _flag("noisy", "Noisy or outlier-heavy", "robust"),
    ],
}


# What each task predicts unless an answer says otherwise (advisor.OUTPUTS).
TASK_OUTPUTS = {
    "surrogate_simulation": "point_field",
    "pde_operator": "point_field",
    "field_fitting": "point_field",
    "segmentation_classification": "shape_label",
    "shape_completion": "point_set",
    "object_detection": "boxes",
    "registration": "rigid_pose",
}


def problem_questions(task: str) -> dict[str, Any]:
    """Domains and questions relevant to ``task``. Unknown or uncovered tasks get all domains, no questions."""
    groups = DOMAIN_GROUPS_BY_TASK.get(task, _ALL_GROUPS)
    return {
        "task": task,
        "domains": [d for d in DOMAINS if d.split(" · ")[0] in groups],
        "questions": QUESTIONS.get(task, []),
    }


def compose_problem(
    task: str, answers: dict[str, Any] | None = None, domains: list[str] | None = None, extra: str = ""
) -> dict[str, Any]:
    """Turn answers into a ``recommend_architecture`` description and pre-flight assumptions.

    answers: question key -> option label (choice) or bool (flag). Missing choices use the first option.
    Returns description, required_output (for recommend_architecture), assumptions (target_kind,
    geometry_varies, topology_changes, evaluate_at_unseen_resolution, add_symmetries) and dynamic_topology.
    """
    answers = answers or {}
    unknown_domains = [d for d in domains or [] if d not in DOMAINS]
    if unknown_domains:
        raise ValueError(f"Unknown domains {unknown_domains}; call get_problem_questions for the list.")
    words = [task.replace("_", " ")] + [DOMAINS[d] for d in domains or []]
    assumptions: dict[str, Any] = {"target_kind": "scalar", "geometry_varies": False, "topology_changes": False,
                                   "evaluate_at_unseen_resolution": False, "add_symmetries": []}
    output = {"required_output": TASK_OUTPUTS.get(task)}

    def apply(keywords: str, sets: dict[str, Any]) -> None:
        words.append(keywords)
        for key, value in sets.items():
            if key == "required_output":
                output[key] = value
            elif key == "add_symmetries":
                assumptions[key] += [s for s in value if s not in assumptions[key]]
            else:
                assumptions[key] = value

    questions = QUESTIONS.get(task, [])
    known = {q["key"] for q in questions}
    if set(answers) - known:
        raise ValueError(f"Unknown answers {sorted(set(answers) - known)} for {task}; expected {sorted(known)}.")
    for q in questions:
        value = answers.get(q["key"])
        if q["kind"] == "flag":
            if value:
                apply(q["keywords"], q["sets"])
            continue
        label = value if value is not None else next(iter(q["options"]))
        if label not in q["options"]:
            raise ValueError(f"{q['key']}: {label!r} is not one of {list(q['options'])}.")
        apply(q["options"][label]["keywords"], q["options"][label]["sets"])

    words.append(extra.strip())
    return {
        "description": " ".join(w for w in words if w),
        "required_output": output["required_output"],
        "assumptions": assumptions,
        "dynamic_topology": assumptions["topology_changes"],
    }
