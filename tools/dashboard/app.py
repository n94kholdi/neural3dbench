"""Neural3DBench Streamlit dashboard: advisor, 3D preview, MLflow training and inspection.

Launch from the repository root::

    streamlit run tools/dashboard/app.py

Runs and checkpoints land in ``./mlruns`` and ``./checkpoints`` relative to the working
directory; view them with ``MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns`` (http://localhost:5000).
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MLFLOW_UI_URL = "http://localhost:5000"
REPRESENTATIONS = {
    "point_cloud": "Point cloud",
    "mesh_graph": "Mesh / graph",
    "voxel_grid": "Voxel grid",
    "continuous_coords": "Continuous fields (coordinates)",
}
TASKS = ("surrogate_simulation", "pde_operator", "segmentation_classification", "field_fitting", "shape_completion",
         "object_detection", "registration")
# Not covered by any registered model yet; listed so the advisor can say so (tools/mcp_advisor/advisor.py).
UNCOVERED_TASKS = ("instance_segmentation", "shape_generation", "scene_flow")
PRECISIONS = ("fp32", "fp16", "bf16", "amp")
POLL_SECONDS = 2.0
_VERTEX_KEYS = ("vertices", "coords", "coordinates", "points", "pos")
_FIELD_KEYS = ("field", "target", "features", "values")
_PLY_LABEL_KEYS = ("label", "class", "part", "scalar_label", "segment", "seg")


# ---------------------------------------------------------------- geometry loading


def _fan(polygon: list[int]) -> list[list[int]]:
    """Triangulate a convex polygon as a fan around its first vertex."""
    return [[polygon[0], polygon[i], polygon[i + 1]] for i in range(1, len(polygon) - 1)]


def _geometry(vertices: Any, faces: Any = None, field: Any = None, labels: Any = None) -> dict[str, Any]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"vertices must have shape [N, 3]; got {vertices.shape}.")
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3) if faces is not None and len(faces) else None
    if faces is not None and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("faces index outside the vertex array.")
    field = None if field is None else np.asarray(field, dtype=np.float64).reshape(len(vertices), -1)
    labels = None if labels is None else np.asarray(labels, dtype=np.int64).reshape(len(vertices))
    return {"vertices": vertices, "faces": faces, "field": field, "labels": labels}


def parse_obj(text: str) -> dict[str, Any]:
    """Wavefront OBJ: ``v`` and ``f`` records only; polygons are fan-triangulated."""
    vertices, faces = [], []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            # "f 1/2/3 ..." -> vertex index 1; negative indices count back from the current end.
            index = [int(token.split("/")[0]) for token in parts[1:]]
            faces += _fan([i - 1 if i > 0 else len(vertices) + i for i in index])
    return _geometry(vertices, faces)


def parse_ply(data: bytes) -> dict[str, Any]:
    """ASCII PLY with a ``vertex`` element (x, y, z, optional per-point label) and an optional ``face`` element.

    shortcut: binary PLY is rejected; add the plyfile dependency if binary uploads matter.
    """
    header, _, body = data.partition(b"end_header")
    lines = header.decode("ascii", errors="replace").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError("not a PLY file.")
    if not any(line.startswith("format ascii") for line in lines):
        raise ValueError("only ASCII PLY is supported; convert binary PLY to ASCII or OBJ.")
    counts: dict[str, int] = {}
    vertex_props: list[str] = []
    element = None
    for line in lines:
        parts = line.split()
        if parts[:1] == ["element"]:
            element, counts[parts[1]] = parts[1], int(parts[2])
        elif parts[:1] == ["property"] and element == "vertex":
            vertex_props.append(parts[-1])
    columns = [vertex_props.index(axis) for axis in ("x", "y", "z")]
    label_column = next((vertex_props.index(k) for k in _PLY_LABEL_KEYS if k in vertex_props), None)
    rows = body.decode("ascii").split("\n")[1:]  # drop the rest of the end_header line
    rows = [row.split() for row in rows if row.strip()]
    n_vertices = counts.get("vertex", 0)
    vertices = [[float(row[c]) for c in columns] for row in rows[:n_vertices]]
    labels = None if label_column is None else [int(float(row[label_column])) for row in rows[:n_vertices]]
    faces = []
    for row in rows[n_vertices : n_vertices + counts.get("face", 0)]:
        faces += _fan([int(i) for i in row[1 : 1 + int(row[0])]])
    return _geometry(vertices, faces, labels=labels)


def _drop_batch(value: Any) -> Any:
    """``[1, N, C]`` -> ``[N, C]``; anything else unchanged."""
    if value is None:
        return None
    tensor = torch.as_tensor(value)
    return tensor[0] if tensor.ndim == 3 and len(tensor) == 1 else tensor


def parse_pt(data: bytes) -> dict[str, Any]:
    """Torch file holding a ``[N, 3]`` / ``[1, N, 3]`` tensor or a dict with vertices, faces and a field."""
    obj = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if isinstance(obj, torch.Tensor):
        return _geometry(_drop_batch(obj))
    if isinstance(obj, dict):
        key = next((k for k in _VERTEX_KEYS if k in obj), None)
        if key is None:
            raise ValueError(f"dict needs one of the keys {_VERTEX_KEYS}; got {sorted(obj)}.")
        field = next((obj[k] for k in _FIELD_KEYS if k in obj), None)
        return _geometry(_drop_batch(obj[key]), obj.get("faces"), _drop_batch(field))
    raise ValueError(f"unsupported .pt content: {type(obj).__name__}.")


def load_geometry(filename: str, data: bytes) -> dict[str, Any]:
    """Dispatch on the file extension; returns ``{"vertices": [N, 3], "faces": [F, 3] | None, "field": [N, C] | None}``."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".obj":
        return parse_obj(data.decode("utf-8", errors="replace"))
    if suffix == ".ply":
        return parse_ply(data)
    if suffix == ".pt":
        return parse_pt(data)
    raise ValueError(f"unsupported file type {suffix!r}; use .obj, .ply or .pt.")


def load_sample(name: str, num_points: int = 4096) -> dict[str, Any]:
    """Repository samples: shapes from examples/train_pointnet_synthetic.py and the trainer datasets."""
    from tools.dashboard.trainer import TrainingConfig, make_synthetic_dataset

    kind, _, shape = name.partition(":")
    if kind == "shape":
        from examples.train_pointnet_synthetic import make_shape

        return _geometry(make_shape(shape, num_points, torch.Generator().manual_seed(0)))
    coords, target = make_synthetic_dataset(TrainingConfig(dataset=shape, num_samples=1, num_points=num_points))
    return _geometry(coords[0], field=target[0])


DOWNLOADS = {
    "shapenet_part": "ShapeNetPart: part segmentation, 16 categories (707 MB)",
    "modelnet10": "ModelNet10: shape classification, 10 classes (473 MB)",
    "modelnet40": "ModelNet40: shape classification, 40 classes (2.0 GB)",
    "modelnet10_mesh": "ModelNet10 meshes: mesh classification, 10 classes (473 MB)",
    "shrec11": "SHREC'11 meshes: mesh classification, 30 classes (3.2 MB)",
}
# Mesh downloads: dashboard dataset name -> data_sources.MESH_DATASETS key.
MESH_DOWNLOADS = {"modelnet10_mesh": "modelnet10", "shrec11": "shrec11"}
# What each download is labelled with (tools/mcp_advisor/advisor.OUTPUTS), to offer only fitting ones.
DOWNLOAD_OUTPUTS = {"shapenet_part": "point_label", "modelnet10": "shape_label", "modelnet40": "shape_label",
                    "modelnet10_mesh": "shape_label", "shrec11": "shape_label"}

SAMPLES = {
    "shape:sphere": "Sphere (examples/train_pointnet_synthetic.py)",
    "shape:cube": "Cube (examples/train_pointnet_synthetic.py)",
    "shape:cylinder": "Cylinder (examples/train_pointnet_synthetic.py)",
    "dataset:deformed_sphere": "Deformed sphere + radial displacement (trainer dataset)",
    "dataset:poisson": "Poisson manufactured solution (trainer dataset)",
    "dataset:chair_parts": "Box chair colored by part id (trainer dataset)",
}


def parse_labelled_upload(files: list[tuple[str, bytes]], num_points: int) -> dict[str, Any]:
    """One ``.npz`` (many samples) or several labelled ASCII ``.ply`` files (one sample each).

    Returns ``{"coords" [B, N, 3], "labels" | "target", "class_names"}`` for ``tasks.save_npz_dataset``.
    """
    from tools.dashboard.data_sources import read_npz_upload, stack_samples

    suffixes = {Path(name).suffix.lower() for name, _ in files}
    if suffixes == {".npz"}:
        if len(files) != 1:
            raise ValueError("upload one .npz holding every sample.")
        return read_npz_upload(files[0][1])
    if suffixes != {".ply"}:
        raise ValueError("upload either one .npz or one or more .ply files, not a mix.")
    if len(files) < 2:
        raise ValueError("a dataset needs at least 2 samples: upload one labelled .ply per shape.")
    samples = []
    for name, data in files:
        geometry = parse_ply(data)
        if geometry["labels"] is None:
            raise ValueError(f"{name}: no per-point label property ({', '.join(_PLY_LABEL_KEYS)}).")
        samples.append((geometry["vertices"], geometry["labels"]))
    coords, labels = stack_samples(samples, num_points)
    return {"coords": coords, "labels": labels, "target": None, "class_names": []}


# ---------------------------------------------------------------- training helpers


def form_values(config: Any) -> dict[str, Any]:
    """Map a ``TrainingConfig`` onto the Tab 3 widget keys."""
    return {
        "train_model": config.model_name,
        "train_model_config": json.dumps(config.model_config),
        "train_dataset": config.dataset,
        "train_epochs": config.epochs,
        "train_lr": config.learning_rate,
        "train_batch": config.batch_size,
        "train_optimizer": config.optimizer,
        "train_momentum": config.momentum,
        "train_weight_decay": config.weight_decay,
        "train_schedule": config.lr_schedule,
        "train_augment_rotation": config.augment_rotation,
        "train_samples": config.num_samples,
        "train_points": config.num_points,
        "train_experiment": config.experiment_name,
        "train_device": config.device,
    }


def config_from_form(state: Any) -> Any:
    from tools.dashboard.trainer import TrainingConfig

    model_config = json.loads(state["train_model_config"] or "{}")
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a JSON object.")
    return TrainingConfig(
        model_name=state["train_model"],
        model_config=model_config,
        dataset=state["train_dataset"],
        epochs=int(state["train_epochs"]),
        learning_rate=float(state["train_lr"]),
        batch_size=int(state["train_batch"]),
        optimizer=state["train_optimizer"],
        momentum=float(state["train_momentum"]),
        weight_decay=float(state["train_weight_decay"]),
        lr_schedule=state["train_schedule"].strip() or "none",
        augment_rotation=bool(state["train_augment_rotation"]),
        num_samples=int(state["train_samples"]),
        num_points=int(state["train_points"]),
        experiment_name=state["train_experiment"].strip() or "neural3dbench",
        device=state["train_device"],
    )


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def cuda_problem() -> str | None:
    """A message when the driver sees an NVIDIA GPU but CUDA cannot initialize; None otherwise."""
    if torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None
    return ("An NVIDIA GPU is present but CUDA failed to initialize, so training runs on the CPU. This often "
            "happens after suspend/resume. Stop this app, run `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm` "
            "(or reboot), then start the dashboard again.")


def mlflow_client(tracking_uri: str) -> Any:
    from mlflow.tracking import MlflowClient

    from tools.dashboard.trainer import _resolve_tracking_uri

    return MlflowClient(tracking_uri=_resolve_tracking_uri(tracking_uri))


def predict_sample(checkpoint_path: str, config: Any, sample: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reload a checkpoint and predict one validation sample: ``(coords [N, 3], target, raw model output)``.

    For a mesh dataset ``coords`` is ``(vertices [V, 3], faces [F, 3])``.

    Target and output shapes follow the dataset's task (see ``tools.dashboard.tasks``), without the batch dim.
    """
    from models import ModelRegistry
    from tools.dashboard import tasks
    from tools.dashboard.trainer import make_synthetic_dataset

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ModelRegistry.create(checkpoint["model_name"], checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    coords, target = make_synthetic_dataset(config)  # validation samples are the first rows, as in TrainingWorker
    with torch.no_grad():
        pred = tasks.forward(tasks.DATASETS[config.dataset].task, model, tasks.model_input(coords, [sample]))
    geometry = coords[sample] if tasks.DATASETS[config.dataset].mesh else coords[sample].numpy()
    return geometry, target[sample].numpy(), pred[0].numpy()


def num_validation_samples(config: Any) -> int:
    return max(1, int(round(config.num_samples * config.val_fraction)))


# ---------------------------------------------------------------- UI


def main() -> None:
    import streamlit as st

    from models import ModelRegistry
    from tools.dashboard.advisor_bridge import AdvisorBridge
    from tools.dashboard.data_sources import (
        SHAPENET_PART_RANGES,
        import_modelnet,
        import_shapenet_part,
        load_mesh_dataset,
        mesh_cache_path,
        suggest_name,
    )
    from tools.dashboard.tasks import (
        BUILTIN_DATASETS,
        DATA_DIR,
        DATASETS,
        choose_dataset,
        compatible_datasets,
        evaluation_figure,
        input_mismatch,
        make_dataset,
        save_npz_dataset,
        update_classes,
    )
    from tools.dashboard.trainer import Trainer, TrainingConfig
    from tools.mcp_advisor.problem_schema import compose_problem, problem_questions
    from tools.dashboard.visualizer import class_color, plot_labelled_points, plot_mesh, plot_point_cloud

    try:  # tools.mcp_advisor needs pyyaml
        from tools.mcp_advisor.guardrails import BOUNDARY_CONDITIONS, GPUS, SYMMETRIES
        from tools.mcp_advisor.results import best_by
    except ImportError:  # registry fallback: no pre-flight checks or measured results
        BOUNDARY_CONDITIONS, GPUS, SYMMETRIES = ("none",), {"RTX4090": None}, ()

        def best_by(rows: list, key: str) -> dict:
            return {}

    st.set_page_config(page_title="Neural3DBench", page_icon=":material/deployed_code:", layout="wide")

    @st.cache_resource
    def get_bridge() -> AdvisorBridge:
        return AdvisorBridge()

    # One trainer per server process, so a browser refresh still sees the running job.
    @st.cache_resource
    def get_trainer() -> Trainer:
        return Trainer()

    @st.cache_data(show_spinner=False)
    def cached_sample(name: str, num_points: int) -> dict[str, Any]:
        return load_sample(name, num_points)

    @st.cache_data(show_spinner=False)
    def cached_compatible(model_name: str, required_output: str | None, dataset_count: int) -> list[str]:
        """dataset_count is a cache key only: a download or upload registers new datasets."""
        return compatible_datasets(model_name, required_output)

    @st.cache_data(show_spinner=False)
    def cached_input_mismatch(model_name: str, dataset_name: str) -> str | None:
        return input_mismatch(model_name, dataset_name)

    @st.cache_data(show_spinner="Running the model on the validation sample...")
    def cached_prediction(checkpoint_path: str, config_json: str, sample: int) -> tuple[np.ndarray, ...]:
        return predict_sample(checkpoint_path, TrainingConfig(**json.loads(config_json)), sample)

    bridge, trainer = get_bridge(), get_trainer()
    model_names = sorted(ModelRegistry._registry)
    for key, value in form_values(TrainingConfig(device=default_device())).items():
        st.session_state.setdefault(key, value)

    def use_model(name: str, benchmarks: list[str]) -> None:
        """Fill the Training tab for ``name``: paper-based settings, a dataset for the problem, and the reasons."""
        problem = st.session_state.get("problem") or {}
        st.session_state["train_model"] = name
        dataset, dataset_reason = choose_dataset(name, problem.get("required_output"), problem.get("task", ""),
                                                 benchmarks)
        if dataset:
            st.session_state["train_dataset"] = dataset
        setup = bridge.training_setup(name, problem.get("num_nodes", 1024), problem.get("vram_gb", 8.0),
                                      problem.get("assumptions"))
        if setup:
            s = setup["setup"]
            st.session_state.update({
                "train_optimizer": s["optimizer"], "train_lr": s["learning_rate"], "train_momentum": s["momentum"],
                "train_weight_decay": s["weight_decay"], "train_schedule": s["lr_schedule"], "train_epochs": s["epochs"],
                "train_batch": s["batch_size"], "train_points": s["num_points"],
                "train_augment_rotation": s["augment_rotation"] and not (dataset and DATASETS[dataset].mesh),
                "train_model_config": json.dumps(s["model_config"]),
            })
        st.session_state["train_reasons"] = {"model": name, "dataset": dataset, "dataset_reason": dataset_reason,
                                             "setup": setup}
        st.toast(f"{name} selected in the Training tab" + (f" with the {dataset} dataset." if dataset else "."))

    @st.cache_data(ttl=10, show_spinner=False)
    def cached_results(tracking_uri: str, run_id: str | None, status: str) -> dict[str, Any]:
        """run_id and status are cache keys only: a run finishing invalidates the cached report."""
        return bridge.benchmark_results(tracking_uri)

    run_state = trainer.get_status()
    measured = cached_results(TrainingConfig().tracking_uri, run_state["run_id"], run_state["status"])

    # ------------------------------------------------------------ header
    st.title(":material/deployed_code: Neural3DBench")
    st.markdown("Pick, check and train 3D deep learning models: **advisor → data → training → inspection → results**.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registered models", len({ModelRegistry.get(n) for n in model_names}), border=True, icon=":material/hub:",
              help="Aliases such as pointnet++ are counted once.")
    k2.metric("Supported tasks", len(TASKS), border=True, icon=":material/task_alt:",
              help="Not covered yet: " + ", ".join(t.replace("_", " ") for t in UNCOVERED_TASKS) + ".")
    k3.metric("Datasets", len(DATASETS), border=True, icon=":material/database:")
    k4.metric("Finished runs", len(measured["runs"]), border=True, icon=":material/monitoring:")
    if bridge.source != "advisor":
        st.warning(f"Advisor unavailable, using the registry fallback: {bridge.error}")

    # ------------------------------------------------------------ sidebar
    with st.sidebar:
        st.subheader(":material/dns: System")
        if torch.cuda.is_available():
            st.badge(f"GPU · {torch.cuda.get_device_name(0)}", icon=":material/memory:", color="green")
        else:
            st.badge("CPU only", icon=":material/memory:", color="orange")
        st.badge("Advisor online" if bridge.source == "advisor" else "Advisor: registry fallback",
                 icon=":material/psychology:",
                 color="green" if bridge.source == "advisor" else "orange")

        @st.fragment(run_every=POLL_SECONDS if run_state["status"] == "training" else None)
        def sidebar_run() -> None:
            current, config = trainer.get_status(), trainer.config
            st.subheader(":material/model_training: Current run")
            if config is None:
                st.caption("No run yet.")
                return
            colors = {"training": "blue", "completed": "green", "failed": "red", "idle": "gray"}
            st.badge(current["status"], color=colors.get(current["status"], "gray"))
            st.markdown(f"**{config.model_name}** on `{config.dataset}`")
            st.progress(current["current_epoch"] / max(current["total_epochs"], 1),
                        text=f"epoch {current['current_epoch']} / {current['total_epochs']}")
            if current["val_loss"] is not None:
                st.caption(f"val loss {current['val_loss']:.4g}")

        sidebar_run()
        if measured["best_per_dataset"]:
            st.subheader(":material/emoji_events: Best so far")
            for row in measured["best_per_dataset"]:
                st.markdown(f"**{row['model']}** · {row['primary_metric'].removeprefix('val_')} {row['score']:.3g}  \n"
                            f":gray[{row['dataset']}]")
        st.divider()
        st.link_button("MLflow UI", MLFLOW_UI_URL, icon=":material/open_in_new:", width="stretch")
        st.link_button("Add a network", "https://github.com/n94kholdi/neural3dbench/blob/main/docs/adding_a_network.md",
                       icon=":material/add_circle:", width="stretch")

    advisor_tab, preview_tab, train_tab, inspect_tab, results_tab = st.tabs([
        ":material/psychology: Problem & Advisor",
        ":material/view_in_ar: Data",
        ":material/model_training: Training",
        ":material/troubleshoot: Inspection",
        ":material/leaderboard: Results",
    ])

    # ------------------------------------------------------------ 1. advisor
    with advisor_tab:
        c1, c2, c3, c4 = st.columns(4)
        task = c1.selectbox("Task domain", TASKS + UNCOVERED_TASKS, format_func=lambda t: t.replace("_", " "),
                            help="The questions below change with the task.")
        representation = c2.selectbox("Data representation", list(REPRESENTATIONS), format_func=REPRESENTATIONS.get)
        num_nodes = c3.number_input("Points / vertices per sample", min_value=1, value=10_000, step=1_000)
        vram_gb = c4.number_input("Target GPU memory (GB)", min_value=0.5, value=8.0, step=1.0)
        schema = problem_questions(task)
        with st.container(border=True):
            st.markdown(f"**Describe your {task.replace('_', ' ')} problem**")
            domains = st.multiselect("Application domain", schema["domains"], key=f"domains_{task}",
                                     placeholder="Search domains...",
                                     help="Pick one or more. Boosts models built for this domain. "
                                          "Not listed? Use Extra keywords below.")
            answers: dict[str, Any] = {}
            choices = [q for q in schema["questions"] if q["kind"] == "choice"]
            flags = [q for q in schema["questions"] if q["kind"] == "flag"]
            for col, q in zip(st.columns(max(len(choices), 1)), choices):
                answers[q["key"]] = col.radio(q["label"], list(q["options"]), key=f"q_{task}_{q['key']}",
                                              help=q["help"] or None)
            for col, q in zip(st.columns(max(len(flags), 1)), flags):
                answers[q["key"]] = col.checkbox(q["label"], key=f"q_{task}_{q['key']}", help=q["help"] or None)
            if not schema["questions"]:
                st.caption("No registered model covers this task yet; the advisor will say what to add.")
            extra = st.text_input("Extra keywords (optional)", placeholder="e.g. airfoil, darcy, modelnet")
        problem = compose_problem(task, answers, domains, extra)
        target_metric = st.radio("Optimize for", ("accuracy", "speed"), horizontal=True)
        with st.expander("Pre-flight checks & training budget"):
            p1, p2, p3 = st.columns(3)
            physics = task in ("surrogate_simulation", "pde_operator", "field_fitting")
            symmetries = p2.multiselect("Required symmetries", SYMMETRIES, default=["permutation"])
            assumptions = {
                **{k: v for k, v in problem["assumptions"].items() if k != "add_symmetries"},
                "boundary_conditions": p1.selectbox("Boundary conditions", BOUNDARY_CONDITIONS) if physics else "none",
                "required_symmetries": sorted(set(symmetries) | set(problem["assumptions"]["add_symmetries"])),
                "target_gpu": p1.selectbox("GPU for the time estimate", list(GPUS), index=list(GPUS).index("RTX4090")),
                "dataset_size": p2.number_input("Training samples", min_value=1, value=1000, step=100),
                "epochs": p3.number_input("Epochs", min_value=1, value=100, step=10),
            }
            if not problem["assumptions"]["evaluate_at_unseen_resolution"]:
                assumptions["evaluate_at_unseen_resolution"] = p3.checkbox("Evaluate at an unseen resolution")
            if problem["assumptions"]["add_symmetries"]:
                st.caption(f"Added from your answers: {', '.join(problem['assumptions']['add_symmetries'])} symmetry.")
        submitted = st.button("Get recommendations", key="get_recommendations", type="primary",
                              icon=":material/auto_awesome:")
        description = problem["description"]
        if submitted:
            try:
                st.session_state["recommendations"] = bridge.get_recommendations(
                    description, representation, int(num_nodes), vram_gb, target_metric,
                    dynamic_topology=problem["dynamic_topology"], required_output=problem["required_output"],
                )
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
            else:
                no_fit = st.session_state["recommendations"].get("verdict") == "no_fit"
                st.session_state["problem"] = None if no_fit else {
                    "task": task, "required_output": problem["required_output"], "num_nodes": int(num_nodes),
                    "vram_gb": float(vram_gb), "assumptions": assumptions,
                }
                st.session_state["problem_context"] = (description, int(num_nodes), assumptions)

        result = st.session_state.get("recommendations")
        if result:
            verdict = result.get("verdict", "fit")
            if verdict == "no_fit":
                st.error(f"**Not supported yet.** {result['reason']}", icon=":material/block:")
                for net in result.get("suggested_networks", []):
                    st.markdown(f"- **{net['name']}**: {net['why']} [paper]({net['reference']})")
                if result.get("suggested_networks"):
                    st.caption("Add one of these as a new architecture (see `docs/adding_a_network.md`); "
                               "the advisor picks it up from the registry and catalog.yaml.")
                if result.get("suggested_datasets"):
                    st.markdown("**Datasets to add:** " + ", ".join(result["suggested_datasets"]))
            elif verdict == "weak_fit":
                st.warning(result["reason"], icon=":material/report:")
            else:
                st.success(f"**{len(result['recommendations'])} models fit this problem**, ranked by the advisor.",
                           icon=":material/check_circle:")
            st.caption(f"{result.get('note', '')} Considered {result.get('considered', 0)} models; source: {result['source']}.")
            st.caption(f"Advisor read the problem as: `{st.session_state.get('problem_context', ('',))[0]}`")
            if not result["recommendations"]:
                st.info("No registered model matches this representation.")
            container = st.expander("Closest registered models (not a fit for this task)") if verdict == "no_fit" else st.container()
            problem_text, problem_nodes, problem_assumptions = st.session_state.get("problem_context", ("", 10_000, {}))
            best_measured = best_by(measured["runs"], "model")
            for rank, row in enumerate(result["recommendations"], 1):
                try:
                    check = bridge.preflight(row["model"], problem_text, problem_nodes, problem_assumptions)
                except (ValueError, KeyError) as exc:
                    check = {"failed": str(exc)}
                budget = (check or {}).get("budget")
                run = best_measured.get(row["model"])
                peak = row.get("estimated_training_peak_gb")
                with container, st.container(border=True):
                    st.markdown(f"#### #{rank} · {row.get('display_name', row['model'])}")
                    with st.container(horizontal=True, gap="small"):
                        fits = row.get("fits_vram")
                        if fits is True:
                            st.badge(f"fits {result['vram_gb']:g} GB", icon=":material/check:", color="green")
                        elif fits is False:
                            st.badge(f"needs ~{peak:.1f} GB", icon=":material/warning:", color="red")
                        if check and "failed" not in check:
                            if check["errors"]:
                                st.badge(f"pre-flight: {check['errors']} error(s)", icon=":material/error:", color="red")
                            elif check["warnings"]:
                                st.badge(f"pre-flight: {check['warnings']} warning(s)", icon=":material/warning:",
                                         color="orange")
                            else:
                                st.badge("pre-flight passed", icon=":material/verified:", color="green")
                        if run:
                            st.badge(f"measured {run['primary_metric'].removeprefix('val_')} {run['score']:.3g} "
                                     f"on {run['dataset']}", icon=":material/monitoring:", color="blue")
                        if row.get("complexity"):
                            st.badge(row["complexity"].replace("*", "×"), icon=":material/functions:", color="gray")

                    score, memory, hours = st.columns(3)
                    score.metric("Advisor score", row.get("score"))
                    memory.metric("Training peak", "n/a" if peak is None else f"{peak:.2f} GB")
                    hours.metric(f"Train time ({budget['gpu']})" if budget else "Train time",
                                 (f"~{budget['total_hours']:.1f} h" if budget["total_hours"] >= 1
                                  else f"~{max(1, round(budget['total_hours'] * 60))} min") if budget else "n/a",
                                 help=None if not budget else
                                 f"Plausible range {budget['plausible_range_hours'][0]}–{budget['plausible_range_hours'][1]} h. "
                                 + budget["assumptions"])

                    evidence = row.get("evidence")
                    if evidence and evidence.get("paper"):
                        paper = evidence["paper"]
                        reported = "; ".join(f"{r['dataset']} {r['metric']} {r['value']}" for r in evidence["reported"])
                        st.caption(f":material/article: [{paper['title']}]({paper['url']}) ({paper['venue']})"
                                   + (f". Paper reports: {reported}." if reported else ".")
                                   + " Paper numbers, not reproduced here.")
                    why, risks = st.columns(2)
                    if row.get("rationale"):
                        why.markdown("**Why it fits**\n" + "\n".join(f"- {line}" for line in row["rationale"]))
                    if row.get("pitfalls"):
                        risks.markdown("**Watch out**\n" + "\n".join(f"- :orange[{line}]" for line in row["pitfalls"]))

                    left, middle, right = st.columns(3)
                    with left.popover("Model dossier", icon=":material/menu_book:", width="stretch"):
                        try:
                            dossier = bridge.get_dossier(row["model"])
                        except KeyError as exc:
                            st.error(str(exc))
                        else:
                            if dossier.get("doc_summary"):
                                st.write(dossier["doc_summary"])
                            if dossier.get("evidence"):
                                ev = dossier["evidence"]
                                st.markdown(f"**Paper**: [{ev['paper']['title']}]({ev['paper']['url']}) "
                                            f"({ev['paper']['venue']}); checked {ev.get('checked', '')}")
                                if ev.get("reported"):
                                    st.dataframe(ev["reported"], hide_index=True, width="stretch")
                                st.markdown(f"**This implementation vs the paper**: {ev.get('implementation_notes', '')}")
                            for title in ("merits", "demerits", "inductive_biases"):
                                if dossier.get(title):
                                    st.markdown(f"**{title.replace('_', ' ').title()}**")
                                    st.markdown("\n".join(f"- {item}" for item in dossier[title]))
                            st.markdown("**Full dossier**")
                            st.json(dossier, expanded=False)
                    if check and "failed" not in check:
                        with middle.popover("Pre-flight details", icon=":material/fact_check:", width="stretch"):
                            icons = {"error": ":material/error:", "warning": ":material/warning:",
                                     "info": ":material/info:", "ok": ":material/check_circle:"}
                            colors = {"error": "red", "warning": "orange", "info": "blue", "ok": "green"}
                            if not check["findings"]:
                                st.markdown(":green[:material/check_circle:] No assumption issues for the selected "
                                            "physics and data.")
                            for finding in check["findings"]:
                                color = colors.get(finding["severity"], "gray")
                                st.markdown(f":{color}[{icons.get(finding['severity'], '')}] "
                                            f"**{finding['check']}**: {finding['message']}")
                            if check["symmetry"]:
                                st.markdown("**Symmetries**")
                                for v in check["symmetry"]["verdicts"]:
                                    mark = (":green[:material/check_circle:]" if v["guaranteed_by_architecture"]
                                            else ":orange[:material/warning:]")
                                    st.markdown(f"{mark} **{v['symmetry']}**: {v['action']}")
                                for warning in check["symmetry"]["warnings"]:
                                    st.caption(warning)
                            for warning in check["budget"]["warnings"]:
                                st.caption(f"⚠ {warning}")
                    elif check:
                        middle.caption(f"Pre-flight unavailable: {check['failed']}")
                    benchmarks = (evidence or {}).get("benchmark_datasets", [])
                    right.button("Use for training", key=f"use_{row['model']}", on_click=use_model,
                                 args=(row["model"], benchmarks),
                                 icon=":material/play_arrow:", type="primary" if rank == 1 else "secondary",
                                 width="stretch", disabled=verdict == "no_fit" or not row.get("supports_output", True))

        with st.expander("Estimate VRAM", expanded=not result):
            c1, c2, c3, c4 = st.columns(4)
            vram_model = c1.selectbox("Model", model_names, key="vram_model")
            vram_nodes = c2.number_input("Points / vertices", min_value=1, value=10_000, step=1_000, key="vram_nodes")
            vram_batch = c3.number_input("Batch size", min_value=1, value=1, key="vram_batch")
            precision = c4.selectbox("Precision", PRECISIONS, key="vram_precision")
            try:
                estimate = bridge.estimate_memory(vram_model, int(vram_nodes), int(vram_batch), precision)
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
            else:
                memory = estimate["memory_gb"]
                m1, m2, m3 = st.columns(3)
                m1.metric("Training peak", f"{memory['training_peak']:.3f} GB")
                m2.metric("Inference peak", "n/a" if memory.get("inference_peak") is None else f"{memory['inference_peak']:.3f} GB")
                m3.metric("Parameters", "n/a" if estimate.get("parameter_count") is None else f"{estimate['parameter_count']:,}")
                for warning in estimate.get("warnings", []):
                    st.caption(f"⚠ {warning}")
                if estimate.get("assumptions"):
                    st.caption(estimate["assumptions"])

    # ------------------------------------------------------------ 2. data
    with preview_tab:
        mode = st.radio("Data", ("Preview geometry", "Download a dataset", "Upload a labelled dataset"),
                        horizontal=True, label_visibility="collapsed")
        if mode == "Preview geometry":
            source = st.radio("Source", ("Repository sample", "Upload file"), horizontal=True)
            geometry, label = None, ""
            try:
                if source == "Upload file":
                    upload = st.file_uploader("3D file", type=["obj", "ply", "pt"])
                    if upload is not None:
                        geometry, label = load_geometry(upload.name, upload.getvalue()), upload.name
                else:
                    c1, c2 = st.columns([3, 1])
                    sample = c1.selectbox("Sample", list(SAMPLES), format_func=SAMPLES.get)
                    points = c2.number_input("Points", min_value=64, max_value=50_000, value=4096, step=512)
                    geometry, label = cached_sample(sample, int(points)), SAMPLES[sample]
            except Exception as exc:  # noqa: BLE001 - any parse error is a user-facing message, not a crash
                st.error(f"Could not load geometry: {exc}")
            if geometry is not None:
                vertices, faces, field = geometry["vertices"], geometry["faces"], geometry["field"]
                st.caption(f"{len(vertices):,} vertices" + (f", {len(faces):,} triangles" if faces is not None else "")
                           + (f", field with {field.shape[1]} channel(s)" if field is not None else ""))
                if geometry["labels"] is not None:
                    figure = plot_labelled_points(vertices, geometry["labels"], title=label)
                elif faces is not None:
                    figure = plot_mesh(vertices, faces, field, title=label)
                else:
                    figure = plot_point_cloud(vertices, field, title=label)
                st.plotly_chart(figure, width="stretch")

        elif mode == "Download a dataset":
            st.caption(f"Archives and converted datasets are cached in `{DATA_DIR}` (set `NEURAL3DBENCH_DATA` to move it).")
            wanted = (st.session_state.get("problem") or {}).get("required_output")
            relevant = [d for d in DOWNLOADS if wanted is None or DOWNLOAD_OUTPUTS[d] == wanted]
            if wanted and not relevant:
                benchmarks = sorted({b for row in (st.session_state.get("recommendations") or {}).get("recommendations", [])
                                     if row.get("supports_output") for b in (row.get("evidence") or {}).get("benchmark_datasets", [])})
                st.info("No download here is labelled for your problem's output, so all are listed. Datasets to add: "
                        + (", ".join(benchmarks) if benchmarks else "none recorded") + ".", icon=":material/info:")
            elif wanted and not st.checkbox("Show all downloads", key="download_show_all",
                                            help="Only downloads labelled for your problem's output are listed."):
                pass
            else:
                relevant = list(DOWNLOADS)
            source = st.selectbox("Dataset", relevant or list(DOWNLOADS), format_func=DOWNLOADS.get)
            c1, c2, c3 = st.columns(3)
            if source in MESH_DOWNLOADS:
                c1.caption("Meshes are kept whole (no point sampling), decimated to at most 2048 faces. "
                           "Train them with meshnet, diffusionnet or meshwalker.")
            elif source == "shapenet_part":
                category = c1.selectbox("Category", [*SHAPENET_PART_RANGES, "All"])
                points = c2.number_input("Points per shape", 256, 8192, 2048, step=256)
                limit = c3.number_input("Max shapes (0 = all)", 0, 20_000, 500, step=100)
            else:
                points = c2.number_input("Points per shape", 256, 8192, 1024, step=256)
                limit = c3.number_input("Max meshes per class (0 = all)", 0, 1_000, 100, step=50)
            if st.button("Download and import", type="primary"):
                bar = st.progress(0.0, text="Starting...")

                def report(fraction: float, message: str) -> None:
                    bar.progress(fraction, text=message)

                try:
                    if source in MESH_DOWNLOADS:
                        load_mesh_dataset(MESH_DOWNLOADS[source], "train", progress=report)
                        dataset = DATASETS[source]
                    elif source == "shapenet_part":
                        dataset = import_shapenet_part(category, int(points), int(limit) or None, progress=report)
                    else:
                        dataset = import_modelnet(int(source[-2:]), int(points), int(limit) or None, progress=report)
                except Exception as exc:  # noqa: BLE001 - network/archive errors are user-facing messages
                    st.error(f"Import failed: {exc}")
                else:
                    st.session_state["train_dataset"] = dataset.name
                    st.success(f"Registered `{dataset.name}`: {dataset.description} Selected in the Training tab.")
            if source in MESH_DOWNLOADS and mesh_cache_path(MESH_DOWNLOADS[source], "train").is_file():
                meshes, labels, class_names = load_mesh_dataset(MESH_DOWNLOADS[source], "train")
                st.caption(f"`{source}` is ready: {len(meshes)} training meshes, {len(class_names)} classes.")
                index = st.slider("Mesh", 0, len(meshes) - 1, 0, key=f"mesh_preview_{source}")
                vertices, faces = meshes[index]
                st.plotly_chart(plot_mesh(vertices, faces, title=f"{class_names[labels[index]]}: "
                                          f"{len(vertices):,} vertices, {len(faces):,} faces"), width="stretch")

        else:
            st.caption("One `.npz` with `coords [B, N, 3]` and `labels` (`[B, N]` part ids or `[B]` class ids) or a "
                       "float `target [B, N, C]`, optionally `class_names`; or one ASCII `.ply` per shape with a "
                       "per-point `label` property.")
            files = st.file_uploader("Labelled files", type=["npz", "ply"], accept_multiple_files=True)
            if files:
                c1, c2 = st.columns(2)
                name = c1.text_input("Dataset name", value=suggest_name(files[0].name))
                points = c2.number_input("Points per shape (.ply files are resampled)", 64, 100_000, 2048, step=256)
                try:
                    parsed = parse_labelled_upload([(f.name, f.getvalue()) for f in files], int(points))
                    coords, labels = np.asarray(parsed["coords"]), parsed["labels"]
                    st.caption(f"{coords.shape[0]} samples x {coords.shape[1]} points"
                               + (f", {int(labels.max()) + 1} classes" if labels is not None else ", float target"))
                    if labels is not None and labels.ndim == 2:
                        st.plotly_chart(plot_labelled_points(coords[0], labels[0], parsed["class_names"],
                                                             title="Sample 0 by label"), width="stretch")
                except Exception as exc:  # noqa: BLE001 - any parse error is a user-facing message
                    st.error(f"Could not read the upload: {exc}")
                else:
                    if st.button("Save dataset", type="primary"):
                        try:
                            dataset = save_npz_dataset(name, coords, labels=labels, target=parsed["target"],
                                                       class_names=parsed["class_names"],
                                                       description=f"Uploaded from {files[0].name}"
                                                       + (f" and {len(files) - 1} more" if len(files) > 1 else "") + ".")
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            st.session_state["train_dataset"] = dataset.name
                            st.success(f"Saved `{dataset.name}` ({dataset.task.replace('_', ' ')}); name its classes below.")

        saved = [n for n, d in DATASETS.items() if n not in BUILTIN_DATASETS and d.task != "shape_completion"]
        if mode != "Preview geometry" and saved:
            st.subheader("Saved datasets: class names and colors")
            pick = st.selectbox("Saved dataset", saved, key="edit_dataset")
            chosen = DATASETS[pick]
            st.caption(f"{chosen.task.replace('_', ' ')}: {chosen.description}")
            if chosen.class_names:
                import pandas as pd

                table = pd.DataFrame({
                    "id": range(chosen.num_classes),
                    "name": list(chosen.class_names),
                    "color": [class_color(i, chosen.class_colors) for i in range(chosen.num_classes)],
                })
                edited = st.data_editor(table, disabled=["id"], hide_index=True, key=f"classes_{pick}",
                                        column_config={"color": st.column_config.TextColumn("color (hex)")})
                if st.button("Save classes"):
                    try:
                        update_classes(pick, edited["name"].astype(str).tolist(), edited["color"].astype(str).tolist())
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.success("Classes saved.")
                        st.rerun()
                coords, target = make_dataset(pick, 1, 4096)
                if chosen.task == "part_segmentation":
                    st.plotly_chart(plot_labelled_points(coords[0], target[0], chosen.class_names, chosen.class_colors,
                                                         title="A random sample by part"), width="stretch")
                else:
                    st.plotly_chart(plot_point_cloud(coords[0], title=f"A random sample: {chosen.class_names[int(target[0])]}"),
                                    width="stretch")

    # ------------------------------------------------------------ 3. training
    with train_tab:
        status = trainer.get_status()
        running = status["status"] == "training"
        problem = st.session_state.get("problem") or {}
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Model", model_names, key="train_model", disabled=running)
        show_all = st.session_state.get("train_show_all", False)
        fitting = cached_compatible(st.session_state["train_model"], problem.get("required_output"), len(DATASETS))
        options = sorted(DATASETS) if show_all or not fitting else fitting
        if st.session_state["train_dataset"] not in options:
            st.session_state["train_dataset"] = options[0]
        c2.selectbox("Dataset", options, key="train_dataset", disabled=running,
                     format_func=lambda d: f"{d} ({DATASETS[d].task.replace('_', ' ')})",
                     help="Only datasets this model can train on" + (" that give your problem's output"
                          if problem.get("required_output") else "") + ". Tick 'Show all datasets' to override.")
        c3.text_input("Experiment name", key="train_experiment", disabled=running)
        c1.number_input("Epochs", min_value=1, max_value=10_000, key="train_epochs", disabled=running)
        c2.number_input("Learning rate", min_value=1e-7, max_value=1.0, format="%.1e", step=1e-4, key="train_lr", disabled=running)
        c3.number_input("Batch size", min_value=1, max_value=4096, key="train_batch", disabled=running)
        chosen = DATASETS[st.session_state["train_dataset"]]
        st.caption(f"**{chosen.task.replace('_', ' ')}**: {chosen.description}"
                   + (f" Classes: {', '.join(chosen.class_names)}." if chosen.class_names else "")
                   + " The model head (`output_dim`, `task`) is sized from the dataset; JSON overrides win.")
        c2.checkbox("Show all datasets", key="train_show_all", disabled=running)
        blocked = not fitting and not show_all
        if not fitting:
            st.warning(choose_dataset(st.session_state["train_model"], problem.get("required_output"),
                                      problem.get("task", ""))[1]
                       + " Training is disabled; tick *Show all datasets* to train on an unrelated dataset anyway.")
        why = st.session_state.get("train_reasons")
        if why and why["model"] == st.session_state["train_model"]:
            with st.expander(f"Why this training setup for {why['model']}", expanded=True, icon=":material/lightbulb:"):
                st.markdown(f"**Dataset**: {why['dataset_reason']}")
                setup = why.get("setup")
                if setup:
                    paper = setup["paper"]
                    if paper:
                        st.markdown(f"**Recipe source**: [{paper['title']}]({paper['url']}) ({paper['venue']}), "
                                    f"checked {setup['checked']}.")
                    st.markdown("\n".join(f"- {r}" for r in setup["reasons"]))
                    for w in setup["warnings"]:
                        st.caption(f"⚠ {w}")
                st.caption("Edits below override these suggestions.")
        mismatch = cached_input_mismatch(st.session_state["train_model"], chosen.name)
        if mismatch:
            st.warning(mismatch)
        if chosen.mesh and st.session_state["train_samples"] < 10 * chosen.num_classes:
            st.warning(f"{st.session_state['train_samples']} samples for {chosen.num_classes} classes is too few to learn; "
                       f"raise Samples (Advanced) to {16 * chosen.num_classes} or more. For diffusionnet add "
                       '`{"input_features": "hks"}` to the config overrides (rotation invariant input).')
        with st.expander("Advanced"):
            a1, a2, a3, a4 = st.columns(4)
            a1.selectbox("Optimizer", ("adam", "adamw", "sgd"), key="train_optimizer", disabled=running)
            a2.number_input("Samples", min_value=2, max_value=100_000, key="train_samples", disabled=running)
            a3.number_input("Points per sample", min_value=8, max_value=200_000, key="train_points", disabled=running)
            devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
            if st.session_state["train_device"] not in devices:
                st.session_state["train_device"] = "cpu"
            a4.selectbox("Device", devices, key="train_device", disabled=running)
            a1.number_input("Momentum (SGD)", min_value=0.0, max_value=0.999, step=0.1, key="train_momentum",
                            disabled=running)
            a2.number_input("Weight decay", min_value=0.0, max_value=1.0, format="%.1e", step=1e-4,
                            key="train_weight_decay", disabled=running)
            a3.text_input("LR schedule", key="train_schedule", disabled=running,
                          help="none, cosine:<eta_min>, step:<every epochs>:<gamma> or multistep:<e1>,<e2>:<gamma>.")
            a4.checkbox("Random rotation augmentation", key="train_augment_rotation", disabled=running or chosen.mesh,
                        help="Rotates each training sample by a random 3D rotation (point datasets).")
        gpu_problem = cuda_problem()
        if gpu_problem:
            st.warning(gpu_problem)
        with st.expander("Model config overrides (JSON)"):
            st.text_area("Model config overrides (JSON)", key="train_model_config", disabled=running,
                         help='Passed to ModelRegistry.create, e.g. {"widths": [64, 64]}.')

        b1, b2, b3 = st.columns([1, 1, 2])
        if b1.button("Start Training", type="primary", icon=":material/play_arrow:", disabled=running or blocked,
                     width="stretch"):
            try:
                trainer.start_training(config_from_form(st.session_state))
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:  # bad JSON/config or a run in progress
                st.error(f"Could not start training: {exc}")
            else:
                st.rerun()
        if b2.button("Stop Training", icon=":material/stop:", disabled=not running, width="stretch"):
            trainer.stop_training(timeout=0)  # do not block the UI; the worker stops after this epoch
            st.toast("Stopping after the current epoch...")
        b3.link_button("Open MLflow UI", MLFLOW_UI_URL, icon=":material/open_in_new:", width="stretch")
        st.caption(f"Start the MLflow UI with `MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns` (serves {MLFLOW_UI_URL}).")

        @st.fragment(run_every=POLL_SECONDS if running else None)
        def live_monitor() -> None:
            current = trainer.get_status()
            config = trainer.config
            if config is None:
                st.info("No run yet. Configure and press Start Training.")
                return
            total = max(current["total_epochs"], 1)
            st.progress(current["current_epoch"] / total,
                        text=f"{current['status']}: epoch {current['current_epoch']} / {current['total_epochs']}")
            if current["status"] == "failed":
                st.error("Training failed.")
                st.code(current["error"] or "", language="text")
            if current["run_id"]:
                import pandas as pd

                client = mlflow_client(config.tracking_uri)
                history = {
                    metric: {m.step: m.value for m in client.get_metric_history(current["run_id"], metric)}
                    for metric in ("train_loss", "val_loss")
                }
                frame = pd.DataFrame(history).rename_axis("epoch").sort_index()
                if not frame.empty:
                    st.line_chart(frame)
                st.caption(f"MLflow run `{current['run_id']}` in experiment `{config.experiment_name}`.")
            if running and current["status"] != "training":
                st.rerun()  # full rerun: stop polling, unlock the form and refresh Tab 4

        live_monitor()

    # ------------------------------------------------------------ 4. inspection
    with inspect_tab:
        status, config = trainer.get_status(), trainer.config
        if status["status"] != "completed" or config is None or not status["checkpoint_path"]:
            st.info("Finish a training run in Tab 3 to inspect its predictions here.")
        else:
            run = mlflow_client(config.tracking_uri).get_run(status["run_id"])
            st.subheader(f"{config.model_name} on {config.dataset}" + (" (stopped early)" if status["stopped"] else ""))
            columns = st.columns(len(run.data.metrics) + 1)
            columns[0].metric("Epochs", f"{status['current_epoch']} / {status['total_epochs']}")
            for column, (name, value) in zip(columns[1:], sorted(run.data.metrics.items())):
                column.metric(f"Final {name}", f"{value:.4g}")
            with st.expander("Run parameters"):
                st.dataframe({"param": list(run.data.params), "value": list(run.data.params.values())},
                             width="stretch", hide_index=True)

            sample = st.slider("Validation sample", 0, num_validation_samples(config) - 1, 0)
            try:
                coords, gt, pred = cached_prediction(status["checkpoint_path"], json.dumps(asdict(config)), sample)
                st.plotly_chart(evaluation_figure(DATASETS[config.dataset], coords, gt, pred), width="stretch")
            except Exception as exc:  # noqa: BLE001 - checkpoint/model mismatch is shown, not raised
                st.error(f"Could not run the trained model: {exc}")

            def relaunch() -> None:
                st.session_state.update(form_values(config))
                st.toast("Training form filled from the last run; edit it in Tab 3 and press Start.")

            st.button("Launch New Run with Modified Params", on_click=relaunch, type="primary", icon=":material/replay:")

    # ------------------------------------------------------------ 5. results
    with results_tab:
        import pandas as pd

        st.caption(measured["note"])
        if not measured["runs"]:
            st.info("No finished runs yet. Train a model in Tab 3; its final validation metrics appear here.")
        else:
            st.subheader(":material/emoji_events: Leaderboard: best model per dataset")
            with st.container(horizontal=True, gap="medium"):
                for row in measured["best_per_dataset"]:
                    st.metric(row["dataset"], row["model"], help=(row["task"] or "task not logged").replace("_", " "),
                              delta=f"{row['primary_metric'].removeprefix('val_')} {row['score']:.4g}",
                              delta_color="off", delta_arrow="off", border=True, width=260)
            st.divider()

            runs = pd.DataFrame(measured["runs"])
            pick = st.selectbox("Dataset", sorted(runs["dataset"].dropna().unique()), key="results_dataset")
            subset = runs[runs["dataset"] == pick]
            metric, higher = subset["primary_metric"].iloc[0], bool(subset["higher_is_better"].iloc[0])
            per_model = subset.groupby("model")["score"].agg("max" if higher else "min").sort_values(ascending=not higher)
            st.markdown(f"**Best {metric} per model on `{pick}`** ({'higher' if higher else 'lower'} is better)")
            st.bar_chart(per_model, horizontal=True)

            columns = ["model", "score", "epochs", "num_parameters", "duration_s", "device", "stopped_early",
                       "experiment", "run_id"]
            st.markdown(f"**All finished runs on `{pick}`**")
            st.dataframe(subset[columns].rename(columns={"score": metric}), hide_index=True, width="stretch")

            chosen_runs = st.multiselect("Compare learning curves", subset["run_id"].tolist(),
                                         default=subset["run_id"].tolist()[:3],
                                         format_func=lambda r: f"{subset.set_index('run_id').at[r, 'model']} · {r[:8]}")
            if chosen_runs:
                client = mlflow_client(TrainingConfig().tracking_uri)
                curves = pd.DataFrame({
                    f"{subset.set_index('run_id').at[r, 'model']} · {r[:8]}":
                        {m.step: m.value for m in client.get_metric_history(r, metric)}
                    for r in chosen_runs
                }).rename_axis("epoch").sort_index()
                st.line_chart(curves, y_label=metric)

            flat = runs.drop(columns=["metrics"]).join(pd.json_normalize(runs["metrics"].tolist()))
            st.download_button("Download all runs (CSV)", flat.to_csv(index=False), "neural3dbench_runs.csv", "text/csv",
                               icon=":material/download:")


if __name__ == "__main__":
    main()
