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
}

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
    x = coords[sample : sample + 1]
    with torch.no_grad():
        pred = tasks.forward(tasks.DATASETS[config.dataset].task, model, x)
    return x[0].numpy(), target[sample].numpy(), pred[0].numpy()


def num_validation_samples(config: Any) -> int:
    return max(1, int(round(config.num_samples * config.val_fraction)))


# ---------------------------------------------------------------- UI


def main() -> None:
    import streamlit as st

    from models import ModelRegistry
    from tools.dashboard.advisor_bridge import AdvisorBridge
    from tools.dashboard.data_sources import SHAPENET_PART_RANGES, import_modelnet, import_shapenet_part, suggest_name
    from tools.dashboard.tasks import (
        BUILTIN_DATASETS,
        DATA_DIR,
        DATASETS,
        dataset_for_problem,
        evaluation_figure,
        input_mismatch,
        make_dataset,
        save_npz_dataset,
        update_classes,
    )
    from tools.dashboard.trainer import Trainer, TrainingConfig
    from tools.dashboard.visualizer import class_color, plot_labelled_points, plot_mesh, plot_point_cloud

    try:  # tools.mcp_advisor needs pyyaml
        from tools.mcp_advisor.guardrails import BOUNDARY_CONDITIONS, GPUS, SYMMETRIES, TARGET_KINDS
        from tools.mcp_advisor.results import best_by
    except ImportError:  # registry fallback: no pre-flight checks or measured results
        BOUNDARY_CONDITIONS, GPUS, SYMMETRIES, TARGET_KINDS = ("none",), {"RTX4090": None}, (), ("scalar",)

        def best_by(rows: list, key: str) -> dict:
            return {}

    st.set_page_config(page_title="Neural3DBench Dashboard", layout="wide")

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
    def cached_input_mismatch(model_name: str, dataset_name: str) -> str | None:
        return input_mismatch(model_name, dataset_name)

    @st.cache_data(show_spinner="Running the model on the validation sample...")
    def cached_prediction(checkpoint_path: str, config_json: str, sample: int) -> tuple[np.ndarray, ...]:
        return predict_sample(checkpoint_path, TrainingConfig(**json.loads(config_json)), sample)

    bridge, trainer = get_bridge(), get_trainer()
    model_names = sorted(ModelRegistry._registry)
    for key, value in form_values(TrainingConfig(device=default_device())).items():
        st.session_state.setdefault(key, value)

    def use_model(name: str) -> None:
        st.session_state["train_model"] = name
        dataset = st.session_state.get("problem_dataset")
        if dataset:
            st.session_state["train_dataset"] = dataset
        st.toast(f"{name} selected in the Training tab" + (f" with the {dataset} dataset." if dataset else "."))

    @st.cache_data(ttl=10, show_spinner=False)
    def cached_results(tracking_uri: str, run_id: str | None, status: str) -> dict[str, Any]:
        """run_id and status are cache keys only: a run finishing invalidates the cached report."""
        return bridge.benchmark_results(tracking_uri)

    st.title("Neural3DBench Dashboard")
    run_state = trainer.get_status()
    measured = cached_results(TrainingConfig().tracking_uri, run_state["run_id"], run_state["status"])
    st.caption(f"{len(model_names)} registered models · {len(TASKS)} supported tasks · {len(DATASETS)} datasets · "
               f"{len(measured['runs'])} finished runs · advisor: {bridge.source}")
    if bridge.source != "advisor":
        st.warning(f"Advisor unavailable, using the registry fallback: {bridge.error}")
    advisor_tab, preview_tab, train_tab, inspect_tab, results_tab = st.tabs(
        ["1 · Problem & Advisor", "2 · Data", "3 · Training", "4 · Inspection", "5 · Results"]
    )

    # ------------------------------------------------------------ 1. advisor
    with advisor_tab:
        with st.form("problem"):
            c1, c2, c3, c4 = st.columns(4)
            task = c1.selectbox("Task domain", TASKS + UNCOVERED_TASKS, format_func=lambda t: t.replace("_", " "))
            representation = c2.selectbox("Data representation", list(REPRESENTATIONS), format_func=REPRESENTATIONS.get)
            num_nodes = c3.number_input("Points / vertices per sample", min_value=1, value=10_000, step=1_000)
            vram_gb = c4.number_input("Target GPU memory (GB)", min_value=0.5, value=8.0, step=1.0)
            description = st.text_input("Problem description (optional)", placeholder="e.g. CFD pressure on car surfaces")
            target_metric = st.radio("Optimize for", ("accuracy", "speed"), horizontal=True)
            with st.expander("Physics & data assumptions (pre-flight checks)"):
                p1, p2, p3 = st.columns(3)
                assumptions = {
                    "boundary_conditions": p1.selectbox("Boundary conditions", BOUNDARY_CONDITIONS),
                    "target_kind": p2.selectbox("Target kind", TARGET_KINDS,
                                                help="Vector/tensor targets need rotation equivariance, not invariance."),
                    "required_symmetries": p3.multiselect("Required symmetries", SYMMETRIES, default=["permutation"]),
                    "topology_changes": p1.checkbox("Topology changes between samples"),
                    "geometry_varies": p2.checkbox("Geometry varies between samples"),
                    "evaluate_at_unseen_resolution": p3.checkbox("Evaluate at an unseen resolution"),
                    "target_gpu": p1.selectbox("GPU for the time estimate", list(GPUS), index=list(GPUS).index("RTX4090")),
                    "dataset_size": p2.number_input("Training samples", min_value=1, value=1000, step=100),
                    "epochs": p3.number_input("Epochs", min_value=1, value=100, step=10),
                }
            submitted = st.form_submit_button("Get recommendations", type="primary")
        if submitted:
            try:
                st.session_state["recommendations"] = bridge.get_recommendations(
                    f"{task} {description}", representation, int(num_nodes), vram_gb, target_metric
                )
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
            else:
                no_fit = st.session_state["recommendations"].get("verdict") == "no_fit"
                st.session_state["problem_dataset"] = None if no_fit else dataset_for_problem(task, description)
                st.session_state["problem_context"] = (f"{task} {description}", int(num_nodes), assumptions)

        result = st.session_state.get("recommendations")
        if result:
            verdict = result.get("verdict", "fit")
            if verdict == "no_fit":
                st.error(f"**Not supported yet.** {result['reason']}")
                for net in result.get("suggested_networks", []):
                    st.markdown(f"- **{net['name']}**: {net['why']} [paper]({net['reference']})")
                if result.get("suggested_networks"):
                    st.caption("Add one of these as a new architecture (see `docs/adding_a_network.md`); "
                               "the advisor picks it up from the registry and catalog.yaml.")
            elif verdict == "weak_fit":
                st.warning(result["reason"])
            st.caption(f"{result.get('note', '')} Considered {result.get('considered', 0)} models; source: {result['source']}.")
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
                with container, st.container(border=True):
                    head, score, memory, fits, hours = st.columns([3, 1, 1, 1, 1])
                    head.subheader(f"#{rank} {row.get('display_name', row['model'])}")
                    score.metric("Score", row.get("score"))
                    peak = row.get("estimated_training_peak_gb")
                    memory.metric("Train peak", "n/a" if peak is None else f"{peak:.2f} GB")
                    fits.metric("Fits VRAM", {True: "yes", False: "no", None: "n/a"}[row.get("fits_vram")])
                    budget = (check or {}).get("budget")
                    hours.metric(f"Train time ({budget['gpu']})" if budget else "Train time",
                                 f"~{budget['total_hours']:.1f} h" if budget else "n/a",
                                 help=None if not budget else
                                 f"Plausible range {budget['plausible_range_hours'][0]}–{budget['plausible_range_hours'][1]} h. "
                                 + budget["assumptions"])
                    for line in row.get("rationale", []):
                        st.markdown(f"- {line}")
                    for line in row.get("pitfalls", []):
                        st.markdown(f"- :orange[{line}]")
                    if row.get("complexity"):
                        st.caption(f"Complexity: `{row['complexity']}`")
                    run = best_measured.get(row["model"])
                    if run:
                        st.caption(f"Measured: best {run['primary_metric']} = {run['score']:.4g} on `{run['dataset']}` "
                                   f"({run['epochs']} epochs). See tab 5.")
                    left, middle, right = st.columns(3)
                    if check and "failed" not in check:
                        label = (f":red[Pre-flight: {check['errors']} error(s)]" if check["errors"]
                                 else f":orange[Pre-flight: {check['warnings']} warning(s)]" if check["warnings"]
                                 else ":green[Pre-flight: passed]")
                        with middle.popover(label, width="stretch"):
                            icons = {"error": "🔴", "warning": "🟠", "info": "🔵", "ok": "🟢"}
                            if not check["findings"]:
                                st.markdown("🟢 No assumption issues for the selected physics and data.")
                            for finding in check["findings"]:
                                st.markdown(f"{icons.get(finding['severity'], '')} **{finding['check']}**: {finding['message']}")
                            if check["symmetry"]:
                                st.markdown("**Symmetries**")
                                for v in check["symmetry"]["verdicts"]:
                                    mark = "🟢" if v["guaranteed_by_architecture"] else "🟠"
                                    st.markdown(f"{mark} **{v['symmetry']}**: {v['action']}")
                                for warning in check["symmetry"]["warnings"]:
                                    st.caption(warning)
                            for warning in check["budget"]["warnings"]:
                                st.caption(f"⚠ {warning}")
                    elif check:
                        middle.caption(f"Pre-flight unavailable: {check['failed']}")
                    with left.popover("Model dossier", width="stretch"):
                        try:
                            dossier = bridge.get_dossier(row["model"])
                        except KeyError as exc:
                            st.error(str(exc))
                        else:
                            if dossier.get("doc_summary"):
                                st.write(dossier["doc_summary"])
                            for title in ("merits", "demerits", "inductive_biases"):
                                if dossier.get(title):
                                    st.markdown(f"**{title.replace('_', ' ').title()}**")
                                    st.markdown("\n".join(f"- {item}" for item in dossier[title]))
                            st.markdown("**Full dossier**")
                            st.json(dossier, expanded=False)
                    right.button("Use for training", key=f"use_{row['model']}", on_click=use_model, args=(row["model"],),
                                 width="stretch", disabled=verdict == "no_fit")

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
            source = st.selectbox("Dataset", list(DOWNLOADS), format_func=DOWNLOADS.get)
            c1, c2, c3 = st.columns(3)
            if source == "shapenet_part":
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
                    if source == "shapenet_part":
                        dataset = import_shapenet_part(category, int(points), int(limit) or None, progress=report)
                    else:
                        dataset = import_modelnet(int(source[-2:]), int(points), int(limit) or None, progress=report)
                except Exception as exc:  # noqa: BLE001 - network/archive errors are user-facing messages
                    st.error(f"Import failed: {exc}")
                else:
                    st.session_state["train_dataset"] = dataset.name
                    st.success(f"Registered `{dataset.name}`: {dataset.description} Selected in the Training tab.")

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
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Model", model_names, key="train_model", disabled=running)
        c2.selectbox("Dataset", sorted(DATASETS), key="train_dataset", disabled=running,
                     format_func=lambda d: f"{d} ({DATASETS[d].task.replace('_', ' ')})")
        c3.text_input("Experiment name", key="train_experiment", disabled=running)
        c1.number_input("Epochs", min_value=1, max_value=10_000, key="train_epochs", disabled=running)
        c2.number_input("Learning rate", min_value=1e-7, max_value=1.0, format="%.1e", step=1e-4, key="train_lr", disabled=running)
        c3.number_input("Batch size", min_value=1, max_value=4096, key="train_batch", disabled=running)
        chosen = DATASETS[st.session_state["train_dataset"]]
        st.caption(f"**{chosen.task.replace('_', ' ')}**: {chosen.description}"
                   + (f" Classes: {', '.join(chosen.class_names)}." if chosen.class_names else "")
                   + " The model head (`output_dim`, `task`) is sized from the dataset; JSON overrides win.")
        mismatch = cached_input_mismatch(st.session_state["train_model"], chosen.name)
        if mismatch:
            st.warning(mismatch)
        problem = st.session_state.get("problem_dataset")
        if problem and DATASETS[problem].task != chosen.task:
            st.warning(f"The advisor problem is {DATASETS[problem].task.replace('_', ' ')}, but `{chosen.name}` "
                       f"trains {chosen.task.replace('_', ' ')}. Pick `{problem}` to train for your problem.")
        with st.expander("Advanced"):
            a1, a2, a3, a4 = st.columns(4)
            a1.selectbox("Optimizer", ("adam", "adamw", "sgd"), key="train_optimizer", disabled=running)
            a2.number_input("Samples", min_value=2, max_value=100_000, key="train_samples", disabled=running)
            a3.number_input("Points per sample", min_value=8, max_value=200_000, key="train_points", disabled=running)
            devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
            if st.session_state["train_device"] not in devices:
                st.session_state["train_device"] = "cpu"
            a4.selectbox("Device", devices, key="train_device", disabled=running)
        gpu_problem = cuda_problem()
        if gpu_problem:
            st.warning(gpu_problem)
        with st.expander("Model config overrides (JSON)"):
            st.text_area("Model config overrides (JSON)", key="train_model_config", disabled=running,
                         help='Passed to ModelRegistry.create, e.g. {"widths": [64, 64]}.')

        b1, b2, b3 = st.columns([1, 1, 2])
        if b1.button("Start Training", type="primary", disabled=running, width="stretch"):
            try:
                trainer.start_training(config_from_form(st.session_state))
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:  # bad JSON/config or a run in progress
                st.error(f"Could not start training: {exc}")
            else:
                st.rerun()
        if b2.button("Stop Training", disabled=not running, width="stretch"):
            trainer.stop_training(timeout=0)  # do not block the UI; the worker stops after this epoch
            st.toast("Stopping after the current epoch...")
        b3.link_button("Open MLflow UI", MLFLOW_UI_URL, width="stretch")
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

            st.button("Launch New Run with Modified Params", on_click=relaunch, type="primary")

    # ------------------------------------------------------------ 5. results
    with results_tab:
        import pandas as pd

        st.caption(measured["note"])
        if not measured["runs"]:
            st.info("No finished runs yet. Train a model in Tab 3; its final validation metrics appear here.")
        else:
            st.subheader("Leaderboard: best model per dataset")
            st.dataframe(pd.DataFrame(measured["best_per_dataset"]), hide_index=True, width="stretch",
                         column_config={"score": st.column_config.NumberColumn(format="%.4g")})

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
            st.download_button("Download all runs (CSV)", flat.to_csv(index=False), "neural3dbench_runs.csv", "text/csv")


if __name__ == "__main__":
    main()
