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
TASKS = ("surrogate_simulation", "pde_operator", "segmentation_classification", "field_fitting")
PRECISIONS = ("fp32", "fp16", "bf16", "amp")
POLL_SECONDS = 2.0
_VERTEX_KEYS = ("vertices", "coords", "coordinates", "points", "pos")
_FIELD_KEYS = ("field", "target", "features", "values")


# ---------------------------------------------------------------- geometry loading


def _fan(polygon: list[int]) -> list[list[int]]:
    """Triangulate a convex polygon as a fan around its first vertex."""
    return [[polygon[0], polygon[i], polygon[i + 1]] for i in range(1, len(polygon) - 1)]


def _geometry(vertices: Any, faces: Any = None, field: Any = None) -> dict[str, Any]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"vertices must have shape [N, 3]; got {vertices.shape}.")
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3) if faces is not None and len(faces) else None
    if faces is not None and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("faces index outside the vertex array.")
    field = None if field is None else np.asarray(field, dtype=np.float64).reshape(len(vertices), -1)
    return {"vertices": vertices, "faces": faces, "field": field}


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
    """ASCII PLY with a ``vertex`` element (x, y, z) and an optional ``face`` element.

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
    rows = body.decode("ascii").split("\n")[1:]  # drop the rest of the end_header line
    rows = [row.split() for row in rows if row.strip()]
    n_vertices = counts.get("vertex", 0)
    vertices = [[float(row[c]) for c in columns] for row in rows[:n_vertices]]
    faces = []
    for row in rows[n_vertices : n_vertices + counts.get("face", 0)]:
        faces += _fan([int(i) for i in row[1 : 1 + int(row[0])]])
    return _geometry(vertices, faces)


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


SAMPLES = {
    "shape:sphere": "Sphere (examples/train_pointnet_synthetic.py)",
    "shape:cube": "Cube (examples/train_pointnet_synthetic.py)",
    "shape:cylinder": "Cylinder (examples/train_pointnet_synthetic.py)",
    "dataset:deformed_sphere": "Deformed sphere + radial displacement (trainer dataset)",
    "dataset:poisson": "Poisson manufactured solution (trainer dataset)",
}


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


def mlflow_client(tracking_uri: str) -> Any:
    from mlflow.tracking import MlflowClient

    from tools.dashboard.trainer import _resolve_tracking_uri

    return MlflowClient(tracking_uri=_resolve_tracking_uri(tracking_uri))


def predict_sample(checkpoint_path: str, config: Any, sample: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reload a checkpoint and predict one validation sample: ``(coords [N, 3], gt [N, 1], pred [N, C])``."""
    from models import ModelInput, ModelOutput, ModelRegistry
    from tools.dashboard.trainer import make_synthetic_dataset

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ModelRegistry.create(checkpoint["model_name"], checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    coords, target = make_synthetic_dataset(config)  # validation samples are the first rows, as in TrainingWorker
    x = coords[sample : sample + 1]
    with torch.no_grad():
        output = model(ModelInput(coordinates=x))
    pred = output.predictions if isinstance(output, ModelOutput) else output
    return x[0].numpy(), target[sample].numpy(), pred.reshape(x.shape[1], -1).numpy()


def num_validation_samples(config: Any) -> int:
    return max(1, int(round(config.num_samples * config.val_fraction)))


# ---------------------------------------------------------------- UI


def main() -> None:
    import streamlit as st

    from models import ModelRegistry
    from tools.dashboard.advisor_bridge import AdvisorBridge
    from tools.dashboard.trainer import SYNTHETIC_DATASETS, Trainer, TrainingConfig
    from tools.dashboard.visualizer import plot_field_comparison, plot_mesh, plot_point_cloud

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

    @st.cache_data(show_spinner="Running the model on the validation sample...")
    def cached_prediction(checkpoint_path: str, config_json: str, sample: int) -> tuple[np.ndarray, ...]:
        return predict_sample(checkpoint_path, TrainingConfig(**json.loads(config_json)), sample)

    bridge, trainer = get_bridge(), get_trainer()
    model_names = sorted(ModelRegistry._registry)
    for key, value in form_values(TrainingConfig()).items():
        st.session_state.setdefault(key, value)

    def use_model(name: str) -> None:
        st.session_state["train_model"] = name
        st.toast(f"{name} selected in the Training tab.")

    st.title("Neural3DBench Dashboard")
    if bridge.source != "advisor":
        st.warning(f"Advisor unavailable, using the registry fallback: {bridge.error}")
    advisor_tab, preview_tab, train_tab, inspect_tab = st.tabs(
        ["1 · Problem & Advisor", "2 · 3D Data Preview", "3 · Training", "4 · Inspection"]
    )

    # ------------------------------------------------------------ 1. advisor
    with advisor_tab:
        with st.form("problem"):
            c1, c2, c3, c4 = st.columns(4)
            task = c1.selectbox("Task domain", TASKS, format_func=lambda t: t.replace("_", " "))
            representation = c2.selectbox("Data representation", list(REPRESENTATIONS), format_func=REPRESENTATIONS.get)
            num_nodes = c3.number_input("Points / vertices per sample", min_value=1, value=10_000, step=1_000)
            vram_gb = c4.number_input("Target GPU memory (GB)", min_value=0.5, value=8.0, step=1.0)
            description = st.text_input("Problem description (optional)", placeholder="e.g. CFD pressure on car surfaces")
            target_metric = st.radio("Optimize for", ("accuracy", "speed"), horizontal=True)
            submitted = st.form_submit_button("Get recommendations", type="primary")
        if submitted:
            try:
                st.session_state["recommendations"] = bridge.get_recommendations(
                    f"{task} {description}", representation, int(num_nodes), vram_gb, target_metric
                )
            except (ValueError, KeyError) as exc:
                st.error(str(exc))

        result = st.session_state.get("recommendations")
        if result:
            st.caption(f"{result.get('note', '')} Considered {result.get('considered', 0)} models; source: {result['source']}.")
            if not result["recommendations"]:
                st.info("No registered model matches this representation.")
            for rank, row in enumerate(result["recommendations"], 1):
                with st.container(border=True):
                    head, score, memory, fits = st.columns([3, 1, 1, 1])
                    head.subheader(f"#{rank} {row.get('display_name', row['model'])}")
                    score.metric("Score", row.get("score"))
                    peak = row.get("estimated_training_peak_gb")
                    memory.metric("Train peak", "n/a" if peak is None else f"{peak:.2f} GB")
                    fits.metric("Fits VRAM", {True: "yes", False: "no", None: "n/a"}[row.get("fits_vram")])
                    for line in row.get("rationale", []):
                        st.markdown(f"- {line}")
                    for line in row.get("pitfalls", []):
                        st.markdown(f"- :orange[{line}]")
                    if row.get("complexity"):
                        st.caption(f"Complexity: `{row['complexity']}`")
                    left, right = st.columns(2)
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
                                 width="stretch")

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

    # ------------------------------------------------------------ 2. preview
    with preview_tab:
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
            if faces is not None:
                figure = plot_mesh(vertices, faces, field, title=label)
            else:
                figure = plot_point_cloud(vertices, field, title=label)
            st.plotly_chart(figure, width="stretch")

    # ------------------------------------------------------------ 3. training
    with train_tab:
        status = trainer.get_status()
        running = status["status"] == "training"
        st.caption("Synthetic datasets feed `coordinates [B, N, 3]` and regress a field `[B, N, 1]`, "
                   "so they suit coordinate and point models (mlp, siren, pointnet, transolver, ...).")
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Model", model_names, key="train_model", disabled=running)
        c2.selectbox("Dataset", sorted(SYNTHETIC_DATASETS), key="train_dataset", disabled=running)
        c3.text_input("Experiment name", key="train_experiment", disabled=running)
        c1.number_input("Epochs", min_value=1, max_value=10_000, key="train_epochs", disabled=running)
        c2.number_input("Learning rate", min_value=1e-7, max_value=1.0, format="%.1e", step=1e-4, key="train_lr", disabled=running)
        c3.number_input("Batch size", min_value=1, max_value=4096, key="train_batch", disabled=running)
        with st.expander("Advanced"):
            a1, a2, a3, a4 = st.columns(4)
            a1.selectbox("Optimizer", ("adam", "adamw", "sgd"), key="train_optimizer", disabled=running)
            a2.number_input("Samples", min_value=2, max_value=100_000, key="train_samples", disabled=running)
            a3.number_input("Points per sample", min_value=8, max_value=200_000, key="train_points", disabled=running)
            devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
            if st.session_state["train_device"] not in devices:
                st.session_state["train_device"] = "cpu"
            a4.selectbox("Device", devices, key="train_device", disabled=running)
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
                metric_name = "Displacement" if config.dataset == "deformed_sphere" else "u"
                st.plotly_chart(plot_field_comparison(coords, None, gt, pred, metric_name), width="stretch")
            except Exception as exc:  # noqa: BLE001 - checkpoint/model mismatch is shown, not raised
                st.error(f"Could not run the trained model: {exc}")

            def relaunch() -> None:
                st.session_state.update(form_values(config))
                st.toast("Training form filled from the last run; edit it in Tab 3 and press Start.")

            st.button("Launch New Run with Modified Params", on_click=relaunch, type="primary")


if __name__ == "__main__":
    main()
