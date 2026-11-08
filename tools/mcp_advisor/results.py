"""Measured results: read dashboard training runs from MLflow and rank them per dataset.

Shared by the ``get_benchmark_results`` MCP tool and the dashboard Results tab, so both
report the same numbers. ``mlflow`` is imported lazily; the advisor works without it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Metric that ranks runs of each task: (metric key, higher is better).
PRIMARY_METRIC = {
    "part_segmentation": ("val_miou", True),
    "shape_classification": ("val_accuracy", True),
    "field_regression": ("val_mae", False),
    "shape_completion": ("val_chamfer", False),
}
_FALLBACK_METRIC = ("val_loss", False)


def primary_metric(task: str | None) -> tuple[str, bool]:
    return PRIMARY_METRIC.get(task or "", _FALLBACK_METRIC)


def _tracking_uri(uri: str) -> str:
    if "://" in uri or uri.startswith("sqlite:"):
        return uri
    # MLflow 3 refuses the file store unless allowed; the dashboard writes ./mlruns.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return Path(uri).resolve().as_uri()


def _number(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def best_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Best run for each value of ``rows[i][key]`` (e.g. "dataset" or "model") by its primary metric."""
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["score"] is None:
            continue
        current = best.get(row[key])
        sign = 1 if row["higher_is_better"] else -1
        if current is None or sign * row["score"] > sign * current["score"]:
            best[row[key]] = row
    return best


def benchmark_results(
    tracking_uri: str = "./mlruns",
    model_name: str | None = None,
    dataset: str | None = None,
    include_unfinished: bool = False,
    limit: int = 200,
) -> dict[str, Any]:
    """Runs with final metrics, plus the best run per dataset by the task's primary metric."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return {"runs": [], "best_per_dataset": [], "note": "mlflow is not installed (pip install -e '.[dashboard]')."}

    if "://" not in tracking_uri and not tracking_uri.startswith("sqlite:") and not Path(tracking_uri).exists():
        return {"runs": [], "best_per_dataset": [], "note": f"No MLflow store at {tracking_uri}; train a model first."}

    client = MlflowClient(tracking_uri=_tracking_uri(tracking_uri))
    experiments = {e.experiment_id: e.name for e in client.search_experiments()}
    runs = client.search_runs(list(experiments), max_results=max(1, limit), order_by=["attributes.start_time DESC"])

    rows = []
    for run in runs:
        params, info = run.data.params, run.info
        if info.status != "FINISHED" and not include_unfinished:
            continue
        name = params.get("model_name")
        if name is None or (model_name and name != model_name) or (dataset and params.get("dataset") != dataset):
            continue
        task = params.get("task")
        metric, higher = primary_metric(task)
        rows.append({
            "run_id": info.run_id,
            "experiment": experiments.get(info.experiment_id, info.experiment_id),
            "model": name,
            "dataset": params.get("dataset"),
            "task": task,
            "status": info.status.lower(),
            "epochs": _number(params.get("epochs")),
            "num_parameters": _number(params.get("num_parameters")),
            "device": params.get("device"),
            "stopped_early": run.data.tags.get("stopped_early") == "True",
            "duration_s": round((info.end_time - info.start_time) / 1000, 1) if info.end_time else None,
            "primary_metric": metric,
            "higher_is_better": higher,
            "score": run.data.metrics.get(metric),
            "metrics": {k: round(v, 6) for k, v in sorted(run.data.metrics.items())},
        })

    leaderboard = [
        {key: row[key] for key in ("dataset", "task", "model", "primary_metric", "score", "run_id")}
        for _, row in sorted(best_by(rows, "dataset").items())
    ]
    return {
        "runs": rows,
        "best_per_dataset": leaderboard,
        "note": ("Final-epoch validation metrics from dashboard runs. Synthetic datasets (deformed_sphere, poisson, "
                 "chair_parts, primitive_shapes) are smoke tests, not published benchmarks."),
    }
