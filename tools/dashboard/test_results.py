"""Results report: MLflow runs are listed and ranked per dataset by the task's primary metric.

Run with ``pytest tools/dashboard/test_results.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
mlflow = pytest.importorskip("mlflow")

from tools.mcp_advisor.results import benchmark_results  # noqa: E402


def _log(model: str, dataset: str, task: str, metrics: dict[str, float]) -> None:
    with mlflow.start_run():
        mlflow.log_params({"model_name": model, "dataset": dataset, "task": task, "epochs": 3})
        mlflow.log_metrics(metrics, step=3)


def test_ranks_by_primary_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert benchmark_results()["runs"] == []  # no store yet: empty, not an error

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test-results")
    _log("pointnet", "chair_parts", "part_segmentation", {"val_miou": 0.61, "val_loss": 0.9})
    _log("dgcnn", "chair_parts", "part_segmentation", {"val_miou": 0.74, "val_loss": 1.2})
    _log("mlp", "poisson", "field_regression", {"val_mae": 0.20})
    _log("siren", "poisson", "field_regression", {"val_mae": 0.05})

    result = benchmark_results()
    assert len(result["runs"]) == 4
    best = {row["dataset"]: row["model"] for row in result["best_per_dataset"]}
    assert best == {"chair_parts": "dgcnn", "poisson": "siren"}  # mIoU up, MAE down; val_loss ignored

    only = benchmark_results(model_name="mlp")["runs"]
    assert [row["model"] for row in only] == ["mlp"] and only[0]["epochs"] == 3
