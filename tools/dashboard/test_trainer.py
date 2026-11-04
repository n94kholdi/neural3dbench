"""Phase 2 test: a 2-epoch synthetic run tracked in ./mlruns through the Trainer state manager.

Run with ``pytest tools/dashboard/test_trainer.py``. The test chdirs into a temp dir so
``./mlruns`` and ``./checkpoints`` do not land in the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

mlflow = pytest.importorskip("mlflow")

from tools.dashboard.trainer import Trainer, TrainingConfig  # noqa: E402


@pytest.mark.parametrize("dataset", ["deformed_sphere", "poisson"])
def test_two_epoch_synthetic_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset: str) -> None:
    monkeypatch.chdir(tmp_path)
    trainer = Trainer()
    assert trainer.get_status()["status"] == "idle"

    trainer.start_training(
        {
            "model_name": "mlp",
            "model_config": {"widths": (32, 32)},
            "dataset": dataset,
            "epochs": 2,
            "batch_size": 4,
            "num_samples": 12,
            "num_points": 64,
            "experiment_name": "test-trainer",
        }
    )
    status = trainer.wait(timeout=120)

    assert status["status"] == "completed", status["error"]
    assert status["current_epoch"] == status["total_epochs"] == 2
    assert status["train_loss"] is not None and status["val_loss"] is not None
    run_id = status["run_id"]
    assert (tmp_path / "mlruns").is_dir()
    assert (tmp_path / "checkpoints" / f"{run_id}.pt").is_file()

    client = mlflow.tracking.MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())
    for metric in ("train_loss", "val_loss"):
        assert [m.step for m in client.get_metric_history(run_id, metric)] == [1, 2]
    run = client.get_run(run_id)
    assert run.data.params["learning_rate"] == "0.001"
    assert run.data.params["optimizer"] == "adam"
    artifacts = {a.path for a in client.list_artifacts(run_id)} | {
        a.path for d in ("checkpoints", "evaluation") for a in client.list_artifacts(run_id, d)
    }
    assert f"checkpoints/{run_id}.pt" in artifacts
    assert "evaluation/evaluation.html" in artifacts


def test_stop_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    trainer = Trainer()
    with pytest.raises(KeyError):
        trainer.start_training({"model_name": "no-such-model"})
    with pytest.raises(ValueError):
        TrainingConfig(dataset="bogus")

    trainer.start_training({"model_name": "mlp", "epochs": 500, "num_samples": 8, "num_points": 32})
    with pytest.raises(RuntimeError):
        trainer.start_training({"model_name": "mlp"})
    trainer.stop_training(timeout=120)
    status = trainer.get_status()
    assert status["status"] == "completed", status["error"]
    assert status["stopped"] and status["current_epoch"] < 500


def test_failed_run_reports_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    trainer = Trainer()
    trainer.start_training({"model_name": "mlp", "model_config": {"output_dim": -1}, "epochs": 1})
    status = trainer.wait(timeout=60)
    assert status["status"] == "failed"
    assert "output_dim" in status["error"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
