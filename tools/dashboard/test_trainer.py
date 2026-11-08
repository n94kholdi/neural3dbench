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

import torch  # noqa: E402

from tools.dashboard import tasks  # noqa: E402
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


@pytest.mark.parametrize(("model_name", "dataset", "metric"), [("pointnet", "chair_parts", "val_miou"),
                                                               ("mlp", "primitive_shapes", "val_accuracy")])
def test_labelled_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_name: str, dataset: str, metric: str) -> None:
    monkeypatch.chdir(tmp_path)
    trainer = Trainer()
    trainer.start_training({"model_name": model_name, "dataset": dataset, "epochs": 2, "batch_size": 4,
                            "num_samples": 12, "num_points": 64, "experiment_name": "test-labelled"})
    status = trainer.wait(timeout=120)
    assert status["status"] == "completed", status["error"]
    client = mlflow.tracking.MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())
    run = client.get_run(status["run_id"])
    assert metric in run.data.metrics and run.data.params["task"] == tasks.DATASETS[dataset].task
    checkpoint = torch.load(status["checkpoint_path"], weights_only=True)
    assert checkpoint["config"]["output_dim"] == tasks.DATASETS[dataset].num_classes


def test_stop_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    trainer = Trainer()
    with pytest.raises(KeyError):
        trainer.start_training({"model_name": "no-such-model"})
    with pytest.raises(ValueError):
        TrainingConfig(dataset="bogus")
    with pytest.raises(ValueError, match="edge_index"):
        trainer.start_training({"model_name": "gcn"})

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


@pytest.mark.parametrize(("model_name", "model_config"), [
    ("pcn", {"latent_dim": 64, "hidden_dim": 32, "num_coarse": 16, "grid_size": 2}),
    ("foldingnet", {"latent_dim": 64, "hidden_dim": 32, "grid_size": 6}),
])
def test_completion_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_name: str, model_config: dict) -> None:
    monkeypatch.chdir(tmp_path)
    trainer = Trainer()
    trainer.start_training({"model_name": model_name, "model_config": model_config, "dataset": "partial_shapes",
                            "epochs": 2, "batch_size": 4, "num_samples": 12, "num_points": 64,
                            "experiment_name": "test-completion"})
    status = trainer.wait(timeout=120)
    assert status["status"] == "completed", status["error"]
    assert status["val_loss"] is not None


def test_completion_model_rejects_field_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="shape-completion"):
        Trainer().start_training({"model_name": "pcn", "dataset": "deformed_sphere"})


def _bumpy_grids(num_samples: int, num_points: int, generator: torch.Generator) -> tuple[list, torch.Tensor]:
    """Grid meshes of varying size, flat (class 0) or bumped (class 1); ``num_points`` is ignored like real mesh data."""
    meshes, labels = [], torch.arange(num_samples) % 2
    for i in range(num_samples):
        n = 5 + i % 3
        u, v = torch.meshgrid(torch.linspace(0, 1, n), torch.linspace(0, 1, n), indexing="ij")
        vertices = torch.stack((u.ravel(), v.ravel(), 0.3 * labels[i] * torch.sin(3 * u.ravel())), dim=1)
        faces = [[a, a + 1, a + n] for a in range(n * n - n) if a % n < n - 1]
        faces += [[a + 1, a + n + 1, a + n] for a in range(n * n - n) if a % n < n - 1]
        meshes.append((vertices.numpy(), torch.tensor(faces).numpy()))
    return meshes, labels


@pytest.mark.parametrize(("model_name", "model_config"), [("meshnet", {"num_kernel": 8}),
                                                          ("diffusionnet", {"width": 16, "num_blocks": 2, "k_eig": 8})])
def test_mesh_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_name: str, model_config: dict) -> None:
    pytest.importorskip("scipy")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(tasks.DATASETS, "grids", tasks.Dataset("grids", tasks.SHAPE_CLASSIFICATION, "test meshes",
                                                               _bumpy_grids, class_names=("flat", "bumpy"), mesh=True))
    assert tasks.input_mismatch(model_name, "grids") is None and tasks.input_mismatch("pointnet", "grids")
    assert tasks.input_mismatch(model_name, "primitive_shapes")
    trainer = Trainer()
    trainer.start_training({"model_name": model_name, "model_config": model_config, "dataset": "grids", "epochs": 2,
                            "batch_size": 4, "num_samples": 12, "experiment_name": "test-mesh"})
    status = trainer.wait(timeout=300)
    assert status["status"] == "completed", status["error"]
    run = mlflow.MlflowClient().get_run(status["run_id"])
    assert "val_accuracy" in run.data.metrics
