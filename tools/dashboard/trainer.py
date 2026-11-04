"""Background training engine with MLflow tracking for the Streamlit dashboard.

Training runs in a daemon ``TrainingWorker`` thread so the Streamlit script thread never
blocks. ``Trainer`` owns the worker and a lock-protected status dict that the UI polls with
``get_status()``.

Usage::

    trainer = Trainer()
    trainer.start_training({"model_name": "mlp", "epochs": 5})
    trainer.get_status()  # {"status": "training", "current_epoch": 2, ...}
    trainer.stop_training()

Synthetic datasets (no files needed) feed ``ModelInput(coordinates=[B, N, 3])`` with a
target field ``[B, N, 1]``, so they suit point/coordinate models (mlp, siren, pointnet, ...).
"""

from __future__ import annotations

import copy
import math
import os
import tempfile
import threading
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch import nn

from models import ModelInput, ModelOutput, ModelRegistry

STATUSES = ("idle", "training", "completed", "failed")
_OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


@dataclass
class TrainingConfig:
    """Everything one dashboard training run needs; all fields are logged as MLflow params."""

    model_name: str = "mlp"
    model_config: dict[str, Any] = field(default_factory=dict)
    dataset: str = "deformed_sphere"  # or "poisson"
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    num_samples: int = 64
    num_points: int = 512
    val_fraction: float = 0.25
    experiment_name: str = "neural3dbench"
    tracking_uri: str = "./mlruns"
    checkpoint_dir: str = "checkpoints"
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.dataset not in SYNTHETIC_DATASETS:
            raise ValueError(f"dataset must be one of {sorted(SYNTHETIC_DATASETS)}; got {self.dataset!r}.")
        if self.optimizer.lower() not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(_OPTIMIZERS)}; got {self.optimizer!r}.")
        for name in ("epochs", "batch_size", "num_samples", "num_points"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1).")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")


# ---------------------------------------------------------------- synthetic data


def _deformed_sphere(num_samples: int, num_points: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Spheres with random low-order radial bumps; target is the radial displacement [B, N, 1]."""
    direction = torch.randn(num_samples, num_points, 3, generator=generator)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    amplitude = 0.2 * torch.rand(num_samples, 1, 3, generator=generator)
    frequency = torch.randint(1, 4, (num_samples, 1, 3), generator=generator).float()
    displacement = (amplitude * torch.sin(math.pi * frequency * direction)).sum(-1, keepdim=True)
    return direction * (1.0 + displacement), displacement


def _poisson(num_samples: int, num_points: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Manufactured Poisson solution u = prod sin(k_i pi x_i) on [0, 1]^3 (-lap u = pi^2 |k|^2 u)."""
    coords = torch.rand(num_samples, num_points, 3, generator=generator)
    k = torch.randint(1, 3, (num_samples, 1, 3), generator=generator).float()
    return coords, torch.sin(math.pi * k * coords).prod(-1, keepdim=True)


SYNTHETIC_DATASETS = {"deformed_sphere": _deformed_sphere, "poisson": _poisson}


def make_synthetic_dataset(config: TrainingConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(coordinates [B, N, 3], target [B, N, 1])`` for ``config.dataset``."""
    generator = torch.Generator().manual_seed(config.seed)
    return SYNTHETIC_DATASETS[config.dataset](config.num_samples, config.num_points, generator)


def _resolve_tracking_uri(uri: str) -> str:
    """Pass real URIs (sqlite:, http:, file:) through; turn a plain path into a file: URI."""
    if "://" in uri or uri.startswith("sqlite:"):
        return uri
    # MLflow >= 3.x raises on the file store unless explicitly allowed; the dashboard wants ./mlruns.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return Path(uri).resolve().as_uri()


# ---------------------------------------------------------------- worker


class TrainingWorker(threading.Thread):
    """Daemon thread running one MLflow-tracked training run, reporting through ``Trainer``."""

    def __init__(self, config: TrainingConfig, trainer: Trainer) -> None:
        super().__init__(name="neural3dbench-training", daemon=True)
        self.config = config
        self.trainer = trainer
        self.stop_event = threading.Event()

    def run(self) -> None:
        try:
            self._train()
        except Exception as exc:  # surface any failure to the UI instead of dying silently
            self.trainer._update(status="failed", error=f"{exc}\n{traceback.format_exc()}")

    def _forward(self, model: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        output = model(ModelInput(coordinates=coords))
        predictions = output.predictions if isinstance(output, ModelOutput) else output
        return predictions.reshape(coords.shape[0], coords.shape[1], -1)

    def _train(self) -> None:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        device = torch.device(cfg.device)
        coords, target = make_synthetic_dataset(cfg)
        n_val = max(1, int(round(cfg.num_samples * cfg.val_fraction)))
        if n_val >= cfg.num_samples:
            raise ValueError("num_samples too small for a train/val split.")
        train_x, val_x = coords[n_val:].to(device), coords[:n_val].to(device)
        train_y, val_y = target[n_val:].to(device), target[:n_val].to(device)

        model = ModelRegistry.create(cfg.model_name, dict(cfg.model_config)).to(device)
        optimizer = _OPTIMIZERS[cfg.optimizer.lower()](model.parameters(), lr=cfg.learning_rate)
        loss_fn = nn.MSELoss()

        mlflow.set_tracking_uri(_resolve_tracking_uri(cfg.tracking_uri))
        mlflow.set_experiment(cfg.experiment_name)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            self.trainer._update(run_id=run_id)
            params = {k: v for k, v in asdict(cfg).items() if k != "model_config"}
            params.update({f"model.{k}": v for k, v in asdict(model.config).items()})
            params["num_parameters"] = sum(p.numel() for p in model.parameters())
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

            for epoch in range(1, cfg.epochs + 1):
                if self.stop_event.is_set():
                    break
                model.train()
                order = torch.randperm(len(train_x), device=device)
                total = 0.0
                for start in range(0, len(order), cfg.batch_size):
                    index = order[start : start + cfg.batch_size]
                    optimizer.zero_grad()
                    loss = loss_fn(self._forward(model, train_x[index]), train_y[index])
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(index)
                train_loss = total / len(train_x)

                model.eval()
                with torch.no_grad():
                    val_pred = self._forward(model, val_x)
                    val_loss = loss_fn(val_pred, val_y).item()
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                self.trainer._update(current_epoch=epoch, train_loss=train_loss, val_loss=val_loss)

            checkpoint = Path(cfg.checkpoint_dir) / f"{run_id}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_name": cfg.model_name, "config": asdict(model.config), "state_dict": model.state_dict()}, checkpoint)
            mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")
            self._log_evaluation_html(model, val_x[:1], val_y[:1])
            stopped = self.stop_event.is_set()
            mlflow.set_tag("stopped_early", str(stopped))

        self.trainer._update(status="completed", checkpoint_path=str(checkpoint), stopped=stopped)

    def _log_evaluation_html(self, model: nn.Module, coords: torch.Tensor, target: torch.Tensor) -> None:
        from tools.dashboard.visualizer import plot_field_comparison  # plotly only needed here

        model.eval()
        with torch.no_grad():
            pred = self._forward(model, coords)
        figure = plot_field_comparison(coords[0].cpu().numpy(), None, target[0].cpu().numpy(), pred[0].cpu().numpy(), "Field")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation.html"
            figure.write_html(path, include_plotlyjs="cdn")
            mlflow.log_artifact(str(path), artifact_path="evaluation")


# ---------------------------------------------------------------- manager


class Trainer:
    """Thread-safe front door for the dashboard: start, stop and poll one training run at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._worker: TrainingWorker | None = None
        self._state: dict[str, Any] = self._blank_state("idle", 0)

    @staticmethod
    def _blank_state(status: str, total_epochs: int) -> dict[str, Any]:
        return {
            "status": status,
            "current_epoch": 0,
            "total_epochs": total_epochs,
            "train_loss": None,
            "val_loss": None,
            "run_id": None,
            "checkpoint_path": None,
            "stopped": False,
            "error": None,
        }

    def _update(self, **changes: Any) -> None:
        with self._lock:
            self._state.update(changes)

    def start_training(self, config: TrainingConfig | dict[str, Any]) -> None:
        """Validate ``config`` and launch a background run. Raises if one is already running."""
        if isinstance(config, dict):
            config = TrainingConfig(**config)
        ModelRegistry.get(config.model_name)  # fail fast on unknown model, before the thread starts
        with self._lock:
            if self._state["status"] == "training":
                raise RuntimeError("A training run is already in progress.")
            self._state = self._blank_state("training", config.epochs)
            self._worker = TrainingWorker(copy.deepcopy(config), self)
            self._worker.start()

    def stop_training(self, timeout: float | None = None) -> None:
        """Ask the worker to stop after the current epoch; it still saves weights and artifacts."""
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.stop_event.set()
            worker.join(timeout)

    def get_status(self) -> dict[str, Any]:
        """Snapshot copy of the run state; safe to call from any thread."""
        with self._lock:
            return dict(self._state)

    def wait(self, timeout: float | None = None) -> dict[str, Any]:
        """Block until the worker exits (for scripts and tests, not the Streamlit thread)."""
        if self._worker is not None:
            self._worker.join(timeout)
        return self.get_status()
