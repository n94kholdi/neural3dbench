"""Background training engine with MLflow tracking for the Streamlit dashboard.

Training runs in a daemon ``TrainingWorker`` thread so the Streamlit script thread never
blocks. ``Trainer`` owns the worker and a lock-protected status dict that the UI polls with
``get_status()``.

Usage::

    trainer = Trainer()
    trainer.start_training({"model_name": "mlp", "epochs": 5})
    trainer.get_status()  # {"status": "training", "current_epoch": 2, ...}
    trainer.stop_training()

Synthetic datasets (no files needed, see ``tools.dashboard.tasks.DATASETS``) feed
``ModelInput(coordinates=[B, N, 3])``; mesh datasets are joined per batch with
``collate_meshes`` (``tasks.model_input``). The dataset's task sets the model head, loss and
metrics: field regression, part segmentation, shape classification or shape completion.
"""

from __future__ import annotations

import copy
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

from models import ModelRegistry
from tools.dashboard import tasks

STATUSES = ("idle", "training", "completed", "failed")
_OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


@dataclass
class TrainingConfig:
    """Everything one dashboard training run needs; all fields are logged as MLflow params."""

    model_name: str = "mlp"
    model_config: dict[str, Any] = field(default_factory=dict)
    dataset: str = "deformed_sphere"  # a key of tasks.DATASETS
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    momentum: float = 0.0  # SGD only
    weight_decay: float = 0.0
    lr_schedule: str = "none"  # none | cosine:<eta_min> | step:<every>:<gamma> | multistep:<e1>,<e2>:<gamma>
    augment_rotation: bool = False  # random SO(3) rotation of each training sample (point datasets)
    num_samples: int = 64
    num_points: int = 512
    val_fraction: float = 0.25
    experiment_name: str = "neural3dbench"
    tracking_uri: str = "./mlruns"
    checkpoint_dir: str = "checkpoints"
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.dataset not in tasks.DATASETS:
            raise ValueError(f"dataset must be one of {sorted(tasks.DATASETS)}; got {self.dataset!r}.")
        if self.optimizer.lower() not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(_OPTIMIZERS)}; got {self.optimizer!r}.")
        for name in ("epochs", "batch_size", "num_samples", "num_points"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1).")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.momentum < 0 or self.weight_decay < 0:
            raise ValueError("momentum and weight_decay must be non-negative.")
        parse_lr_schedule(self.lr_schedule)
        if self.augment_rotation and tasks.DATASETS[self.dataset].mesh:
            raise ValueError("rotation augmentation supports point datasets only.")


def parse_lr_schedule(spec: str) -> tuple[str, list[float]]:
    """``none`` | ``cosine[:eta_min]`` | ``step:<every>:<gamma>`` | ``multistep:<e1>,<e2>,...:<gamma>``."""
    kind, *args = (spec or "none").strip().lower().split(":")
    try:
        if kind == "none" and not args:
            return kind, []
        if kind == "cosine" and len(args) <= 1:
            return kind, [float(args[0]) if args else 0.0]
        if kind == "step" and len(args) == 2 and int(args[0]) > 0:
            return kind, [int(args[0]), float(args[1])]
        if kind == "multistep" and len(args) == 2:
            return kind, [*(int(m) for m in args[0].split(",")), float(args[1])]
    except ValueError:
        pass
    raise ValueError(f"lr_schedule {spec!r} must be none, cosine:<eta_min>, step:<every>:<gamma> "
                     "or multistep:<e1>,<e2>:<gamma>.")


def make_scheduler(optimizer: torch.optim.Optimizer, spec: str, epochs: int):
    """Per-epoch LR scheduler for ``spec`` (see ``parse_lr_schedule``), or None."""
    kind, args = parse_lr_schedule(spec)
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args[0])
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args[0], gamma=args[1])
    if kind == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args[:-1], gamma=args[-1])
    return None


def random_rotations(count: int, device: torch.device) -> torch.Tensor:
    """``count`` uniform random rotation matrices ``[count, 3, 3]`` (QR of Gaussian matrices)."""
    q, r = torch.linalg.qr(torch.randn(count, 3, 3, device=device))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    q[torch.det(q) < 0, :, 0] *= -1  # proper rotations only
    return q


def make_synthetic_dataset(config: TrainingConfig) -> tuple[torch.Tensor | list, torch.Tensor]:
    """Return ``(coordinates [B, N, 3], target)`` for ``config.dataset``; target shape follows its task."""
    return tasks.make_dataset(config.dataset, config.num_samples, config.num_points, config.seed)


def build_model(config: TrainingConfig) -> nn.Module:
    """Create the model with its head sized for the dataset's task; explicit ``model_config`` keys win."""
    head = tasks.head_config(config.model_name, tasks.DATASETS[config.dataset])
    return ModelRegistry.create(config.model_name, {**head, **config.model_config})


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

    def _train(self) -> None:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        device = torch.device(cfg.device)
        dataset = tasks.DATASETS[cfg.dataset]
        task = dataset.task
        coords, target = make_synthetic_dataset(cfg)
        n_val = max(1, int(round(cfg.num_samples * cfg.val_fraction)))
        if n_val >= cfg.num_samples:
            raise ValueError("num_samples too small for a train/val split.")
        model = build_model(cfg).to(device)
        if dataset.mesh:  # a list of (vertices, faces); model_input collates each batch
            coords = tasks.attach_operators(model, coords)
            train_x, val_x = coords[n_val:], coords[:n_val]
        else:
            train_x, val_x = coords[n_val:].to(device), coords[:n_val].to(device)
        train_y, val_y = target[n_val:].to(device), target[:n_val].to(device)
        val_input = tasks.model_input(val_x, range(len(val_x)))
        extra = {"momentum": cfg.momentum} if cfg.optimizer.lower() == "sgd" else {}
        optimizer = _OPTIMIZERS[cfg.optimizer.lower()](model.parameters(), lr=cfg.learning_rate,
                                                       weight_decay=cfg.weight_decay, **extra)
        scheduler = make_scheduler(optimizer, cfg.lr_schedule, cfg.epochs)

        mlflow.set_tracking_uri(_resolve_tracking_uri(cfg.tracking_uri))
        mlflow.set_experiment(cfg.experiment_name)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            self.trainer._update(run_id=run_id)
            params = {k: v for k, v in asdict(cfg).items() if k != "model_config"}
            params.update({f"model.{k}": v for k, v in asdict(model.config).items()})
            params.update(task=task, num_classes=dataset.num_classes)
            params["num_parameters"] = sum(p.numel() for p in model.parameters())
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

            for epoch in range(1, cfg.epochs + 1):
                if self.stop_event.is_set():
                    break
                model.train()
                order = torch.randperm(len(train_x), device=device)
                total, seen = 0.0, 0
                for start in range(0, len(order), cfg.batch_size):
                    index = order[start : start + cfg.batch_size]
                    if len(index) == 1 and start:  # BatchNorm cannot train on one sample; drop a 1-sample tail
                        continue
                    optimizer.zero_grad()
                    batch_x, batch_y = tasks.model_input(train_x, index), train_y[index]
                    if cfg.augment_rotation:
                        rotation = random_rotations(len(index), device)
                        batch_x = batch_x @ rotation
                        if task == tasks.SHAPE_COMPLETION:  # the complete cloud turns with the partial one
                            batch_y = batch_y @ rotation
                    loss = tasks.loss(task, tasks.forward(task, model, batch_x), batch_y)
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(index)
                    seen += len(index)
                train_loss = total / seen
                if scheduler is not None:
                    scheduler.step()

                model.eval()
                with torch.no_grad():
                    val_pred = tasks.forward(task, model, val_input)
                    val_loss = tasks.loss(task, val_pred, val_y).item()
                    val_metrics = tasks.metrics(task, val_pred, val_y, dataset.num_classes)
                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}},
                    step=epoch,
                )
                self.trainer._update(current_epoch=epoch, train_loss=train_loss, val_loss=val_loss)

            checkpoint = Path(cfg.checkpoint_dir) / f"{run_id}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_name": cfg.model_name, "config": asdict(model.config), "state_dict": model.state_dict()}, checkpoint)
            mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")
            self._log_evaluation_html(model, dataset, val_x, val_y)
            stopped = self.stop_event.is_set()
            mlflow.set_tag("stopped_early", str(stopped))

        self.trainer._update(status="completed", checkpoint_path=str(checkpoint), stopped=stopped)

    def _log_evaluation_html(self, model: nn.Module, dataset: tasks.Dataset, data: torch.Tensor | list, target: torch.Tensor) -> None:
        """Preview of the first validation sample."""
        model.eval()
        with torch.no_grad():
            pred = tasks.forward(dataset.task, model, tasks.model_input(data, [0]))
        sample = data[0][:2] if dataset.mesh else data[0].cpu().numpy()
        figure = tasks.evaluation_figure(dataset, sample, target[0].cpu(), pred[0].cpu())
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
        mismatch = tasks.input_mismatch(config.model_name, config.dataset)
        if mismatch:
            raise ValueError(mismatch.replace("`", ""))
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

    @property
    def config(self) -> TrainingConfig | None:
        """Config of the latest run (running or finished), or ``None`` before the first run."""
        return self._worker.config if self._worker is not None else None

    def get_status(self) -> dict[str, Any]:
        """Snapshot copy of the run state; safe to call from any thread."""
        with self._lock:
            return dict(self._state)

    def wait(self, timeout: float | None = None) -> dict[str, Any]:
        """Block until the worker exits (for scripts and tests, not the Streamlit thread)."""
        if self._worker is not None:
            self._worker.join(timeout)
        return self.get_status()
