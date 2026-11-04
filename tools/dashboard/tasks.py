"""Task and dataset contract shared by the advisor bridge, the trainer and the previews.

A *task* fixes how a model is trained and judged: the head it needs (``output_dim``, and
``task`` for models that switch between a global and a per-point head), the loss, the
metrics and the preview figure. A *dataset* fixes the data and names the task it feeds.

==================== ============================ ======================= =====================
task                 target                       loss                    metrics
==================== ============================ ======================= =====================
field_regression     float ``[B, N, C]``          MSE                     MAE
part_segmentation    int part id ``[B, N]``       per-point cross-entropy accuracy, mIoU
shape_classification int class id ``[B]``         cross-entropy           accuracy
shape_completion     complete cloud ``[B, N, 3]``  Chamfer (squared L2)    Chamfer
==================== ============================ ======================= =====================

Per-point models without a global head (mlp, siren, ...) are mean-pooled over points for
shape classification. Completion models (pcn, foldingnet) take the partial cloud and return a
new point set ``[B, M, 3]``; M need not equal N.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models import ModelInput, ModelOutput, ModelRegistry

FIELD_REGRESSION = "field_regression"
PART_SEGMENTATION = "part_segmentation"
SHAPE_CLASSIFICATION = "shape_classification"
SHAPE_COMPLETION = "shape_completion"
TASKS = (FIELD_REGRESSION, PART_SEGMENTATION, SHAPE_CLASSIFICATION, SHAPE_COMPLETION)

Generator = Callable[[int, int, torch.Generator], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class Dataset:
    """A data source: ``make(num_samples, num_points, generator) -> (coords [B, N, 3], target)``."""

    name: str
    task: str
    description: str
    make: Generator
    class_names: tuple[str, ...] = ()
    field_name: str = "value"
    class_colors: tuple[str, ...] = ()  # hex per class; empty means the default palette

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


# ---------------------------------------------------------------- synthetic regression data


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


# ---------------------------------------------------------------- synthetic labelled data

CHAIR_PARTS = ("seat", "back", "legs", "armrests")


def _sample_boxes(centers: torch.Tensor, halves: torch.Tensor, count: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform points on the surfaces of boxes ``[K, 3]`` (center, half-size); returns points and box index."""
    hx, hy, hz = halves.unbind(-1)
    face_area = torch.stack((hy * hz, hx * hz, hx * hy), dim=-1)  # per axis, the two faces normal to it
    box = torch.multinomial(face_area.sum(-1), count, replacement=True, generator=generator)
    axis = torch.multinomial(face_area[box], 1, generator=generator)[:, 0]
    unit = 2 * torch.rand(count, 3, generator=generator) - 1
    unit[torch.arange(count), axis] = (2 * torch.randint(0, 2, (count,), generator=generator) - 1).float()
    return centers[box] + unit * halves[box], box


def _chair_parts(num_samples: int, num_points: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Box chairs (z up, back at -y) with random proportions; target is the part id [B, N] over ``CHAIR_PARTS``."""

    def uniform(low: float, high: float) -> float:
        return low + (high - low) * torch.rand(1, generator=generator).item()

    clouds, labels = [], []
    for _ in range(num_samples):
        width, depth, height = uniform(0.8, 1.2), uniform(0.7, 1.1), uniform(0.7, 1.1)
        leg, slab, back_height = uniform(0.04, 0.1), uniform(0.05, 0.1), uniform(0.6, 1.2)
        boxes = [((0, 0, height), (width / 2, depth / 2, slab / 2), 0)]
        boxes.append(((0, -depth / 2 + slab / 2, height + back_height / 2), (width / 2, slab / 2, back_height / 2), 1))
        for sx in (-1, 1):
            for sy in (-1, 1):
                center = (sx * (width / 2 - leg / 2), sy * (depth / 2 - leg / 2), height / 2)
                boxes.append((center, (leg / 2, leg / 2, height / 2), 2))
        if torch.rand(1, generator=generator).item() < 0.5:
            arm_z = height + uniform(0.2, 0.35)
            for sx in (-1, 1):
                boxes.append(((sx * width / 2, 0, arm_z), (slab / 2, depth / 2, slab / 2), 3))
        centers = torch.tensor([b[0] for b in boxes], dtype=torch.float32)
        halves = torch.tensor([b[1] for b in boxes], dtype=torch.float32)
        points, box = _sample_boxes(centers, halves, num_points, generator)
        points = points + 0.005 * torch.randn(points.shape, generator=generator)
        points = points - points.mean(0)
        clouds.append(points / points.norm(dim=-1).max())
        labels.append(torch.tensor([b[2] for b in boxes])[box])
    return torch.stack(clouds), torch.stack(labels)


PRIMITIVES = ("sphere", "cube", "cylinder")


def _primitive_shapes(num_samples: int, num_points: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy sphere / cube / cylinder surfaces (examples/train_pointnet_synthetic.py); target is the class id [B]."""
    from examples.train_pointnet_synthetic import make_shape

    labels = torch.arange(num_samples) % len(PRIMITIVES)
    clouds = [make_shape(PRIMITIVES[i], num_points, generator) for i in labels.tolist()]
    clouds = torch.stack(clouds)
    return clouds + 0.02 * torch.randn(clouds.shape, generator=generator), labels


def _partial_shapes(num_samples: int, num_points: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Primitive shapes cut by a random plane (the 40% highest along a random direction is removed).

    Returns the partial cloud resampled to ``num_points`` and the complete cloud ``[B, N, 3]`` as target.
    """
    return _cut_side(_primitive_shapes(num_samples, num_points, generator)[0], generator)


def _cut_side(full: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the 40% of each cloud ``[B, N, 3]`` highest along a random direction; resample to N points.

    Returns ``(partial [B, N, 3], full [B, N, 3])``.
    """
    num_samples, num_points, _ = full.shape
    direction = torch.randn(num_samples, 1, 3, generator=generator)
    height = (full * direction).sum(-1)
    kept = height.argsort(dim=1)[:, : max(1, int(0.6 * num_points))]
    pick = torch.randint(0, kept.shape[1], (num_samples, num_points), generator=generator)
    partial = full.gather(1, kept.gather(1, pick).unsqueeze(-1).expand(-1, -1, 3))
    return partial, full


DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in (
        Dataset("deformed_sphere", FIELD_REGRESSION, "Deformed spheres; regress the radial displacement [B, N, 1].",
                _deformed_sphere, field_name="Displacement"),
        Dataset("poisson", FIELD_REGRESSION, "Manufactured Poisson solution on [0, 1]^3; regress u [B, N, 1].",
                _poisson, field_name="u"),
        Dataset("chair_parts", PART_SEGMENTATION, "Box chairs; label each point seat / back / legs / armrests.",
                _chair_parts, class_names=CHAIR_PARTS),
        Dataset("primitive_shapes", SHAPE_CLASSIFICATION, "Noisy sphere / cube / cylinder clouds; classify the shape.",
                _primitive_shapes, class_names=PRIMITIVES),
        Dataset("partial_shapes", SHAPE_COMPLETION, "Primitive shapes with a random side cut away; complete the cloud.",
                _partial_shapes),
    )
}


# ---------------------------------------------------------------- file datasets (downloads and uploads)

# Downloads, converted .npz datasets and raw archives live here, outside the repository.
DATA_DIR = Path(os.environ.get("NEURAL3DBENCH_DATA", Path.home() / ".cache" / "neural3dbench"))
DATASET_NAME = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")


def dataset_dir() -> Path:
    return DATA_DIR / "datasets"


@functools.lru_cache(maxsize=8)
def _load_npz(path: str, mtime: float) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:  # no pickle: uploads are untrusted
        return {key: data[key] for key in data.files}


def _npz_make(path: Path) -> Generator:
    """Random ``num_samples`` shapes, each resampled to ``num_points`` of its stored points."""

    def make(num_samples: int, num_points: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        data = _load_npz(str(path), path.stat().st_mtime)
        coords = torch.from_numpy(data["coords"]).float()
        target = torch.from_numpy(data["labels"]).long() if "labels" in data else torch.from_numpy(data["target"]).float()
        rows = torch.randperm(len(coords), generator=generator)[:num_samples]
        coords, target = coords[rows], target[rows]
        stored = coords.shape[1]
        if num_points <= stored:
            points = torch.rand(len(rows), stored, generator=generator).argsort(dim=1)[:, :num_points]
        else:
            points = torch.randint(0, stored, (len(rows), num_points), generator=generator)
        coords = coords.gather(1, points.unsqueeze(-1).expand(-1, -1, 3))
        if target.ndim >= 2:  # per-point target
            index = points if target.ndim == 2 else points.unsqueeze(-1).expand(-1, -1, target.shape[-1])
            target = target.gather(1, index)
        return coords, target

    return make


def save_npz_dataset(
    name: str,
    coords: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    target: np.ndarray | None = None,
    class_names: list[str] | tuple[str, ...] = (),
    class_colors: list[str] | tuple[str, ...] = (),
    description: str = "",
) -> Dataset:
    """Validate, write ``<DATA_DIR>/datasets/<name>.npz`` and register it.

    ``coords [B, N, 3]`` plus either ``labels`` (``[B, N]`` part ids or ``[B]`` class ids) or a
    float ``target [B, N, C]``.
    """
    if not DATASET_NAME.match(name):
        raise ValueError("dataset name must be lowercase letters, digits and underscores (max 64).")
    if name in BUILTIN_DATASETS:
        raise ValueError(f"{name!r} is a built-in dataset name.")
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[2] != 3 or len(coords) < 2:
        raise ValueError(f"coords must have shape [B, N, 3] with B >= 2; got {coords.shape}.")
    if (labels is None) == (target is None):
        raise ValueError("give exactly one of labels or target.")
    arrays: dict[str, np.ndarray] = {"coords": coords, "description": np.array(description)}
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape not in (coords.shape[:2], coords.shape[:1]) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"labels must be integers of shape {coords.shape[:2]} or {coords.shape[:1]}; got {labels.shape}.")
        if labels.min() < 0:
            raise ValueError("labels must be non-negative class ids.")
        count = max(int(labels.max()) + 1, len(class_names))
        class_names = list(class_names) + [f"class {c}" for c in range(len(class_names), count)]
        arrays.update(labels=labels.astype(np.int64), class_names=np.array(class_names, dtype=str))
        if class_colors:
            if len(class_colors) != count or not all(re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in class_colors):
                raise ValueError(f"class_colors needs {count} hex colors like #1f77b4.")
            arrays["class_colors"] = np.array(class_colors, dtype=str)
    else:
        target = np.asarray(target, dtype=np.float32)
        target = target[..., None] if target.ndim == 2 else target
        if target.shape[:2] != coords.shape[:2]:
            raise ValueError(f"target must have shape [B, N, C] matching coords {coords.shape[:2]}; got {target.shape}.")
        arrays["target"] = target
    path = dataset_dir() / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return register_npz(path)


def update_classes(name: str, class_names: list[str], class_colors: list[str]) -> Dataset:
    """Rename or recolor the classes of a saved dataset; the points and labels are unchanged."""
    if name in BUILTIN_DATASETS or name not in DATASETS:
        raise ValueError(f"{name!r} is not a saved dataset.")
    data = _load_npz(str(dataset_dir() / f"{name}.npz"), (dataset_dir() / f"{name}.npz").stat().st_mtime)
    if len(class_names) != len(data["class_names"]) or not all(n.strip() for n in class_names):
        raise ValueError(f"give a non-empty name for each of the {len(data['class_names'])} classes.")
    return save_npz_dataset(name, data["coords"], labels=data["labels"], class_names=[n.strip() for n in class_names],
                            class_colors=class_colors, description=str(data["description"]))


def _completion_make(make: Generator) -> Generator:
    """Completion data from any dataset: its clouds with a random side cut away; labels are ignored."""

    def completion(num_samples: int, num_points: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        return _cut_side(make(num_samples, num_points, generator)[0], generator)

    return completion


def register_npz(path: Path) -> Dataset:
    """Register a saved ``.npz`` dataset under its file stem (re-registering replaces it).

    Also registers ``<stem>_completion``: the same clouds as a shape-completion dataset.
    """
    data = _load_npz(str(path), path.stat().st_mtime)
    if "labels" in data:
        task = PART_SEGMENTATION if data["labels"].ndim == 2 else SHAPE_CLASSIFICATION
    else:
        task = FIELD_REGRESSION
    dataset = Dataset(
        path.stem,
        task,
        str(data.get("description", "")) or f"File dataset {path.name}.",
        _npz_make(path),
        class_names=tuple(str(n) for n in data.get("class_names", ())),
        class_colors=tuple(str(c) for c in data.get("class_colors", ())),
    )
    DATASETS[dataset.name] = dataset
    DATASETS[f"{dataset.name}_completion"] = Dataset(
        f"{dataset.name}_completion", SHAPE_COMPLETION,
        f"{dataset.name} clouds with a random side cut away; complete the cloud.",
        _completion_make(dataset.make),
    )
    return dataset


def register_cached_datasets() -> list[str]:
    """Register every ``.npz`` in the dataset cache; unreadable files are skipped, not fatal."""
    names = []
    for path in sorted(dataset_dir().glob("*.npz")):
        try:
            names.append(register_npz(path).name)
        except Exception:  # noqa: BLE001 - one corrupt cache file must not break the dashboard
            continue
    return names


BUILTIN_DATASETS = frozenset(DATASETS)
register_cached_datasets()


def make_dataset(name: str, num_samples: int, num_points: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(coords [B, N, 3], target)`` for dataset ``name``; target shape depends on its task."""
    return DATASETS[name].make(num_samples, num_points, torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- advisor -> training contract


def dataset_for_problem(advisor_task: str, description: str = "") -> str:
    """Pick the synthetic dataset that matches an advisor task domain and free-text description."""
    text = description.lower()
    if advisor_task == "shape_completion":
        return "partial_shapes"
    if advisor_task == "segmentation_classification":
        if "classif" in text and "segment" not in text:
            return "primitive_shapes"
        return "chair_parts"
    return "poisson" if advisor_task == "pde_operator" else "deformed_sphere"


# ---------------------------------------------------------------- model head, loss, metrics


def head_config(model_name: str, dataset: Dataset) -> dict[str, Any]:
    """Config keys that size the model's head for ``dataset``; only keys the model's config accepts."""
    accepted = {f.name for f in dataclasses.fields(ModelRegistry.get_config_class(model_name))}
    head: dict[str, Any] = {"task": "classification" if dataset.task == SHAPE_CLASSIFICATION else "segmentation"}
    if dataset.num_classes:
        head["output_dim"] = dataset.num_classes
    return {k: v for k, v in head.items() if k in accepted}


def generates_points(model_name: str) -> bool:
    """True for models that output a new point set (their config has ``num_output_points``)."""
    return hasattr(ModelRegistry.get_config_class(model_name), "num_output_points")


def input_mismatch(model_name: str, dataset_name: str | None = None) -> str | None:
    """Why ``model_name`` cannot train on these datasets (they only provide ``coordinates [B, N, 3]``), or None.

    With ``dataset_name``, also reject a completion model on a per-point dataset and the reverse.
    """
    if dataset_name is not None:
        completion = DATASETS[dataset_name].task == SHAPE_COMPLETION
        if generates_points(model_name) != completion:
            if completion:
                return f"`{dataset_name}` is shape completion; train it with a completion model (pcn, foldingnet)."
            return f"`{model_name}` outputs a new point set; train it on a shape-completion dataset (partial_shapes)."
    capabilities = ModelRegistry.create(model_name).capabilities
    extra = sorted(set(capabilities.requires) - {"coordinates"})
    if capabilities.representation == "POINT_CLOUD" and not extra:
        return None
    needs = ", ".join(extra) if extra else f"{capabilities.representation.lower()} coordinates"
    return (f"`{model_name}` takes {capabilities.representation.lower().replace('_', ' ')} input and needs {needs}; "
            "the dashboard datasets only provide point coordinates [B, N, 3]. Pick a point-cloud model "
            "(pointnet, pointnet2, dgcnn, pointnext, transolver, mlp, siren).")


def forward(task: str, model: nn.Module, coords: torch.Tensor) -> torch.Tensor:
    """Run the model; returns ``[B, C]`` logits for shape classification, else ``[B, N, C]``."""
    output = model(ModelInput(coordinates=coords))
    predictions = output.predictions if isinstance(output, ModelOutput) else output
    if task == SHAPE_COMPLETION:
        return predictions  # [B, M, 3], M set by the model
    if task == SHAPE_CLASSIFICATION:
        if predictions.ndim == 3:  # per-point model: pool the point logits into one shape logit
            predictions = predictions.mean(dim=1)
        return predictions.reshape(coords.shape[0], -1)
    return predictions.reshape(coords.shape[0], coords.shape[1], -1)


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric squared-L2 Chamfer distance between ``[B, M, 3]`` and ``[B, N, 3]``, averaged over the batch."""
    # Expanded form avoids torch.cdist's NaN gradient at zero distance.
    sq = (pred.pow(2).sum(-1, keepdim=True) + target.pow(2).sum(-1).unsqueeze(1)
          - 2 * pred @ target.transpose(1, 2)).clamp_min(0)  # [B, M, N]
    return sq.min(2).values.mean() + sq.min(1).values.mean()


def loss(task: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if task == SHAPE_COMPLETION:
        # shortcut: no Chamfer term on PCN's coarse output (paper uses one); add it if PCN converges slowly.
        return chamfer_distance(pred, target)
    if task == FIELD_REGRESSION:
        return F.mse_loss(pred, target)
    return F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))


def mean_iou(pred_labels: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """IoU averaged over the classes present in the prediction or the ground truth."""
    ious = []
    for c in range(num_classes):
        p, t = pred_labels == c, target == c
        union = (p | t).sum().item()
        if union:
            ious.append((p & t).sum().item() / union)
    return sum(ious) / len(ious) if ious else 0.0


def metrics(task: str, pred: torch.Tensor, target: torch.Tensor, num_classes: int = 0) -> dict[str, float]:
    if task == FIELD_REGRESSION:
        return {"mae": (pred - target).abs().mean().item()}
    if task == SHAPE_COMPLETION:
        return {"chamfer": chamfer_distance(pred, target).item()}
    labels = pred.argmax(-1)
    result = {"accuracy": (labels == target).float().mean().item()}
    if task == PART_SEGMENTATION:
        result["miou"] = mean_iou(labels, target, num_classes)
    return result


def evaluation_figure(dataset: Dataset, coords: Any, target: Any, pred: Any) -> Any:
    """Plotly preview of one sample: ``coords [N, 3]``, its target and the raw model output (no batch dim)."""
    from tools.dashboard import visualizer  # plotly only needed here

    pred = torch.as_tensor(pred)
    if dataset.task == FIELD_REGRESSION:
        return visualizer.plot_field_comparison(coords, None, target, pred, dataset.field_name)
    if dataset.task == SHAPE_COMPLETION:
        return visualizer.plot_completion_comparison(coords, target, pred)
    if dataset.task == PART_SEGMENTATION:
        return visualizer.plot_segmentation_comparison(coords, target, pred.argmax(-1), dataset.class_names,
                                                       dataset.class_colors)
    probs = pred.softmax(-1)
    true, guess = dataset.class_names[int(target)], dataset.class_names[int(probs.argmax())]
    verdict = "correct" if true == guess else "wrong"
    return visualizer.plot_point_cloud(coords, title=f"True: {true} · predicted: {guess} ({probs.max():.0%}, {verdict})")
