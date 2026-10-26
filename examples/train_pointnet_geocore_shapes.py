"""Train PointNet to classify surface point clouds created by geocore.

The example creates sphere, box, and cylinder problems with geocore, saves one
canonical ``point_cloud.npz``, trains PointNet, and saves validation results.

Run from the workspace root:

    .venv/bin/python neural3dbench/examples/train_pointnet_geocore_shapes.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
NEURAL3DBENCH_ROOT = WORKSPACE_ROOT / "neural3dbench"
GEOCORE_ROOT = WORKSPACE_ROOT / "geometry-py" / "geocore"
for source_root in (NEURAL3DBENCH_ROOT, GEOCORE_ROOT):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import geocore as gc  # noqa: E402
from models import (  # noqa: E402
    ModelInput,
    PointCloudRepresentationAdapter,
    PointNetConfig,
    create_model,
)
from models.pointnet import feature_transform_regularizer  # noqa: E402


CLASS_NAMES = ("sphere", "box", "cylinder")
METADATA_KEYS = (
    "schema_version",
    "representation",
    "coordinate_system",
    "units",
    "normalization_id",
)


def metadata_from(raw: dict) -> dict:
    metadata = {}
    for key in METADATA_KEYS:
        value = np.asarray(raw[key])
        metadata[key] = value.item() if value.ndim == 0 else value.tolist()
    return metadata


def make_geometries(resolution: float = 0.12) -> tuple[gc.Geometry, ...]:
    """Create three cached geocore geometry problems."""
    return (
        gc.Geometry(gc.Sphere((0.0, 0.0, 0.0), 1.0), resolution),
        gc.Geometry(gc.Box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)), resolution),
        gc.Geometry(gc.Cylinder((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), 0.72), resolution),
    )


def create_cloud(geometry: gc.Geometry, point_count: int, seed: int) -> dict:
    cloud = dict(
        geometry.output(
            "POINT_CLOUD",
            n_points=point_count,
            mode="surface",
            features=None,
            normalization="unit_sphere",
            rng=seed,
        )
    )
    gc.validate(cloud, input_dim=3)
    return cloud


def adapt_cloud(cloud: dict) -> ModelInput:
    return PointCloudRepresentationAdapter().adapt(
        {**cloud, "metadata": metadata_from(cloud)}
    )


def make_dataset(
    geometries: tuple[gc.Geometry, ...],
    samples_per_class: int,
    point_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    points, labels = [], []
    for label, geometry in enumerate(geometries):
        for sample_index in range(samples_per_class):
            cloud = create_cloud(
                geometry,
                point_count,
                seed + label * samples_per_class + sample_index,
            )
            model_input = adapt_cloud(cloud)
            points.append(model_input.coordinates[0])
            labels.append(label)

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(points), generator=generator)
    return torch.stack(points)[order], torch.tensor(labels, dtype=torch.long)[order]


def iterate_minibatches(
    points: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
):
    for indices in torch.randperm(len(points), generator=generator).split(batch_size):
        if len(indices) > 1:  # PointNet's BatchNorm requires two training examples.
            yield points[indices], labels[indices]


def evaluate(model, points: torch.Tensor, labels: torch.Tensor, device: torch.device):
    model.eval()
    with torch.no_grad():
        logits = model(
            ModelInput(
                coordinates=points.to(device),
                representation="POINT_CLOUD",
            )
        ).predictions
        loss = nn.functional.cross_entropy(logits, labels.to(device)).item()
        prediction = logits.argmax(dim=1)
        accuracy = (prediction == labels.to(device)).float().mean().item()
    return loss, accuracy, logits.cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--train-samples-per-class", type=int, default=24)
    parser.add_argument("--validation-samples-per-class", type=int, default=6)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--global-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pointnet_geocore_shapes"),
    )
    args = parser.parse_args()
    if min(
        args.epochs,
        args.train_samples_per_class,
        args.validation_samples_per_class,
        args.num_points,
        args.batch_size,
        args.global_dim,
    ) < 1:
        parser.error(
            "epochs, sample counts, point count, batch size, and global dim "
            "must be positive"
        )
    if args.batch_size < 2:
        parser.error("batch-size must be at least 2 because PointNet uses batch normalization")
    return args


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    geometries = make_geometries()
    canonical = create_cloud(geometries[0], args.num_points, args.seed)
    canonical_path = Path(gc.save(str(args.output_dir / "point_cloud.npz"), canonical))

    train_points, train_labels = make_dataset(
        geometries,
        args.train_samples_per_class,
        args.num_points,
        args.seed + 100,
    )
    validation_points, validation_labels = make_dataset(
        geometries,
        args.validation_samples_per_class,
        args.num_points,
        args.seed + 10_000,
    )

    config = PointNetConfig(
        input_dim=3,
        output_dim=len(CLASS_NAMES),
        task="classification",
        global_dim=args.global_dim,
        dropout=0.2,
    )
    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed + 1)

    best_accuracy = -1.0
    best_validation_loss = float("inf")
    best_state = None
    log_every = max(1, args.epochs // 10)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_points, batch_labels in iterate_minibatches(
            train_points, train_labels, args.batch_size, generator
        ):
            output = model(
                ModelInput(
                    coordinates=batch_points.to(device),
                    representation="POINT_CLOUD",
                )
            )
            loss = nn.functional.cross_entropy(
                output.predictions, batch_labels.to(device)
            )
            transform = output.auxiliary.get("feature_transform")
            if transform is not None:
                loss = loss + 1e-3 * feature_transform_regularizer(transform)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        validation_loss, validation_accuracy, _ = evaluate(
            model, validation_points, validation_labels, device
        )
        if (validation_accuracy, -validation_loss) > (
            best_accuracy,
            -best_validation_loss,
        ):
            best_accuracy = validation_accuracy
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | train loss {np.mean(losses):.6f} | "
                f"validation loss {validation_loss:.6f} | accuracy {validation_accuracy:.3f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    final_loss, final_accuracy, logits = evaluate(
        model, validation_points, validation_labels, device
    )
    probabilities = logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)

    checkpoint_path = args.output_dir / "pointnet_shapes.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": config.to_dict(),
            "class_names": list(CLASS_NAMES),
            "normalization_strategy": "per_sample_unit_sphere",
            "normalization_id": metadata_from(canonical)["normalization_id"],
            "best_validation_loss": best_validation_loss,
            "best_validation_accuracy": best_accuracy,
        },
        checkpoint_path,
    )
    result_path = args.output_dir / "validation_result.npz"
    np.savez_compressed(
        result_path,
        coordinates=validation_points.numpy(),
        targets=validation_labels.numpy(),
        predictions=predictions.numpy(),
        probabilities=probabilities.numpy(),
        class_names=np.asarray(CLASS_NAMES, dtype=np.str_),
    )

    print("\nTraining complete")
    print(f"  device: {device}")
    print(f"  training / validation clouds: {len(train_points)} / {len(validation_points)}")
    print(f"  points per cloud: {args.num_points}")
    print(f"  validation loss / accuracy: {final_loss:.6f} / {final_accuracy:.3f}")
    print(f"  canonical point cloud: {canonical_path}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  validation results: {result_path}")


if __name__ == "__main__":
    main()
