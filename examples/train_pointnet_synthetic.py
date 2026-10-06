"""Small PointNet/PointNet++ checks for classification and point regression."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from models import ModelInput, PointNet2Config, PointNetConfig, create_model
from models.pointnet import feature_transform_regularizer
from pointcloud import normalize_points, plot_point_cloud


SHAPES = ("sphere", "cube", "cylinder")


def make_shape(name: str, count: int, generator: torch.Generator) -> torch.Tensor:
    if name == "sphere":
        points = torch.randn((count, 3), generator=generator)
        return points / torch.linalg.vector_norm(points, dim=1, keepdim=True)
    if name == "cube":
        points = 2 * torch.rand((count, 3), generator=generator) - 1
        face_axis = torch.randint(3, (count,), generator=generator)
        face_sign = 2 * torch.randint(2, (count,), generator=generator) - 1
        points[torch.arange(count), face_axis] = face_sign.float()
        return normalize_points(points)
    if name == "cylinder":
        theta = 2 * torch.pi * torch.rand(count, generator=generator)
        height = 2 * torch.rand(count, generator=generator) - 1
        return torch.stack((theta.cos(), theta.sin(), height), dim=1)
    raise ValueError(f"Unknown shape: {name}")


def make_classification_data(
    sample_count: int, point_count: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    clouds, labels = [], []
    for index in range(sample_count):
        label = index % len(SHAPES)
        cloud = make_shape(SHAPES[label], point_count, generator)
        cloud = cloud + 0.02 * torch.randn(cloud.shape, generator=generator)
        clouds.append(cloud)
        labels.append(label)
    return torch.stack(clouds), torch.tensor(labels)


def make_regression_data(
    sample_count: int, point_count: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    points = 2 * torch.rand(
        (sample_count, point_count, 3), generator=generator
    ) - 1
    targets = points.square().sum(dim=-1, keepdim=True)
    return points, targets


def iterate_minibatches(
    points: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
):
    for indices in torch.randperm(points.shape[0], generator=generator).split(batch_size):
        yield points[indices], targets[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pointnet", "pointnet2"), default="pointnet")
    parser.add_argument(
        "--task", choices=("classification", "segmentation"), default="classification"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.samples, args.num_points, args.batch_size) < 1:
        parser.error("epochs, samples, num-points, and batch-size must be positive")
    if args.samples < 3:
        parser.error("samples must be at least 3 to form training and validation sets")
    if args.batch_size < 2:
        parser.error("batch-size must be at least 2 because PointNet uses batch normalization")

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    if args.task == "classification":
        points, targets = make_classification_data(
            args.samples, args.num_points, generator
        )
        output_dim = len(SHAPES)
    else:
        points, targets = make_regression_data(args.samples, args.num_points, generator)
        output_dim = 1

    split = max(2, int(0.8 * args.samples))
    split = min(split, args.samples - 1) if args.samples > 2 else args.samples
    train_points, validation_points = points[:split], points[split:]
    train_targets, validation_targets = targets[:split], targets[split:]
    if validation_points.shape[0] == 0:
        validation_points, validation_targets = train_points, train_targets

    common_config = {
        "input_dim": 3,
        "output_dim": output_dim,
        "task": args.task,
        "global_dim": 1024,
        "dropout": 0.3,
    }
    if args.model == "pointnet2":
        first_count = min(64, args.num_points)
        config = PointNet2Config(
            sample_counts=(first_count, min(16, first_count)),
            radii=(0.25, 0.5),
            neighbor_counts=(16, 32),
            **common_config,
        )
    else:
        config = PointNetConfig(**common_config)
    model = create_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint = args.checkpoint or Path(f"checkpoints/{args.model}_{args.task}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_points, batch_targets in iterate_minibatches(
            train_points, train_targets, args.batch_size, generator
        ):
            # Avoid a final one-item batch because BatchNorm needs two examples.
            if batch_points.shape[0] == 1:
                continue
            output = model(ModelInput(coordinates=batch_points))
            prediction = output.predictions
            loss = (
                F.cross_entropy(prediction, batch_targets)
                if args.task == "classification"
                else F.mse_loss(prediction, batch_targets)
            )
            feature_transform = output.auxiliary.get("feature_transform")
            if feature_transform is not None:
                loss = loss + 1e-3 * feature_transform_regularizer(feature_transform)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            validation_prediction = model(
                ModelInput(coordinates=validation_points)
            ).predictions
            if args.task == "classification":
                metric = (validation_prediction.argmax(1) == validation_targets).float().mean()
                metric_text = f"accuracy={metric.item():.3f}"
            else:
                metric = F.mse_loss(validation_prediction, validation_targets)
                metric_text = f"mse={metric.item():.6f}"
        print(f"Epoch {epoch:03d} | loss={sum(losses) / len(losses):.6f} | {metric_text}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict()}, checkpoint
    )
    print(f"Checkpoint: {checkpoint}")

    if args.visualize:
        if args.task == "classification":
            predicted = int(validation_prediction[0].argmax())
            plot_point_cloud(
                validation_points[0],
                predicted_class=SHAPES[predicted],
                title=f"{args.model} shape classification",
            )
        else:
            plot_point_cloud(
                validation_points[0],
                ground_truth=validation_targets[0],
                prediction=validation_prediction[0],
                title=f"{args.model} radial field",
            )


if __name__ == "__main__":
    main()
