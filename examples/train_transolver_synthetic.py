"""Train Transolver on a smooth scalar or multi-field function over random points."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from models import ModelInput, TransolverConfig, create_model
from pointcloud import plot_point_cloud, pointwise_regression_loss


def target_field(points: torch.Tensor, output_dim: int) -> torch.Tensor:
    x, y = points.unbind(-1)
    fields = [torch.sin(torch.pi * x) * torch.cos(torch.pi * y)]
    if output_dim > 1:
        fields.extend((x.square() + y.square(), torch.sin(torch.pi * (x + y))))
    while len(fields) < output_dim:
        fields.append(fields[-1] * fields[0])
    return torch.stack(fields[:output_dim], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-slices", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/transolver_synthetic.pt"))
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.samples, args.num_points, args.output_dim) < 1:
        parser.error("epochs, samples, num-points, and output-dim must be positive")

    torch.manual_seed(7)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    points = 2 * torch.rand(args.samples, args.num_points, 2) - 1
    targets = target_field(points, args.output_dim)
    split = max(1, int(0.8 * args.samples))
    train_points, validation_points = points[:split], points[split:]
    train_targets, validation_targets = targets[:split], targets[split:]
    if validation_points.numel() == 0:
        validation_points, validation_targets = train_points, train_targets

    config = TransolverConfig(
        spatial_dim=2,
        input_dim=0,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_slices=args.num_slices,
        mlp_ratio=2,
    )
    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_input = ModelInput(coordinates=train_points.to(device))
    validation_input = ModelInput(coordinates=validation_points.to(device))
    train_targets = train_targets.to(device)
    validation_targets = validation_targets.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        prediction = model(train_input).predictions
        loss = F.mse_loss(prediction, train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                validation_prediction = model(validation_input).predictions
                validation_mse = F.mse_loss(validation_prediction, validation_targets)
                validation_rel_l2 = pointwise_regression_loss(
                    validation_prediction, validation_targets, loss="relative_l2"
                )
            print(
                f"Epoch {epoch:04d} | train_mse={loss.item():.6f} | "
                f"val_mse={validation_mse.item():.6f} | rel_l2={validation_rel_l2.item():.4f}"
            )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()}, args.checkpoint)
    print(f"Checkpoint: {args.checkpoint}")
    if args.visualize:
        xyz = F.pad(validation_points[0], (0, 1))
        plot_point_cloud(
            xyz,
            ground_truth=validation_targets[0, :, 0].cpu(),
            prediction=validation_prediction[0, :, 0].cpu(),
            title="Transolver synthetic field",
        )


if __name__ == "__main__":
    main()
