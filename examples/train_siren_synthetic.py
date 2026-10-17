"""Fit SIREN to a continuous multi-frequency scalar field in 3D."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models import ModelInput, SIRENConfig, create_model
from pointcloud import pointwise_regression_loss


def target_field(coordinates: torch.Tensor) -> torch.Tensor:
    """A deterministic field with low- and high-frequency components."""
    x, y, z = coordinates.unbind(-1)
    values = (
        0.6 * torch.sin(2.0 * torch.pi * x) * torch.cos(3.0 * torch.pi * y)
        + 0.3 * torch.sin(5.0 * torch.pi * z)
        + 0.1 * torch.cos(4.0 * torch.pi * (x + y - z))
    )
    return values.unsqueeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--train-points", type=int, default=512)
    parser.add_argument("--evaluation-points", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/siren_synthetic.pt"),
    )
    args = parser.parse_args()
    if min(
        args.steps,
        args.train_points,
        args.evaluation_points,
        args.hidden_dim,
    ) < 1:
        parser.error("steps, point counts, and hidden-dim must be positive")
    if args.num_hidden_layers < 0:
        parser.error("num-hidden-layers cannot be negative")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_coordinates = 2.0 * torch.rand(
        1, args.train_points, 3, generator=generator
    ) - 1.0
    evaluation_coordinates = 2.0 * torch.rand(
        1, args.evaluation_points, 3, generator=generator
    ) - 1.0
    train_targets = target_field(train_coordinates)
    evaluation_targets = target_field(evaluation_coordinates)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    config = SIRENConfig(
        spatial_dim=3,
        input_dim=0,
        output_dim=1,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    )
    model = create_model(config).to(device)
    train_input = ModelInput(coordinates=train_coordinates.to(device))
    evaluation_input = ModelInput(coordinates=evaluation_coordinates.to(device))
    train_targets = train_targets.to(device)
    evaluation_targets = evaluation_targets.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.eval()
    with torch.no_grad():
        initial_train = pointwise_regression_loss(
            model(train_input).predictions, train_targets
        ).item()
        initial_evaluation = pointwise_regression_loss(
            model(evaluation_input).predictions, evaluation_targets
        ).item()
    print(
        f"Initial | train_mse={initial_train:.6f} | "
        f"evaluation_mse={initial_evaluation:.6f}"
    )

    model.train()
    for _ in range(args.steps):
        prediction = model(train_input).predictions
        loss = pointwise_regression_loss(prediction, train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_train = pointwise_regression_loss(
            model(train_input).predictions, train_targets
        ).item()
        evaluation_prediction = model(evaluation_input).predictions
        final_evaluation = pointwise_regression_loss(
            evaluation_prediction, evaluation_targets
        ).item()
    print(
        f"Final   | train_mse={final_train:.6f} | "
        f"evaluation_mse={final_evaluation:.6f}"
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict()},
        args.checkpoint,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    restored = create_model(checkpoint["config"]).to(device).eval()
    restored.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        restored_prediction = restored(evaluation_input).predictions
    torch.testing.assert_close(restored_prediction, evaluation_prediction)
    print(f"Checkpoint saved, reloaded, and verified: {args.checkpoint}")


if __name__ == "__main__":
    main()
