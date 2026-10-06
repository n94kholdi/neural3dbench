"""Train 3D U-Net to predict a radial scalar field inside synthetic spheres."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from models import ModelInput, UNet3DConfig, create_model
from volumetric import plot_volume_slices


def make_sphere_fields(
    samples: int, resolution: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-1.0, 1.0, resolution)
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    coordinates = torch.stack((x, y, z))
    inputs = []
    targets = []
    for _ in range(samples):
        center = torch.empty(3).uniform_(-0.25, 0.25, generator=generator)
        radius = float(torch.empty(1).uniform_(0.45, 0.8, generator=generator))
        relative = coordinates - center.view(3, 1, 1, 1)
        squared_radius = relative.square().sum(dim=0, keepdim=True)
        geometry_mask = (squared_radius <= radius**2).float()
        # Geometry is one ordinary input channel; XYZ channels expose location.
        inputs.append(torch.cat((geometry_mask, coordinates), dim=0))
        targets.append(squared_radius * geometry_mask)
    return torch.stack(inputs), torch.stack(targets)


def iterate_minibatches(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
):
    for indices in torch.randperm(inputs.shape[0], generator=generator).split(batch_size):
        yield inputs[indices], targets[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/unet3d_sphere.pt"))
    args = parser.parse_args()
    if min(args.epochs, args.samples, args.resolution, args.batch_size) < 1:
        parser.error("epochs, samples, resolution, and batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if args.resolution < 2 ** (args.levels - 1):
        parser.error("resolution is too small for the requested number of levels")

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    inputs, targets = make_sphere_fields(args.samples, args.resolution, generator)
    device = torch.device(args.device)
    model = create_model(
        UNet3DConfig(
            input_channels=4,
            output_channels=1,
            base_channels=args.base_channels,
            levels=args.levels,
            normalization="group",
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_inputs, batch_targets in iterate_minibatches(
            inputs, targets, args.batch_size, generator
        ):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            prediction = model(ModelInput(voxel_fields=batch_inputs)).predictions
            loss = F.mse_loss(prediction, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch:03d} | mse={sum(losses) / len(losses):.6f}")

    model.eval()
    with torch.no_grad():
        prediction = model(ModelInput(voxel_fields=inputs[:1].to(device))).predictions.cpu()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": model.config.to_dict()},
        args.checkpoint,
    )
    print(f"Checkpoint: {args.checkpoint}")
    if args.visualize:
        plot_volume_slices(
            inputs[0, :1],
            ground_truth=targets[0],
            prediction=prediction[0],
            title="3D U-Net synthetic sphere field",
        )


if __name__ == "__main__":
    main()
