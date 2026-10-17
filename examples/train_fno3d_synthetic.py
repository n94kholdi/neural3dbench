"""Train FNO3D on a synthetic source-to-smoothed-solution operator."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from models import FNO3DConfig, ModelInput, create_model
from volumetric import plot_volume_slices


def make_operator_data(
    samples: int, resolution: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create varied source fields and deterministic Helmholtz-like responses."""
    noise = torch.randn(samples, 1, resolution, resolution, resolution, generator=generator)
    source_ft = torch.fft.rfftn(noise, dim=(-3, -2, -1))
    kd = torch.fft.fftfreq(resolution).reshape(1, 1, resolution, 1, 1)
    kh = torch.fft.fftfreq(resolution).reshape(1, 1, 1, resolution, 1)
    kw = torch.fft.rfftfreq(resolution).reshape(1, 1, 1, 1, -1)
    squared_frequency = kd.square() + kh.square() + kw.square()

    # Smooth random source functions, then apply a sample-dependent inverse
    # elliptic multiplier. Alpha is supplied as a dense physical input channel.
    source = torch.fft.irfftn(
        source_ft * torch.exp(-24.0 * squared_frequency),
        s=(resolution, resolution, resolution),
        dim=(-3, -2, -1),
    )
    source = source / source.flatten(1).std(dim=1).view(-1, 1, 1, 1, 1).clamp_min(1e-6)
    alpha = 0.25 + 0.75 * torch.rand(samples, 1, 1, 1, 1, generator=generator)
    source_ft = torch.fft.rfftn(source, dim=(-3, -2, -1))
    target = torch.fft.irfftn(
        source_ft / (1.0 + alpha * 80.0 * squared_frequency),
        s=(resolution, resolution, resolution),
        dim=(-3, -2, -1),
    )
    inputs = torch.cat((source, alpha.expand_as(source)), dim=1)
    return inputs.float(), target.float()


def minibatches(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
):
    for indices in torch.randperm(inputs.shape[0], generator=generator).split(batch_size):
        yield inputs[indices], targets[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--projection-channels", type=int, default=16)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fno3d_synthetic.pt"))
    args = parser.parse_args()
    if min(args.epochs, args.samples, args.resolution, args.batch_size, args.width, args.layers, args.modes) < 1:
        parser.error("positive values are required for sizes, epochs, layers, width, and modes")
    if args.modes > args.resolution // 2:
        parser.error("modes must not exceed resolution // 2")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    inputs, targets = make_operator_data(args.samples, args.resolution, generator)
    device = torch.device(args.device)
    config = FNO3DConfig(
        input_channels=2,
        output_channels=1,
        width=args.width,
        num_layers=args.layers,
        modes=(args.modes,) * 3,
        projection_channels=args.projection_channels,
        append_coordinates=True,
        padding=args.padding,
    )
    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_inputs, batch_targets in minibatches(inputs, targets, args.batch_size, generator):
            prediction = model(
                ModelInput(voxel_fields=batch_inputs.to(device), representation="GRID")
            ).predictions
            loss = F.mse_loss(prediction, batch_targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch:03d} | mse={sum(losses) / len(losses):.6f}")

    model.eval()
    with torch.no_grad():
        prediction = model(ModelInput(voxel_fields=inputs[:1].to(device))).predictions.cpu()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()}, args.checkpoint)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    restored = create_model(checkpoint["config"]).to(device).eval()
    restored.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        restored_prediction = restored(ModelInput(voxel_fields=inputs[:1].to(device))).predictions.cpu()
    torch.testing.assert_close(restored_prediction, prediction)
    print(f"Checkpoint saved and reloaded: {args.checkpoint}")

    if args.visualize:
        plot_volume_slices(
            inputs[0, :1],
            ground_truth=targets[0],
            prediction=prediction[0],
            title="FNO3D synthetic operator: source / target / prediction",
        )


if __name__ == "__main__":
    main()
