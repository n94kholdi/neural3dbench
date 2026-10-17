"""Benchmark FNO3D latency, training-step time, memory, and parameters."""

from __future__ import annotations

import argparse
import time

import torch

from models import FNO3DConfig, ModelInput, create_model


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mean_time(action, iterations: int, device: torch.device) -> float:
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        action()
    synchronize(device)
    return 1000.0 * (time.perf_counter() - start) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-channels", type=int, default=2)
    parser.add_argument("--output-channels", type=int, default=1)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--projection-channels", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(*args.resolutions, args.batch_size, args.width, args.layers, args.modes, args.iterations) < 1:
        parser.error("resolutions, sizes, layers, modes, and iterations must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if any(args.modes > resolution // 2 for resolution in args.resolutions):
        parser.error("modes must not exceed resolution // 2 for any resolution")

    device = torch.device(args.device)
    print("resolution,parameters,forward_ms,forward_backward_ms,peak_cuda_mib")
    for resolution in args.resolutions:
        model = create_model(FNO3DConfig(
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            width=args.width,
            num_layers=args.layers,
            modes=(args.modes,) * 3,
            projection_channels=args.projection_channels,
        )).to(device)
        x = torch.randn(
            args.batch_size,
            args.input_channels,
            resolution,
            resolution,
            resolution,
            device=device,
        )
        inputs = ModelInput(voxel_fields=x, representation="GRID")

        def forward_only() -> None:
            with torch.no_grad():
                model(inputs)

        def training_step() -> None:
            model.zero_grad(set_to_none=True)
            model(inputs).predictions.square().mean().backward()

        for _ in range(args.warmup):
            forward_only()
            training_step()
        if device.type == "cuda":
            synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        forward_ms = mean_time(forward_only, args.iterations, device)
        train_ms = mean_time(training_step, args.iterations, device)
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else float("nan")
        )
        parameters = sum(parameter.numel() for parameter in model.parameters())
        peak_text = f"{peak_memory:.3f}" if device.type == "cuda" else "N/A"
        print(f"{resolution},{parameters},{forward_ms:.3f},{train_ms:.3f},{peak_text}")


if __name__ == "__main__":
    main()
