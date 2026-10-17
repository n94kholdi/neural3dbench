"""Measure Transolver forward/training time and CUDA peak memory by point count."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from models import ModelInput, TransolverConfig, create_model


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-counts", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-slices", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    model = create_model(
        TransolverConfig(
            spatial_dim=3,
            input_dim=5,
            output_dim=4,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_slices=args.num_slices,
        )
    ).to(device)
    print("points\tforward_ms\tstep_ms\tpeak_gpu_mib")
    for point_count in args.point_counts:
        coordinates = torch.randn(args.batch_size, point_count, 3, device=device)
        features = torch.randn(args.batch_size, point_count, 5, device=device)
        inputs = ModelInput(coordinates=coordinates, point_features=features)
        for _ in range(args.warmup):
            model(inputs).predictions.sum().backward()
            model.zero_grad(set_to_none=True)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        forward_times, step_times = [], []
        for _ in range(args.repeats):
            model.eval()
            synchronize(device)
            start = time.perf_counter()
            with torch.no_grad():
                model(inputs)
            synchronize(device)
            forward_times.append((time.perf_counter() - start) * 1000)

            model.train()
            start = time.perf_counter()
            model(inputs).predictions.square().mean().backward()
            synchronize(device)
            step_times.append((time.perf_counter() - start) * 1000)
            model.zero_grad(set_to_none=True)
        memory = (
            f"{torch.cuda.max_memory_allocated(device) / 2**20:.1f}"
            if device.type == "cuda" else "n/a"
        )
        print(
            f"{point_count}\t{statistics.median(forward_times):.2f}\t"
            f"{statistics.median(step_times):.2f}\t{memory}"
        )


if __name__ == "__main__":
    main()
