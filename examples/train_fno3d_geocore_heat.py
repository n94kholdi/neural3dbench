"""Train FNO3D on geocore voxelized geometry and volumetric heat sources.

Each sample is a sphere or rounded box in a fixed physical domain. Geocore
rasterizes occupancy, signed distance, boundary, and source channels. A small
reference diffusion operator produces the target temperature field.

Run from the workspace root:

    .venv/bin/python neural3dbench/examples/train_fno3d_geocore_heat.py

This is an educational dense-grid operator problem, not a validated heat solver.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
NEURAL3DBENCH_ROOT = WORKSPACE_ROOT / "neural3dbench"
GEOCORE_ROOT = WORKSPACE_ROOT / "geometry-py" / "geocore"
for source_root in (NEURAL3DBENCH_ROOT, GEOCORE_ROOT):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import geocore as gc  # noqa: E402
from geocore_fno3d_integration import adapt_voxel_grid, metadata_from  # noqa: E402
from models import FNO3DConfig, ModelInput, create_model  # noqa: E402


CHANNEL_NAMES = ("occupancy", "sdf", "boundary", "source")
DOMAIN_BOUNDS = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))


def create_problem(resolution: int, seed: int) -> dict:
    """Create one varied geocore geometry/source problem on a fixed grid."""
    rng = np.random.default_rng(seed)
    center = rng.uniform(-0.18, 0.18, 3)
    if seed % 2 == 0:
        geometry = gc.Sphere(tuple(center), float(rng.uniform(0.42, 0.64)))
    else:
        half_size = rng.uniform(0.34, 0.58, 3)
        geometry = gc.Box(tuple(center), tuple(2.0 * half_size), rounding=0.1)

    hot_spot = center + rng.uniform(-0.2, 0.2, 3)
    sharpness = float(rng.uniform(14.0, 28.0))

    def source(points: np.ndarray) -> np.ndarray:
        offset = points - hot_spot
        return np.exp(-sharpness * np.sum(offset * offset, axis=1))

    problem = dict(
        gc.voxel_grid(
            geometry,
            shape=resolution,
            bounds=DOMAIN_BOUNDS,
            channels=CHANNEL_NAMES,
            fields={"source": source},
            normalization="unit_cube",
        )
    )
    gc.validate(problem, input_channels=len(CHANNEL_NAMES))
    return problem


def heat_response(voxel_fields: torch.Tensor, steps: int = 6) -> torch.Tensor:
    """Diffuse an internal heat source while damping temperature outside solids."""
    occupancy = voxel_fields[:, 0:1].clamp(0.0, 1.0)
    source = voxel_fields[:, 3:4].clamp_min(0.0) * occupancy
    temperature = source
    for _ in range(steps):
        neighbor_average = F.avg_pool3d(temperature, kernel_size=3, stride=1, padding=1)
        temperature = (0.35 * source + 0.65 * neighbor_average) * occupancy
    return temperature


def make_dataset(samples: int, resolution: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    fields = []
    for index in range(samples):
        problem = create_problem(resolution, seed + index)
        fields.append(adapt_voxel_grid(problem).voxel_fields[0])
    inputs = torch.stack(fields).float()
    return inputs, heat_response(inputs)


def minibatches(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
):
    for indices in torch.randperm(len(inputs), generator=generator).split(batch_size):
        yield inputs[indices], targets[indices]


def evaluate(model, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sample in inputs.split(2):
            predictions.append(
                model(
                    ModelInput(voxel_fields=sample.to(device), representation="GRID")
                ).predictions.cpu()
            )
    prediction = torch.cat(predictions)
    mse = F.mse_loss(prediction, targets).item()
    mae = F.l1_loss(prediction, targets).item()
    relative_l2 = (
        torch.linalg.vector_norm(prediction - targets)
        / torch.linalg.vector_norm(targets).clamp_min(1e-12)
    ).item()
    return mse, mae, relative_l2, prediction


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-samples", type=int, default=16)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--modes", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fno3d_geocore_heat"),
    )
    args = parser.parse_args()
    if min(
        args.epochs,
        args.train_samples,
        args.validation_samples,
        args.batch_size,
        args.width,
        args.layers,
        args.modes,
    ) < 1:
        parser.error("epochs, sample counts, batch size, width, layers, and modes must be positive")
    if args.resolution < 4:
        parser.error("resolution must be at least 4")
    if args.modes > args.resolution // 2:
        parser.error("modes must not exceed resolution // 2")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    canonical = create_problem(args.resolution, args.seed)
    canonical_path = Path(gc.save(str(args.output_dir / "voxel_grid.npz"), canonical))
    reloaded = gc.load(str(canonical_path))
    gc.validate(reloaded, input_channels=len(CHANNEL_NAMES))
    canonical_input = adapt_voxel_grid(reloaded)
    assert canonical_input.representation == "GRID"

    train_inputs, train_targets = make_dataset(
        args.train_samples, args.resolution, args.seed + 100
    )
    validation_inputs, validation_targets = make_dataset(
        args.validation_samples, args.resolution, args.seed + 10_000
    )

    config = FNO3DConfig(
        input_channels=len(CHANNEL_NAMES),
        output_channels=1,
        width=args.width,
        num_layers=args.layers,
        modes=(args.modes,) * 3,
        projection_channels=max(8, args.width * 2),
        append_coordinates=True,
    )
    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed + 1)

    best_validation_mse = float("inf")
    best_state = None
    log_every = max(1, args.epochs // 10)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_inputs, batch_targets in minibatches(
            train_inputs, train_targets, args.batch_size, generator
        ):
            prediction = model(
                ModelInput(
                    voxel_fields=batch_inputs.to(device),
                    representation="GRID",
                )
            ).predictions
            loss = nn.functional.mse_loss(prediction, batch_targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        validation_mse, validation_mae, validation_relative_l2, _ = evaluate(
            model, validation_inputs, validation_targets, device
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | train MSE {np.mean(losses):.6f} | "
                f"validation MSE {validation_mse:.6f} | MAE {validation_mae:.6f} | "
                f"relative L2 {validation_relative_l2:.4f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    final_mse, final_mae, final_relative_l2, predictions = evaluate(
        model, validation_inputs, validation_targets, device
    )

    checkpoint_path = args.output_dir / "fno3d_geocore_heat.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": config.to_dict(),
            "channel_names": list(CHANNEL_NAMES),
            "normalization_id": metadata_from(reloaded)["normalization_id"],
            "reference_operator": {"name": "masked_avg_pool_diffusion", "steps": 6},
            "best_validation_mse": best_validation_mse,
            "validation_mae": final_mae,
            "validation_relative_l2": final_relative_l2,
        },
        checkpoint_path,
    )

    # Verify that the saved checkpoint reproduces the evaluated prediction.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored = create_model(checkpoint["config"]).to(device).eval()
    restored.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        restored_first = restored(
            ModelInput(
                voxel_fields=validation_inputs[:1].to(device),
                representation="GRID",
            )
        ).predictions.cpu()
    torch.testing.assert_close(restored_first, predictions[:1])

    result_path = args.output_dir / "validation_result.npz"
    np.savez_compressed(
        result_path,
        inputs=validation_inputs.numpy(),
        targets=validation_targets.numpy(),
        predictions=predictions.numpy(),
        channel_names=np.asarray(CHANNEL_NAMES, dtype=np.str_),
        validation_mse=np.float64(final_mse),
        validation_mae=np.float64(final_mae),
        validation_relative_l2=np.float64(final_relative_l2),
    )

    print("\nTraining complete")
    print(f"  device: {device}")
    print(f"  training / validation grids: {len(train_inputs)} / {len(validation_inputs)}")
    print(f"  grid shape: {tuple(train_inputs.shape[-3:])} (D,H,W = Z,Y,X)")
    print(f"  validation MSE / MAE / relative L2: {final_mse:.6f} / {final_mae:.6f} / {final_relative_l2:.4f}")
    print(f"  canonical voxel grid: {canonical_path}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  validation results: {result_path}")


if __name__ == "__main__":
    main()
