"""Train Deep NURBS on a manufactured field over a geocore NURBS domain.

The target is constructed as the geocore admissible field times a smooth
physical-coordinate function, so it has an exact homogeneous Dirichlet trace.
This is a compact CPU training/serialization example, not a benchmark result.

Run from the workspace root:

    .venv/bin/python neural3dbench/examples/train_deep_nurbs_geocore_field.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
NEURAL3DBENCH_ROOT = WORKSPACE_ROOT / "neural3dbench"
GEOCORE_ROOT = WORKSPACE_ROOT / "geometry-py" / "geocore"
for source_root in (NEURAL3DBENCH_ROOT, GEOCORE_ROOT):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import geocore as gc  # noqa: E402
from geocore_deep_nurbs_integration import (  # noqa: E402
    adapt_parametric_surface,
    create_problem,
    metadata_from,
)
from models import ModelInput, ParametricSurfaceRepresentationAdapter, create_model  # noqa: E402


def manufactured_target(
    physical_coordinates: torch.Tensor, admissible_field: torch.Tensor
) -> torch.Tensor:
    """Smooth reference field with an exact zero boundary trace."""
    x = physical_coordinates[..., 0:1]
    y = physical_coordinates[..., 1:2]
    interior_field = 0.65 * torch.sin(2.2 * x) + 0.35 * torch.cos(1.7 * y) + 0.2 * x * y
    return admissible_field * interior_field


def model_input_from(raw: dict, device: torch.device) -> ModelInput:
    adapted = ParametricSurfaceRepresentationAdapter().adapt(
        {**raw, "metadata": metadata_from(raw)}
    )
    adapted.coordinates = adapted.coordinates.to(device=device, dtype=torch.float32)
    return adapted


def evaluate(model, model_input: ModelInput):
    model.eval()
    with torch.no_grad():
        output = model(model_input)
        target = manufactured_target(
            output.metadata["physical_coordinates"], output.metadata["admissible_field"]
        )
        prediction = output.predictions
        mse = F.mse_loss(prediction, target).item()
        mae = F.l1_loss(prediction, target).item()
        relative_l2 = (
            torch.linalg.vector_norm(prediction - target)
            / torch.linalg.vector_norm(target).clamp_min(1e-12)
        ).item()
        uv = output.metadata["parametric_coordinates"]
        boundary = ((uv == 0.0) | (uv == 1.0)).any(dim=-1)
        boundary_max_abs = prediction[boundary].abs().max().item()
    return mse, mae, relative_l2, boundary_max_abs, output, target


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--train-points", type=int, default=1024)
    parser.add_argument("--validation-resolution", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/deep_nurbs_geocore_field"),
    )
    args = parser.parse_args()
    if min(
        args.epochs,
        args.train_points,
        args.validation_resolution,
        args.batch_size,
        args.hidden_dim,
        args.layers,
    ) < 1:
        parser.error("epochs, sample sizes, batch size, width, and layers must be positive")
    if args.validation_resolution < 3:
        parser.error("validation-resolution must be at least 3")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    canonical = create_problem(
        n_points=args.train_points,
        seed=args.seed,
        method="sobol",
        include_boundary=0,
    )
    canonical_path = Path(
        gc.save(str(args.output_dir / "parametric_surface.npz"), canonical)
    )
    reloaded = gc.load(str(canonical_path))
    gc.validate(reloaded)
    config, train_input = adapt_parametric_surface(
        reloaded,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.layers,
        activation="tanh",
    )
    train_input.coordinates = train_input.coordinates.to(device)

    validation = create_problem(
        n_points=args.validation_resolution**2,
        seed=args.seed + 10_000,
        method="grid",
        include_boundary=0,
    )
    validation_input = model_input_from(validation, device)

    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed + 1)
    best_validation_mse = float("inf")
    best_state = None
    log_every = max(1, args.epochs // 10)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        order = torch.randperm(
            train_input.coordinates.shape[1], generator=generator
        )
        for indices in order.split(args.batch_size):
            batch = ModelInput(
                coordinates=train_input.coordinates[:, indices],
                representation="PARAMETRIC_SURFACE",
            )
            output = model(batch)
            target = manufactured_target(
                output.metadata["physical_coordinates"],
                output.metadata["admissible_field"],
            )
            loss = F.mse_loss(output.predictions, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        validation_metrics = evaluate(model, validation_input)
        validation_mse = validation_metrics[0]
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | train MSE {np.mean(losses):.7f} | "
                f"validation MSE {validation_mse:.7f} | "
                f"relative L2 {validation_metrics[2]:.5f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    mse, mae, relative_l2, boundary_max_abs, output, target = evaluate(
        model, validation_input
    )

    checkpoint_path = args.output_dir / "deep_nurbs_geocore_field.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": config.to_dict(),
            "normalization_id": metadata_from(reloaded)["normalization_id"],
            "target": "admissible * (0.65*sin(2.2*x) + 0.35*cos(1.7*y) + 0.2*x*y)",
            "best_validation_mse": best_validation_mse,
            "validation_mae": mae,
            "validation_relative_l2": relative_l2,
            "boundary_max_abs": boundary_max_abs,
        },
        checkpoint_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored = create_model(checkpoint["config"]).to(device).eval()
    restored.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        restored_prediction = restored(validation_input).predictions
    torch.testing.assert_close(restored_prediction, output.predictions)

    result_path = args.output_dir / "validation_result.npz"
    np.savez_compressed(
        result_path,
        parametric_coordinates=output.metadata["parametric_coordinates"].cpu().numpy(),
        physical_coordinates=output.metadata["physical_coordinates"].cpu().numpy(),
        targets=target.cpu().numpy(),
        predictions=output.predictions.cpu().numpy(),
        validation_mse=np.float64(mse),
        validation_mae=np.float64(mae),
        validation_relative_l2=np.float64(relative_l2),
        boundary_max_abs=np.float64(boundary_max_abs),
        validation_resolution=np.int64(args.validation_resolution),
    )

    print("\nTraining complete")
    print(f"  device: {device}")
    print(f"  training / validation points: {args.train_points} / {args.validation_resolution**2}")
    print(f"  validation MSE / MAE / relative L2: {mse:.7f} / {mae:.7f} / {relative_l2:.5f}")
    print(f"  boundary max |u|: {boundary_max_abs:.3e}")
    print(f"  canonical parametric surface: {canonical_path}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  validation results: {result_path}")


if __name__ == "__main__":
    main()
