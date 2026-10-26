"""Train GINO to interpolate analytic fields on geocore-sampled solids.

Each sample contains source field values and separate query locations. This is
a compact integration problem, not a validated physical solver.

Run from the workspace root:

    .venv/bin/python neural3dbench/examples/geocore/train_gino_geocore_field.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
NEURAL3DBENCH_ROOT = WORKSPACE_ROOT / "neural3dbench"
GEOCORE_ROOT = WORKSPACE_ROOT / "geometry-py" / "geocore"
for source_root in (NEURAL3DBENCH_ROOT, GEOCORE_ROOT):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import geocore as gc  # noqa: E402
from geocore_gino_integration import create_sample, metadata_from  # noqa: E402
from models import GINOConfig, ModelInput, create_model  # noqa: E402


def make_dataset(samples: int, source_count: int, query_count: int, seed: int):
    clouds, targets = [], []
    for index in range(samples):
        cloud, target = create_sample(
            source_count=source_count,
            query_count=query_count,
            seed=seed + index,
        )
        clouds.append(cloud)
        targets.append(target[0])
    return {
        "coordinates": torch.from_numpy(
            np.concatenate([cloud["coordinates"] for cloud in clouds], axis=0)
        ),
        "point_features": torch.from_numpy(
            np.concatenate([cloud["point_features"] for cloud in clouds], axis=0)
        ),
        "query_coordinates": torch.from_numpy(
            np.concatenate([cloud["query_coordinates"] for cloud in clouds], axis=0)
        ),
        "targets": torch.from_numpy(np.stack(targets)),
    }


def model_input(dataset: dict[str, torch.Tensor], indices, device: torch.device):
    return ModelInput(
        coordinates=dataset["coordinates"][indices].to(device),
        point_features=dataset["point_features"][indices].to(device),
        query_coordinates=dataset["query_coordinates"][indices].to(device),
        representation="POINT_CLOUD",
    )


def evaluate(model, dataset: dict[str, torch.Tensor], device: torch.device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for indices in torch.arange(len(dataset["targets"])).split(4):
            predictions.append(model(model_input(dataset, indices, device)).predictions.cpu())
    prediction = torch.cat(predictions)
    target = dataset["targets"]
    mse = nn.functional.mse_loss(prediction, target).item()
    mae = nn.functional.l1_loss(prediction, target).item()
    return mse, mae, prediction


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-samples", type=int, default=12)
    parser.add_argument("--validation-samples", type=int, default=3)
    parser.add_argument("--source-points", type=int, default=48)
    parser.add_argument("--query-points", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--latent-resolution", type=int, default=6)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--modes", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/gino_geocore_field")
    )
    args = parser.parse_args()
    if min(
        args.epochs,
        args.train_samples,
        args.validation_samples,
        args.source_points,
        args.query_points,
        args.batch_size,
        args.latent_resolution,
        args.width,
        args.layers,
        args.modes,
    ) < 1:
        parser.error("counts, dimensions, epochs, and modes must be positive")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    canonical, _ = create_sample(
        source_count=args.source_points,
        query_count=args.query_points,
        seed=args.seed,
    )
    cloud_path = Path(gc.save(str(args.output_dir / "point_cloud.npz"), canonical))
    reloaded = gc.load(str(cloud_path))
    gc.validate(reloaded)

    train_data = make_dataset(
        args.train_samples, args.source_points, args.query_points, args.seed + 100
    )
    validation_data = make_dataset(
        args.validation_samples,
        args.source_points,
        args.query_points,
        args.seed + 10_000,
    )
    config = GINOConfig(
        input_dim=1,
        output_dim=1,
        latent_resolution=(args.latent_resolution,) * 3,
        width=args.width,
        modes=(args.modes,) * 3,
        num_layers=args.layers,
        kernel_radius=0.6,
        kernel_hidden_dim=max(8, args.width * 2),
        kernel_chunk_size=128,
        kernel_max_pairs=8192,
    )
    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed + 1)
    best_mse = float("inf")
    best_state = None
    log_every = max(1, args.epochs // 6)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        order = torch.randperm(len(train_data["targets"]), generator=generator)
        for indices in order.split(args.batch_size):
            inputs = model_input(train_data, indices, device)
            target = train_data["targets"][indices].to(device)
            prediction = model(inputs).predictions
            loss = nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        validation_mse, validation_mae, _ = evaluate(model, validation_data, device)
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | train MSE {np.mean(losses):.6f} | "
                f"validation MSE {validation_mse:.6f} | MAE {validation_mae:.6f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    final_mse, final_mae, predictions = evaluate(model, validation_data, device)
    checkpoint_path = args.output_dir / "gino_geocore_field.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": config.to_dict(),
            "normalization_id": metadata_from(reloaded)["normalization_id"],
            "best_validation_mse": best_mse,
            "validation_mae": final_mae,
        },
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored = create_model(checkpoint["config"]).to(device).eval()
    restored.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        restored_first = restored(
            model_input(validation_data, torch.tensor([0]), device)
        ).predictions.cpu()
    torch.testing.assert_close(restored_first, predictions[:1])

    result_path = args.output_dir / "validation_result.npz"
    np.savez_compressed(
        result_path,
        source_coordinates=validation_data["coordinates"].numpy(),
        source_values=validation_data["point_features"].numpy(),
        query_coordinates=validation_data["query_coordinates"].numpy(),
        targets=validation_data["targets"].numpy(),
        predictions=predictions.numpy(),
        validation_mse=np.float64(final_mse),
        validation_mae=np.float64(final_mae),
    )

    print("\nTraining complete")
    print(f"  device: {device}")
    print(f"  source/query points: {args.source_points} / {args.query_points}")
    print(f"  training/validation samples: {args.train_samples} / {args.validation_samples}")
    print(f"  validation MSE / MAE: {final_mse:.6f} / {final_mae:.6f}")
    print(f"  canonical point cloud: {cloud_path}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  validation results: {result_path}")


if __name__ == "__main__":
    main()