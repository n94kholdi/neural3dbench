"""Train MeshGraphNet to solve heat diffusion on a geocore mesh.

The example follows the complete integration path:

1. create a triangulated plate with geocore;
2. define hot/cold Dirichlet boundaries and a discrete heat equation;
3. serialize one canonical ``MESH_GRAPH`` problem as ``mesh_graph.npz``;
4. batch geocore graphs and adapt them to neural3dbench ``ModelInput``;
5. train MeshGraphNet to predict one heat-diffusion step;
6. roll the learned solver forward and compare it with the reference solver.

Run from the workspace root:

    .venv/bin/python neural3dbench/examples/train_meshgraphnet_geocore_heat.py

This is a small educational problem, not a replacement for a validated FEM or
CFD heat solver.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
NEURAL3DBENCH_ROOT = WORKSPACE_ROOT / "neural3dbench"
GEOCORE_ROOT = WORKSPACE_ROOT / "geometry-py" / "geocore"
for source_root in (NEURAL3DBENCH_ROOT, GEOCORE_ROOT):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import geocore as gc  # noqa: E402
from models import (  # noqa: E402
    GraphRepresentationAdapter,
    MeshGraphNetConfig,
    ModelInput,
    create_model,
)


FEATURE_NAMES = np.array(
    ["temperature", "x", "y", "z", "hot_boundary", "cold_boundary"],
    dtype=np.str_,
)


@dataclass
class HeatBatch:
    model_input: ModelInput
    target: torch.Tensor


def make_plate_mesh(nx: int, ny: int, length: float = 2.0, width: float = 1.0) -> gc.Mesh:
    """Create an ``nx`` by ``ny`` triangular plate in the XY plane."""
    x = np.linspace(0.0, length, nx)
    y = np.linspace(0.0, width, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    vertices = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(nx * ny)))

    faces = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            lower_left = row * nx + column
            lower_right = lower_left + 1
            upper_left = lower_left + nx
            upper_right = upper_left + 1
            faces.append((lower_left, lower_right, upper_right))
            faces.append((lower_left, upper_right, upper_left))
    return gc.Mesh(vertices, np.asarray(faces, dtype=np.int64))


def create_problem(nx: int, ny: int) -> dict:
    """Use geocore to construct the normalized mesh graph and boundary masks."""
    plate = make_plate_mesh(nx, ny)
    graph = dict(
        plate.output(
            "MESH_GRAPH",
            normalization="unit_cube",
        )
    )

    coordinates = graph["coordinates"]
    x = coordinates[:, 0]
    hot = np.isclose(x, x.min()).astype(np.float32)
    cold = np.isclose(x, x.max()).astype(np.float32)

    # The temperature column is replaced for every training sample. Coordinates
    # and boundary masks are static problem data supplied to the network.
    temperature = np.zeros((len(coordinates), 1), dtype=np.float32)
    graph["node_features"] = np.column_stack(
        (temperature, coordinates, hot, cold)
    ).astype(np.float32)
    graph["node_feature_names"] = FEATURE_NAMES.copy()
    graph.pop("faces", None)
    gc.validate(graph)
    return graph


def boundary_masks(graph: dict) -> tuple[np.ndarray, np.ndarray]:
    features = graph["node_features"]
    return features[:, 4].astype(bool), features[:, 5].astype(bool)


def heat_step(graph: dict, temperature: np.ndarray, diffusivity: float) -> np.ndarray:
    """Advance the reference graph heat equation by one explicit time step."""
    edge_index = graph["edge_index"]
    sender, receiver = edge_index
    distance = np.maximum(graph["edge_features"][:, 3], 1e-6)
    weights = 1.0 / distance

    neighbor_delta = weights * (temperature[sender] - temperature[receiver])
    delta_sum = np.zeros_like(temperature)
    weight_sum = np.zeros_like(temperature)
    np.add.at(delta_sum, receiver, neighbor_delta)
    np.add.at(weight_sum, receiver, weights)
    next_temperature = temperature + diffusivity * delta_sum / np.maximum(weight_sum, 1e-8)

    hot, cold = boundary_masks(graph)
    next_temperature[hot] = 1.0
    next_temperature[cold] = 0.0
    return next_temperature.astype(np.float32)


def random_problem_state(graph: dict, rng: np.random.Generator, diffusivity: float):
    sample = {key: np.asarray(value).copy() for key, value in graph.items()}
    temperature = rng.uniform(0.0, 1.0, len(graph["coordinates"])).astype(np.float32)
    hot, cold = boundary_masks(graph)
    temperature[hot] = 1.0
    temperature[cold] = 0.0
    sample["node_features"][:, 0] = temperature
    target = heat_step(sample, temperature, diffusivity)
    return sample, target[:, None]


def metadata_from(graph: dict) -> dict:
    keys = (
        "schema_version",
        "representation",
        "coordinate_system",
        "units",
        "normalization_id",
        "node_feature_names",
        "edge_feature_names",
    )
    metadata = {}
    for key in keys:
        if key not in graph:
            continue
        value = np.asarray(graph[key])
        metadata[key] = value.item() if value.ndim == 0 else value.tolist()
    return metadata


def adapt_graph(graph: dict) -> ModelInput:
    """Convert a canonical geocore MESH_GRAPH to neural3dbench ModelInput."""
    return GraphRepresentationAdapter().adapt(
        {**graph, "metadata": metadata_from(graph)},
        batch=graph.get("batch"),
    )


def make_batch(
    problem: dict,
    sample_count: int,
    diffusivity: float,
    seed: int,
) -> HeatBatch:
    rng = np.random.default_rng(seed)
    samples, targets = zip(
        *(random_problem_state(problem, rng, diffusivity) for _ in range(sample_count))
    )
    batched_graph = gc.batch_graphs(samples)
    gc.validate(batched_graph)
    return HeatBatch(
        model_input=adapt_graph(batched_graph),
        target=torch.from_numpy(np.concatenate(targets, axis=0)),
    )


def move_batch(batch: HeatBatch, device: torch.device) -> HeatBatch:
    source = batch.model_input
    return HeatBatch(
        model_input=ModelInput(
            coordinates=source.coordinates.to(device),
            node_features=source.node_features.to(device),
            edge_index=source.edge_index.to(device),
            edge_features=source.edge_features.to(device),
            batch=source.batch.to(device),
            metadata=source.metadata,
            representation="GRAPH",
        ),
        target=batch.target.to(device),
    )


def evaluate(model, batch: HeatBatch) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        prediction = model(batch.model_input).predictions
        mse = nn.functional.mse_loss(prediction, batch.target).item()
        mae = nn.functional.l1_loss(prediction, batch.target).item()
    return mse, mae


def rollout(
    model,
    sample: dict,
    diffusivity: float,
    steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out learned and reference solvers from the same initial state."""
    initial = sample["node_features"][:, 0].copy()
    learned = initial.copy()
    reference = initial.copy()
    hot, cold = boundary_masks(sample)

    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            reference = heat_step(sample, reference, diffusivity)

            model_graph = {key: np.asarray(value).copy() for key, value in sample.items()}
            model_graph["node_features"][:, 0] = learned
            inputs = adapt_graph(model_graph)
            inputs = ModelInput(
                coordinates=inputs.coordinates.to(device),
                node_features=inputs.node_features.to(device),
                edge_index=inputs.edge_index.to(device),
                edge_features=inputs.edge_features.to(device),
                batch=inputs.batch.to(device),
                metadata=inputs.metadata,
                representation="GRAPH",
            )
            learned = model(inputs).predictions[:, 0].cpu().numpy()
            # Dirichlet values are part of the physical time integrator and are
            # imposed exactly after every learned update.
            learned[hot] = 1.0
            learned[cold] = 0.0

    return initial, reference, learned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--diffusivity", type=float, default=0.35)
    parser.add_argument("--rollout-steps", type=int, default=12)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--processor-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/meshgraphnet_geocore_heat"),
    )
    args = parser.parse_args()
    if min(
        args.epochs,
        args.train_samples,
        args.validation_samples,
        args.rollout_steps,
    ) < 1:
        parser.error("epochs, sample counts, and rollout steps must be positive")
    if args.nx < 3 or args.ny < 3:
        parser.error("nx and ny must each be at least 3")
    if not 0.0 < args.diffusivity <= 1.0:
        parser.error("diffusivity must be in (0, 1]")
    return args


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    problem = create_problem(args.nx, args.ny)
    problem_path = Path(gc.save(str(args.output_dir / "mesh_graph.npz"), problem))

    training = move_batch(
        make_batch(problem, args.train_samples, args.diffusivity, args.seed + 1),
        device,
    )
    validation = move_batch(
        make_batch(problem, args.validation_samples, args.diffusivity, args.seed + 2),
        device,
    )

    config = MeshGraphNetConfig(
        node_input_dim=training.model_input.node_features.shape[1],
        edge_input_dim=training.model_input.edge_features.shape[1],
        output_dim=1,
        latent_dim=args.latent_dim,
        processor_steps=args.processor_steps,
        mlp_hidden_dim=args.latent_dim,
        mlp_layers=2,
    )
    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_validation_mse = float("inf")
    best_state = None
    log_every = max(1, args.epochs // 10)
    for epoch in range(1, args.epochs + 1):
        model.train()
        prediction = model(training.model_input).predictions
        loss = nn.functional.mse_loss(prediction, training.target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        validation_mse, validation_mae = evaluate(model, validation)
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:04d} | train MSE {loss.item():.6f} | "
                f"validation MSE {validation_mse:.6f} | MAE {validation_mae:.6f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)

    rng = np.random.default_rng(args.seed + 3)
    rollout_sample, _ = random_problem_state(problem, rng, args.diffusivity)
    initial, reference, learned = rollout(
        model,
        rollout_sample,
        args.diffusivity,
        args.rollout_steps,
        device,
    )
    rollout_mae = float(np.mean(np.abs(learned - reference)))

    checkpoint_path = args.output_dir / "meshgraphnet_heat.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": config.to_dict(),
            "problem": {
                "diffusivity": args.diffusivity,
                "nx": args.nx,
                "ny": args.ny,
                "node_feature_names": FEATURE_NAMES.tolist(),
            },
            "best_validation_mse": best_validation_mse,
        },
        checkpoint_path,
    )
    result_path = args.output_dir / "rollout_result.npz"
    np.savez_compressed(
        result_path,
        coordinates=problem["coordinates"],
        initial_temperature=initial,
        reference_temperature=reference,
        predicted_temperature=learned,
    )

    final_mse, final_mae = evaluate(model, validation)
    print("\nTraining complete")
    print(f"  device: {device}")
    print(f"  mesh: {len(problem['coordinates'])} nodes, {problem['edge_index'].shape[1]} directed edges")
    print(f"  one-step validation MSE / MAE: {final_mse:.6f} / {final_mae:.6f}")
    print(f"  {args.rollout_steps}-step rollout MAE: {rollout_mae:.6f}")
    print(f"  problem graph: {problem_path}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  rollout values: {result_path}")


if __name__ == "__main__":
    main()
