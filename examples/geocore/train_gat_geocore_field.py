"""Train GAT on a synthetic scalar field over a geocore mesh.

Run from the workspace root with:

    .venv/bin/python neural3dbench/examples/geocore/train_gat_geocore_field.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

import geocore as gc
from geocore_gat_integration import adapt_graph, make_mesh_graph
from models import GATConfig, create_model


def manufactured_field(coordinates: np.ndarray) -> np.ndarray:
    """Return a smooth scalar field with spatial variation in both axes."""
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    return (
        0.5
        + 0.25 * np.sin(np.pi * x) * np.cos(np.pi * y)
        + 0.15 * x * y
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--nx", type=int, default=10)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gat_geocore_field"),
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.nx < 2 or args.ny < 2:
        parser.error("epochs must be positive and nx/ny must be at least 2")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = make_mesh_graph(args.nx, args.ny)
    graph_path = Path(gc.save(str(output_dir / "gat_mesh_graph.npz"), graph))
    raw = gc.load(str(graph_path))
    model_input = adapt_graph(raw)
    model_input.node_features = model_input.node_features.to(device)
    model_input.edge_index = model_input.edge_index.to(device)
    target = torch.from_numpy(manufactured_field(raw["coordinates"])[:, None]).to(device)

    config = GATConfig(
        input_dim=model_input.node_features.shape[1],
        output_dim=1,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
    )
    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial_loss = float("nan")
    for epoch in range(1, args.epochs + 1):
        model.train()
        prediction = model(model_input).predictions
        loss = nn.functional.mse_loss(prediction, target)
        if epoch == 1:
            initial_loss = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        prediction = model(model_input).predictions[:, 0].cpu().numpy()
    target_values = target[:, 0].cpu().numpy()
    final_loss = float(np.mean((prediction - target_values) ** 2))

    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict()},
        output_dir / "gat_geocore_field.pt",
    )
    np.savez_compressed(
        output_dir / "gat_field_result.npz",
        coordinates=raw["coordinates"],
        target=target_values,
        prediction=prediction,
        absolute_error=np.abs(prediction - target_values),
    )
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs} | initial MSE: {initial_loss:.6f} | final MSE: {final_loss:.6f}")
    print(f"Saved graph: {graph_path}")
    print(f"Saved checkpoint: {output_dir / 'gat_geocore_field.pt'}")
    print(f"Saved field result: {output_dir / 'gat_field_result.npz'}")


if __name__ == "__main__":
    main()