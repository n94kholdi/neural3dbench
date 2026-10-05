from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import GATConfig, GCNConfig, ModelInput, ModelOutput, ModelRegistry


@dataclass
class SyntheticGraph:
    model_input: ModelInput
    target: torch.Tensor


def make_graph(node_count: int, k: int, generator: torch.Generator) -> SyntheticGraph:
    coordinates = torch.rand((node_count, 3), generator=generator) * 2.0 - 1.0
    distances = torch.cdist(coordinates, coordinates)
    distances.fill_diagonal_(float("inf"))
    neighbors = distances.topk(min(k, node_count - 1), largest=False).indices

    sources = torch.arange(node_count).unsqueeze(1).expand_as(neighbors).reshape(-1)
    targets = neighbors.reshape(-1)
    edge_index = torch.stack((sources, targets), dim=0)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    target = (
        torch.sin(coordinates[:, 0:1])
        + torch.cos(coordinates[:, 1:2])
        + coordinates[:, 2:3].square()
    )
    return SyntheticGraph(
        model_input=ModelInput(
            coordinates=coordinates,
            node_features=coordinates,
            edge_index=edge_index,
            batch=torch.zeros(node_count, dtype=torch.long),
        ),
        target=target,
    )


def make_dataset(
    graph_count: int,
    min_nodes: int,
    max_nodes: int,
    k: int,
    seed: int,
) -> list[SyntheticGraph]:
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.randint(min_nodes, max_nodes + 1, (graph_count,), generator=generator)
    return [make_graph(int(size), k, generator) for size in sizes]


def move_graph(graph: SyntheticGraph, device: torch.device) -> tuple[ModelInput, torch.Tensor]:
    model_input = ModelInput(
        coordinates=graph.model_input.coordinates.to(device),
        node_features=graph.model_input.node_features.to(device),
        edge_index=graph.model_input.edge_index.to(device),
        batch=graph.model_input.batch.to(device),
    )
    return model_input, graph.target.to(device)


def supervised_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Keep supervised loss isolated so physics losses can be added later."""
    return nn.functional.mse_loss(prediction, target)


def evaluate(
    model: torch.nn.Module,
    dataset: list[SyntheticGraph],
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for graph in dataset:
            model_input, target = move_graph(graph, device)
            output: ModelOutput = model(model_input)
            losses.append(supervised_loss(output.predictions, target).item())
    return sum(losses) / len(losses)


def parse_args(default_model: str = "gcn") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GCN or GAT on synthetic 3D graphs.")
    parser.add_argument("--model", choices=("gcn", "gat"), default=default_model)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-graphs", type=int, default=80)
    parser.add_argument("--val-graphs", type=int, default=20)
    parser.add_argument("--min-nodes", type=int, default=50)
    parser.add_argument("--max-nodes", type=int, default=150)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4, help="GAT attention heads")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.train_graphs < 1 or args.val_graphs < 1:
        parser.error("epochs, train-graphs, and val-graphs must be positive")
    if args.min_nodes < 2 or args.max_nodes < args.min_nodes:
        parser.error("require 2 <= min-nodes <= max-nodes")
    if (
        args.k < 1
        or args.hidden_dim < 1
        or args.num_layers < 1
        or args.num_heads < 1
        or args.learning_rate <= 0
    ):
        parser.error("k, hidden-dim, num-layers, num-heads, and learning-rate must be positive")
    if args.model == "gat" and args.hidden_dim % args.num_heads != 0:
        parser.error("hidden-dim must be divisible by num-heads for GAT")
    if args.checkpoint is None:
        args.checkpoint = Path(f"checkpoints/{args.model}_synthetic.pt")
    return args


def main(default_model: str = "gcn") -> None:
    args = parse_args(default_model=default_model)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    training_data = make_dataset(
        args.train_graphs, args.min_nodes, args.max_nodes, args.k, args.seed + 1
    )
    validation_data = make_dataset(
        args.val_graphs, args.min_nodes, args.max_nodes, args.k, args.seed + 2
    )

    common_config = {
        "input_dim": 3,
        "hidden_dim": args.hidden_dim,
        "output_dim": 1,
        "num_layers": args.num_layers,
    }
    if args.model == "gat":
        config = GATConfig(
            **common_config,
            num_heads=args.num_heads,
            activation="elu",
        )
    else:
        config = GCNConfig(**common_config)
    model = ModelRegistry.create(args.model, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(training_data)
        train_losses = []
        for graph in training_data:
            model_input, target = move_graph(graph, device)
            output: ModelOutput = model(model_input)
            loss = supervised_loss(output.predictions, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = evaluate(model, validation_data, device)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "config": config.to_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                args.checkpoint,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_loss = evaluate(model, validation_data, device)
    print(f"Best validation loss: {final_val_loss:.4f}")
    print(f"Best checkpoint: {args.checkpoint}")

    sample = validation_data[0]
    model_input, target = move_graph(sample, device)
    model.eval()
    with torch.no_grad():
        output: ModelOutput = model(model_input)
        prediction = output.predictions
    print("Node | Predicted | True")
    for node in range(min(5, target.shape[0])):
        print(f"{node:4d} | {prediction[node, 0].item():9.4f} | {target[node, 0].item():.4f}")


if __name__ == "__main__":
    main()
