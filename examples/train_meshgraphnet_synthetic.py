from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import MeshGraphNetConfig, ModelInput, ModelOutput, create_model


@dataclass
class DiffusionGraph:
    model_input: ModelInput
    target: torch.Tensor


def make_graph(
    node_count: int, k: int, diffusion: float, generator: torch.Generator
) -> DiffusionGraph:
    coordinates = torch.rand((node_count, 2), generator=generator)
    distances = torch.cdist(coordinates, coordinates)
    distances.fill_diagonal_(float("inf"))
    neighbors = distances.topk(min(k, node_count - 1), largest=False).indices
    sources = torch.arange(node_count).unsqueeze(1).expand_as(neighbors).reshape(-1)
    edge_index = torch.stack((sources, neighbors.reshape(-1)))
    edge_index = torch.unique(torch.cat((edge_index, edge_index.flip(0)), dim=1), dim=1)

    relative = coordinates[edge_index[1]] - coordinates[edge_index[0]]
    distance = relative.norm(dim=-1, keepdim=True)
    edge_features = torch.cat((relative, distance), dim=-1)

    state = torch.randn((node_count, 1), generator=generator)
    weights = torch.exp(-8.0 * distance.square())
    weighted_difference = weights * (state[edge_index[0]] - state[edge_index[1]])
    diffusion_sum = torch.zeros_like(state)
    weight_sum = torch.zeros_like(state)
    diffusion_sum.index_add_(0, edge_index[1], weighted_difference)
    weight_sum.index_add_(0, edge_index[1], weights)
    target = state + diffusion * diffusion_sum / weight_sum.clamp_min(1e-8)

    node_features = torch.cat((state, coordinates), dim=-1)
    return DiffusionGraph(
        ModelInput(
            coordinates=coordinates,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            batch=torch.zeros(node_count, dtype=torch.long),
            representation="GRAPH",
        ),
        target,
    )


def make_dataset(count, min_nodes, max_nodes, k, diffusion, seed):
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.randint(min_nodes, max_nodes + 1, (count,), generator=generator)
    return [make_graph(int(size), k, diffusion, generator) for size in sizes]


def move_graph(graph: DiffusionGraph, device: torch.device):
    source = graph.model_input
    return (
        ModelInput(
            coordinates=source.coordinates.to(device),
            node_features=source.node_features.to(device),
            edge_index=source.edge_index.to(device),
            edge_features=source.edge_features.to(device),
            batch=source.batch.to(device),
            representation="GRAPH",
        ),
        graph.target.to(device),
    )


def evaluate(model, dataset, device):
    model.eval()
    with torch.no_grad():
        losses = []
        for graph in dataset:
            inputs, target = move_graph(graph, device)
            losses.append(nn.functional.mse_loss(model(inputs).predictions, target).item())
    return sum(losses) / len(losses)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MeshGraphNet on one-step graph diffusion."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-graphs", type=int, default=24)
    parser.add_argument("--val-graphs", type=int, default=6)
    parser.add_argument("--min-nodes", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=40)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--diffusion", type=float, default=0.2)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--processor-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/meshgraphnet_synthetic.pt")
    )
    args = parser.parse_args()
    if min(args.epochs, args.train_graphs, args.val_graphs, args.k) < 1:
        parser.error("epochs, graph counts, and k must be positive")
    if args.min_nodes < 2 or args.max_nodes < args.min_nodes:
        parser.error("require 2 <= min-nodes <= max-nodes")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    training = make_dataset(
        args.train_graphs, args.min_nodes, args.max_nodes, args.k,
        args.diffusion, args.seed + 1
    )
    validation = make_dataset(
        args.val_graphs, args.min_nodes, args.max_nodes, args.k,
        args.diffusion, args.seed + 2
    )
    config = MeshGraphNetConfig(
        node_input_dim=3,
        edge_input_dim=3,
        output_dim=1,
        latent_dim=args.latent_dim,
        processor_steps=args.processor_steps,
        mlp_hidden_dim=args.latent_dim,
        mlp_layers=2,
    )
    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(training)
        train_losses = []
        for graph in training:
            inputs, target = move_graph(graph, device)
            output: ModelOutput = model(inputs)
            loss = nn.functional.mse_loss(output.predictions, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        validation_loss = evaluate(model, validation, device)
        train_loss = sum(train_losses) / len(train_losses)
        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | "
            f"Val Loss: {validation_loss:.6f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "config": config.to_dict(),
                    "epoch": epoch,
                    "val_loss": best_loss,
                },
                args.checkpoint,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best validation loss: {evaluate(model, validation, device):.6f}")
    print(f"Best checkpoint: {args.checkpoint}")
    inputs, target = move_graph(validation[0], device)
    with torch.no_grad():
        prediction = model(inputs).predictions
    print("Node | Predicted | Target")
    for node in range(min(5, target.shape[0])):
        print(f"{node:4d} | {prediction[node, 0].item():9.5f} | {target[node, 0].item():9.5f}")


if __name__ == "__main__":
    main()
