"""Train MeshNet, DiffusionNet or MeshWalker on triangle meshes (no point sampling).

Datasets: modelnet10, modelnet40 (Princeton) and shrec11 (SHREC'11, 600 meshes, 3.2 MB download).
Uses the official train/test split. Meshes are decimated to at most ``--max-faces`` triangles and
cached under ``$NEURAL3DBENCH_DATA/meshes`` (default ``~/.cache/neural3dbench``); the first run
reads the archive and takes several minutes. DiffusionNet operators are cached next to the meshes.

    python examples/train_mesh_classifier.py --model meshnet --epochs 30
    python examples/train_mesh_classifier.py --model diffusionnet --dataset shrec11 --epochs 50
    python examples/train_mesh_classifier.py --model diffusionnet --k-eig 64
    python examples/train_mesh_classifier.py --model meshwalker --config '{"gru_dims": [256, 256]}'
    python examples/train_mesh_classifier.py --model meshnet --max-per-class 20 --epochs 2  # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import collate_meshes, create_model
from models.diffusionnet import batch_operators, compute_operators
from models.diffusionnet.model import OPERATOR_KEY
from tools.dashboard import tasks
from tools.dashboard.data_sources import MESH_DATASETS, load_mesh_dataset


def progress(fraction: float, message: str) -> None:
    print(f"\r{message}", end="\n" if fraction >= 1 else "", flush=True)


def diffusion_operators(meshes, k_eig: int, cache: Path) -> list[dict[str, np.ndarray]]:
    """Per-mesh DiffusionNet operators, cached with ``torch.save``."""
    if cache.is_file():
        return torch.load(cache, weights_only=False)
    operators = []
    for i, (vertices, faces) in enumerate(meshes):
        operators.append(compute_operators(vertices, faces, k_eig))
        progress((i + 1) / len(meshes), f"DiffusionNet operators: {i + 1} / {len(meshes)} meshes")
    torch.save(operators, cache)
    return operators


def make_input(meshes, operators, index):
    inputs = collate_meshes([meshes[i] for i in index])
    if operators is not None:
        inputs.metadata[OPERATOR_KEY] = batch_operators([operators[i] for i in index])
    return inputs


@torch.no_grad()
def evaluate(model, meshes, labels, operators, batch_size: int) -> float:
    model.eval()
    correct = 0
    for start in range(0, len(meshes), batch_size):
        index = range(start, min(start + batch_size, len(meshes)))
        logits = model(make_input(meshes, operators, index)).predictions
        correct += (logits.argmax(-1).cpu() == torch.as_tensor(labels[list(index)])).sum().item()
    return correct / len(meshes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("meshnet", "diffusionnet", "meshwalker"), required=True)
    parser.add_argument("--dataset", choices=sorted(MESH_DATASETS), default="modelnet10")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-faces", type=int, default=2048)
    parser.add_argument("--max-per-class", type=int, default=None, help="cap meshes per class and split")
    parser.add_argument("--k-eig", type=int, default=128, help="DiffusionNet eigenvectors")
    parser.add_argument("--config", default="{}", help="JSON overrides for the model config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    split = {}
    for name in ("train", "test"):
        split[name] = load_mesh_dataset(args.dataset, name, args.max_faces, args.max_per_class, args.seed, progress)
    train_meshes, train_labels, class_names = split["train"]
    test_meshes, test_labels, _ = split["test"]
    print(f"{args.dataset}: {len(train_meshes)} train / {len(test_meshes)} test meshes, {len(class_names)} classes, "
          f"median {int(np.median([len(f) for _, f in train_meshes]))} faces")

    config = {"name": args.model, "output_dim": len(class_names), **json.loads(args.config)}
    train_ops = test_ops = None
    if args.model == "diffusionnet":
        config.setdefault("k_eig", args.k_eig)
        stem = f"{args.dataset}_f{args.max_faces}_c{args.max_per_class or 'all'}_s{args.seed}_k{config['k_eig']}"
        train_ops = diffusion_operators(train_meshes, config["k_eig"], tasks.DATA_DIR / "meshes" / f"{stem}_train_ops.pt")
        test_ops = diffusion_operators(test_meshes, config["k_eig"], tasks.DATA_DIR / "meshes" / f"{stem}_test_ops.pt")
    model = create_model(config).to(args.device)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} parameters on {args.device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    checkpoint = args.checkpoint or Path("checkpoints") / f"{args.model}_{args.dataset}.pt"

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total, seen = time.time(), 0.0, 0
        order = torch.randperm(len(train_meshes)).tolist()
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            if len(index) == 1:
                continue  # BatchNorm cannot train on one mesh
            target = torch.as_tensor(train_labels[index], device=args.device)
            loss = F.cross_entropy(model(make_input(train_meshes, train_ops, index)).predictions, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, seen = total + loss.item() * len(index), seen + len(index)
        scheduler.step()
        accuracy = evaluate(model, test_meshes, test_labels, test_ops, args.batch_size)
        print(f"epoch {epoch:3d}  train loss {total / seen:.4f}  test accuracy {accuracy:.4f}  ({time.time() - started:.0f}s)")
        if accuracy > best:
            best = accuracy
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_name": args.model, "config": model.config.to_dict(), "class_names": class_names,
                        "test_accuracy": accuracy, "state_dict": model.state_dict()}, checkpoint)
    print(f"best test accuracy {best:.4f}; weights in {checkpoint}")


if __name__ == "__main__":
    main()
