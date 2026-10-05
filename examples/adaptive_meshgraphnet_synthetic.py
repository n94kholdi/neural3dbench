from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh import MeshAdaptationConfig, MeshMode, TriangleMeshGraphBuilder, TriangularMesh
from models import MeshGraphNetConfig, create_model
from simulation import MeshSimulationRunner


def make_grid(resolution: int = 5) -> TriangularMesh:
    axis = torch.linspace(0.0, 1.0, resolution)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    vertices = torch.stack((x.flatten(), y.flatten()), dim=1)
    triangles = []
    for row in range(resolution - 1):
        for column in range(resolution - 1):
            lower_left = row * resolution + column
            lower_right = lower_left + 1
            upper_left = lower_left + resolution
            upper_right = upper_left + 1
            triangles.extend(
                ([lower_left, lower_right, upper_right], [lower_left, upper_right, upper_left])
            )
    bottom = [[index, index + 1] for index in range(resolution - 1)]
    right = [
        [(index + 1) * resolution - 1, (index + 2) * resolution - 1]
        for index in range(resolution - 1)
    ]
    top = [
        [(resolution - 1) * resolution + index + 1, (resolution - 1) * resolution + index]
        for index in range(resolution - 1)
    ]
    left = [[(index + 1) * resolution, index * resolution] for index in range(resolution - 1)]
    boundary_edges = torch.tensor(bottom + right + top + left)
    boundary_tags = torch.cat(
        [torch.full((resolution - 1,), tag) for tag in (1, 2, 3, 4)]
    )
    return TriangularMesh(
        vertices=vertices,
        triangles=torch.tensor(triangles),
        boundary_edges=boundary_edges,
        boundary_tags=boundary_tags,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one MeshGraphNet in fixed or deterministic adaptive-mesh mode."
    )
    parser.add_argument("--mesh-mode", choices=[mode.value for mode in MeshMode], default="fixed")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--adapt-every", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--max-nodes", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    mesh = make_grid()
    center = torch.tensor([0.72, 0.68])
    state = {
        "concentration": torch.exp(
            -80.0 * ((mesh.vertices - center) ** 2).sum(dim=1, keepdim=True)
        )
    }
    model = create_model(
        MeshGraphNetConfig(
            node_input_dim=3,
            edge_input_dim=3,
            output_dim=1,
            latent_dim=16,
            processor_steps=2,
            mlp_hidden_dim=16,
            mlp_layers=1,
        )
    )
    adaptation = MeshAdaptationConfig(
        mode=args.mesh_mode,
        adapt_every=args.adapt_every,
        criterion_field="concentration",
        refinement_threshold=args.threshold,
        max_nodes=args.max_nodes,
        max_elements=200,
        max_refinement_level=2,
    )
    runner = MeshSimulationRunner(
        model,
        TriangleMeshGraphBuilder(["concentration"], include_coordinates=True),
        mesh_adaptation=adaptation,
        # This demonstration isolates remeshing; a real task applies predictions here.
        state_updater=lambda mesh, state, output: dict(state),
    )
    result = runner.run(mesh, state, steps=args.steps)
    counts = [mesh.num_nodes] + [frame.mesh_after_step.num_nodes for frame in result.frames]
    refined = result.final_mesh.vertices[mesh.num_nodes :]
    near_peak = (
        torch.linalg.vector_norm(refined - center, dim=1) < 0.35
    ).sum().item() if refined.numel() else 0
    print(f"mode={args.mesh_mode} node_counts={counts}")
    print(f"new_nodes_near_high_gradient_region={near_peak}/{refined.shape[0]}")
    print(f"final_cells={result.final_mesh.num_cells}")


if __name__ == "__main__":
    main()
