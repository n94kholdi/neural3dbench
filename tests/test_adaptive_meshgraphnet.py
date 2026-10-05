from __future__ import annotations

import importlib.util

import pytest
import torch

from mesh import (
    BarycentricStateTransfer,
    LocalVariationCriterion,
    MeshAdaptationConfig,
    MeshAdaptationController,
    MeshMode,
    TransferMode,
    TriangleCentroidRemesher,
    TriangleMeshGraphBuilder,
    TriangularMesh,
)
from models import MeshGraphNet, MeshGraphNetConfig
from simulation import MeshSimulationRunner
from examples.adaptive_meshgraphnet_synthetic import make_grid


def square_mesh() -> TriangularMesh:
    return TriangularMesh(
        vertices=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ),
        triangles=torch.tensor([[0, 1, 2], [0, 2, 3]]),
        node_data={
            "node_type": torch.tensor([1, 1, 2, 2]),
            "boundary": torch.tensor([1, 1, 1, 1]),
        },
        node_data_modes={
            "node_type": TransferMode.CATEGORICAL,
            "boundary": TransferMode.BOUNDARY,
        },
        cell_data={"material": torch.tensor([7, 9])},
        boundary_edges=torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]),
        boundary_tags=torch.tensor([10, 20, 30, 40]),
    )


def config(mode=MeshMode.ADAPTIVE, **kwargs) -> MeshAdaptationConfig:
    values = dict(
        mode=mode,
        refinement_threshold=0.1,
        max_nodes=20,
        max_elements=30,
        max_refinement_level=2,
    )
    values.update(kwargs)
    return MeshAdaptationConfig(**values)


def controller(adaptation_config: MeshAdaptationConfig) -> MeshAdaptationController:
    return MeshAdaptationController(
        adaptation_config,
        LocalVariationCriterion("state"),
        TriangleCentroidRemesher(),
        BarycentricStateTransfer(),
        {"category": TransferMode.CATEGORICAL},
    )


def tiny_model() -> MeshGraphNet:
    return MeshGraphNet(
        MeshGraphNetConfig(
            node_input_dim=3,
            edge_input_dim=3,
            output_dim=1,
            latent_dim=8,
            processor_steps=2,
            mlp_hidden_dim=8,
            mlp_layers=1,
        )
    )


def test_fixed_mode_has_no_mesh_or_criterion_overhead():
    class FailingCriterion(LocalVariationCriterion):
        def compute(self, *args, **kwargs):
            raise AssertionError("fixed mode must not compute indicators")

    mesh = square_mesh()
    fixed = MeshAdaptationController(
        config(MeshMode.FIXED),
        FailingCriterion(),
        TriangleCentroidRemesher(),
        BarycentricStateTransfer(),
    )
    state = {"state": mesh.vertices[:, :1]}
    outcome = fixed.adapt(1, mesh, state)
    assert outcome.mesh is mesh
    assert outcome.state["state"] is state["state"]
    assert outcome.remesh is None


def test_adaptive_refinement_is_valid_and_preserves_boundaries_and_regions():
    mesh = square_mesh()
    state = {"state": torch.tensor([[0.0], [0.0], [2.0], [0.0]])}
    outcome = controller(config()).adapt(1, mesh, state)
    refined = outcome.mesh
    assert refined.num_nodes > mesh.num_nodes
    assert refined.num_cells > mesh.num_cells
    refined.validate()
    assert torch.equal(refined.boundary_edges, mesh.boundary_edges)
    assert torch.equal(refined.boundary_tags, mesh.boundary_tags)
    assert torch.equal(refined.node_data["boundary"][:4], mesh.node_data["boundary"])
    assert torch.all(refined.node_data["boundary"][4:] == 0)
    assert set(refined.cell_data["material"].tolist()) == {7, 9}


def test_barycentric_state_transfer_is_exact_for_linear_field_and_discrete_for_categories():
    mesh = square_mesh()
    continuous = (2 * mesh.vertices[:, 0] - 3 * mesh.vertices[:, 1] + 0.5)[:, None]
    state = {
        "state": continuous,
        "category": torch.tensor([3, 3, 8, 8]),
    }
    outcome = controller(config(refinement_threshold=0.0)).adapt(1, mesh, state)
    expected = (
        2 * outcome.mesh.vertices[:, 0] - 3 * outcome.mesh.vertices[:, 1] + 0.5
    )[:, None]
    assert torch.allclose(outcome.state["state"], expected, atol=1e-6)
    assert outcome.state["category"].dtype == torch.long
    assert set(outcome.state["category"].tolist()) <= {3, 8}


def test_graph_is_rebuilt_with_fresh_bidirectional_geometric_features():
    mesh = square_mesh()
    outcome = controller(config()).adapt(
        1, mesh, {"state": torch.tensor([[0.0], [0.0], [2.0], [0.0]])}
    )
    builder = TriangleMeshGraphBuilder(["state"], include_coordinates=True)
    graph = builder(outcome.mesh, outcome.state)
    assert graph.edge_index.min() >= 0
    assert graph.edge_index.max() < outcome.mesh.num_nodes
    assert graph.edge_features.shape[0] == graph.edge_index.shape[1]
    expected_relative = (
        outcome.mesh.vertices[graph.edge_index[1]]
        - outcome.mesh.vertices[graph.edge_index[0]]
    )
    assert torch.allclose(graph.edge_features[:, :2], expected_relative)
    assert torch.allclose(
        graph.edge_features[:, 2], torch.linalg.vector_norm(expected_relative, dim=1)
    )
    edges = {tuple(edge) for edge in graph.edge_index.t().tolist()}
    assert all((target, source) in edges for source, target in edges)


def test_limits_and_schedule_are_respected():
    mesh = square_mesh()
    state = {"state": torch.tensor([[0.0], [0.0], [2.0], [0.0]])}
    limited = controller(config(adapt_every=2, max_nodes=5, max_elements=4))
    skipped = limited.adapt(1, mesh, state)
    assert skipped.mesh is mesh
    outcome = limited.adapt(2, mesh, state)
    assert outcome.mesh.num_nodes == 5
    assert outcome.mesh.num_cells == 4
    assert outcome.remesh.metadata["limited"]


def test_localized_gaussian_refines_preferentially_near_its_gradient():
    mesh = make_grid()
    center = torch.tensor([0.72, 0.68])
    state = {
        "state": torch.exp(
            -80.0 * ((mesh.vertices - center) ** 2).sum(dim=1, keepdim=True)
        )
    }
    outcome = controller(
        config(refinement_threshold=2.0, max_nodes=80, max_elements=200)
    ).adapt(
        1, mesh, state
    )
    inserted = outcome.mesh.vertices[mesh.num_nodes :]
    assert inserted.shape[0] > 0
    assert torch.all(torch.linalg.vector_norm(inserted - center, dim=1) < 0.35)


@pytest.mark.parametrize("mode", [MeshMode.FIXED, MeshMode.ADAPTIVE])
def test_end_to_end_rollout_uses_same_meshgraphnet(mode):
    torch.manual_seed(2)
    mesh = square_mesh()
    initial = {"state": torch.exp(-40 * ((mesh.vertices - 0.75) ** 2).sum(dim=1))[:, None]}
    model = tiny_model()
    builder = TriangleMeshGraphBuilder(["state"], include_coordinates=True)

    def preserve_state(mesh, state, output):
        assert output.predictions.shape[0] == mesh.num_nodes
        return dict(state)

    runner = MeshSimulationRunner(
        model,
        builder,
        mesh_adaptation=config(mode, refinement_threshold=0.001, adapt_every=1),
        state_updater=preserve_state,
    )
    result = runner.run(mesh, initial, steps=2)
    assert runner.model is model
    assert len(result.frames) == 2
    if mode is MeshMode.FIXED:
        assert result.final_mesh is mesh
        assert all(frame.model_input.node_features.shape[0] == 4 for frame in result.frames)
    else:
        assert result.final_mesh.num_nodes > mesh.num_nodes
        assert result.frames[1].model_input.node_features.shape[0] > 4


def test_adaptive_rollout_keeps_model_gradients():
    mesh = square_mesh()
    state = {"state": mesh.vertices[:, :1].clone().requires_grad_()}
    model = tiny_model()
    runner = MeshSimulationRunner(
        model,
        TriangleMeshGraphBuilder(["state"], include_coordinates=True),
        mesh_adaptation=config(refinement_threshold=0.01),
        state_updater=lambda mesh, state, output: dict(state),
    )
    result = runner.run(mesh, state, steps=1)
    result.frames[0].model_output.predictions.sum().backward()
    assert state["state"].grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_disconnected_adaptive_graphs_can_still_be_batched():
    model = tiny_model()
    builder = TriangleMeshGraphBuilder(["state"], include_coordinates=True)
    first_mesh = square_mesh()
    first = builder(first_mesh, {"state": first_mesh.vertices[:, :1]})
    shifted_mesh = square_mesh()
    shifted_mesh.vertices = shifted_mesh.vertices + 2
    second = builder(shifted_mesh, {"state": shifted_mesh.vertices[:, :1]})
    offset = first.node_features.shape[0]
    from models import ModelInput

    batch = ModelInput(
        node_features=torch.cat((first.node_features, second.node_features)),
        edge_index=torch.cat((first.edge_index, second.edge_index + offset), dim=1),
        edge_features=torch.cat((first.edge_features, second.edge_features)),
        batch=torch.tensor([0] * offset + [1] * second.node_features.shape[0]),
    )
    assert model(batch).predictions.shape == (8, 1)


def test_optional_pyg_data_conversion_has_clear_dependency_boundary():
    mesh = square_mesh()
    builder = TriangleMeshGraphBuilder(["state"])
    state = {"state": mesh.vertices[:, :1]}
    if importlib.util.find_spec("torch_geometric") is None:
        with pytest.raises(ImportError, match="optional"):
            builder.to_pyg_data(mesh, state)
    else:
        data = builder.to_pyg_data(mesh, state)
        assert data.num_nodes == mesh.num_nodes
