import pytest
import torch

from models import GAT, GATConfig, ModelInput, ModelRegistry, create_model
from models.graph.layers import GATBlock


def _bidirectional_chain(node_count: int) -> torch.Tensor:
    source = torch.arange(node_count - 1, dtype=torch.long)
    forward = torch.stack((source, source + 1))
    return torch.cat((forward, forward.flip(0)), dim=1)


def _input(node_count: int, input_dim: int = 4) -> ModelInput:
    return ModelInput(
        node_features=torch.randn(node_count, input_dim),
        edge_index=_bidirectional_chain(node_count),
    )


def test_gat_direct_construction_config_and_registry():
    config = GATConfig(
        input_dim=5,
        output_dim=2,
        hidden_dim=16,
        num_layers=3,
        num_heads=[1, 2, 4],
        activation="gelu",
        dropout=0.1,
        attention_dropout=0.2,
        normalization="layer_norm",
        residual=True,
    )
    assert isinstance(GAT(config), GAT)
    assert config.layer_heads == (1, 2, 4)

    model = create_model(
        {
            "name": "gat",
            "input_dim": 5,
            "output_dim": 2,
            "hidden_dim": 16,
            "num_layers": 2,
            "num_heads": 4,
        }
    )
    assert isinstance(model, GAT)
    assert ModelRegistry.get("gat") is GAT
    assert ModelRegistry.create("gat", config).config is config
    assert create_model(config).config is config


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_dim", 0, "input_dim"),
        ("hidden_dim", 0, "hidden_dim"),
        ("output_dim", 0, "output_dim"),
        ("num_layers", 0, "num_layers"),
        ("num_heads", 0, "num_heads"),
        ("dropout", 1.0, "dropout"),
        ("attention_dropout", -0.1, "attention_dropout"),
        ("activation", "not_an_activation", "activation"),
        ("normalization", "group_norm", "normalization"),
    ],
)
def test_gat_rejects_invalid_config(field, value, message):
    kwargs = {field: value}
    with pytest.raises((TypeError, ValueError), match=message):
        GATConfig(**kwargs)


def test_gat_validates_head_sequence_and_concat_dimensions():
    with pytest.raises(ValueError, match="one value per GAT layer"):
        GATConfig(num_layers=2, num_heads=[2])
    with pytest.raises(ValueError, match="divisible"):
        GATConfig(hidden_dim=10, num_heads=4, concat_heads=True)

    config = GATConfig(hidden_dim=10, num_heads=4, concat_heads=False)
    assert config.layer_heads == (4, 4)


@pytest.mark.parametrize(
    ("num_heads", "concat_heads", "output_dim"),
    [(1, True, 1), (4, True, 3), (3, False, 5)],
)
def test_gat_forward_has_fixed_output_shape(num_heads, concat_heads, output_dim):
    model = GAT(
        GATConfig(
            input_dim=4,
            output_dim=output_dim,
            hidden_dim=12,
            num_layers=2,
            num_heads=num_heads,
            concat_heads=concat_heads,
        )
    )
    output = model(_input(7))
    assert output.predictions.shape == (7, output_dim)
    assert output.latent.shape == (7, 12)


def test_gat_supports_variable_graph_sizes():
    model = GAT(GATConfig(input_dim=3, output_dim=2, hidden_dim=8, num_heads=2))
    for node_count in (4, 9, 15):
        output = model(_input(node_count, input_dim=3))
        assert output.predictions.shape == (node_count, 2)


def test_gat_rejects_invalid_inputs():
    model = GAT(GATConfig(input_dim=4, output_dim=2, hidden_dim=8, num_heads=2))
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    with pytest.raises(ValueError, match="requires inputs: edge_index"):
        model(ModelInput(node_features=torch.randn(3, 4)))
    with pytest.raises(ValueError, match="requires inputs: node_features"):
        model(ModelInput(edge_index=edges))
    with pytest.raises(ValueError, match="edge_index must have shape"):
        model(ModelInput(node_features=torch.randn(3, 4), edge_index=edges[:1]))
    with pytest.raises(ValueError, match="input feature dimension"):
        model(ModelInput(node_features=torch.randn(3, 3), edge_index=edges))
    with pytest.raises(TypeError, match="floating-point"):
        model(ModelInput(node_features=torch.ones(3, 4, dtype=torch.long), edge_index=edges))
    with pytest.raises(TypeError, match="integer dtype"):
        model(ModelInput(node_features=torch.randn(3, 4), edge_index=edges.float()))


def test_gat_gradients_and_disconnected_batch_support():
    torch.manual_seed(3)
    model = GAT(
        GATConfig(
            input_dim=3,
            output_dim=1,
            hidden_dim=8,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
            residual=True,
        )
    )
    edges_a = _bidirectional_chain(4)
    edges_b = _bidirectional_chain(5) + 4
    batch = torch.tensor([0] * 4 + [1] * 5)
    inputs = ModelInput(
        node_features=torch.randn(9, 3),
        edge_index=torch.cat((edges_a, edges_b), dim=1),
        batch=batch,
    )

    output = model(inputs)
    assert output.predictions.shape == (9, 1)
    assert torch.equal(output.metadata["batch"], batch)
    output.predictions.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_gat_residual_supports_matching_and_projected_dimensions():
    edges = _bidirectional_chain(5)
    matching = GATBlock(8, 8, num_heads=2, residual=True)
    projected = GATBlock(5, 8, num_heads=2, residual=True)

    assert isinstance(matching.residual_projection, torch.nn.Identity)
    assert isinstance(projected.residual_projection, torch.nn.Linear)
    assert matching(torch.randn(5, 8), edges).shape == (5, 8)
    assert projected(torch.randn(5, 5), edges).shape == (5, 8)


def test_gat_optionally_returns_attention_without_changing_predictions():
    torch.manual_seed(4)
    config = GATConfig(
        input_dim=4,
        output_dim=2,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        attention_dropout=0.0,
        return_attention_weights=False,
    )
    model = GAT(config).eval()
    inputs = _input(6)
    ordinary = model(inputs)
    assert ordinary.auxiliary is None

    config.return_attention_weights = True
    inspected = model(inputs)
    assert inspected.predictions.shape == (6, 2)
    assert torch.allclose(ordinary.predictions, inspected.predictions)
    weights = inspected.auxiliary["attention_weights"]
    assert len(weights) == 2
    for layer_weights in weights:
        edge_index = layer_weights["edge_index"]
        coefficients = layer_weights["coefficients"]
        assert edge_index.shape[0] == 2
        assert coefficients.shape == (edge_index.shape[1], 2)
        incoming_sums = coefficients.new_zeros((6, 2))
        incoming_sums.index_add_(0, edge_index[1], coefficients)
        assert torch.allclose(incoming_sums, torch.ones_like(incoming_sums), atol=1e-6)


def test_gat_replaces_existing_self_loops_with_one_per_node():
    model = GAT(GATConfig(input_dim=2, hidden_dim=4, num_heads=2, add_self_loops=True))
    edges = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.long)
    prepared = model._prepare_edge_index(edges, num_nodes=3)
    loops = prepared[:, prepared[0] == prepared[1]]
    assert torch.equal(loops, torch.arange(3).repeat(2, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gat_runs_on_cuda_when_available():
    model = GAT(
        GATConfig(input_dim=3, output_dim=2, hidden_dim=8, num_heads=2)
    ).to("cuda")
    inputs = ModelInput(
        node_features=torch.randn(7, 3, device="cuda"),
        edge_index=_bidirectional_chain(7).to("cuda"),
    )
    output = model(inputs)
    assert output.predictions.shape == (7, 2)
    assert output.predictions.is_cuda
