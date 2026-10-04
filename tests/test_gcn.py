import pytest
import torch

from models import ModelInput, ModelRegistry, create_model
from models.graph import GCN, GCNConfig


def _make_graph(node_count: int, input_dim: int = 4, output_dim: int = 3):
    coords = torch.linspace(0.0, 1.0, steps=node_count, dtype=torch.float32).unsqueeze(-1)
    features = torch.randn(node_count, input_dim)
    edges = []
    for i in range(node_count):
        for j in range(i + 1, min(node_count, i + 3)):
            if i != j:
                edges.append([i, j])
                edges.append([j, i])
    if not edges:
        edges = [[0, 0], [0, 0]]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return ModelInput(
        coordinates=coords,
        node_features=features,
        edge_index=edge_index,
        batch=torch.zeros(node_count, dtype=torch.long),
    ), coords


def test_gcn_config_and_registry():
    config = GCNConfig(
        input_dim=5,
        output_dim=2,
        hidden_dim=16,
        num_layers=3,
        activation="gelu",
        dropout=0.1,
        normalization="layer_norm",
        residual=True,
    )
    assert config.name == "gcn"

    model = create_model({
        "name": "gcn",
        "input_dim": 5,
        "output_dim": 2,
        "hidden_dim": 16,
        "num_layers": 3,
        "activation": "gelu",
        "dropout": 0.1,
        "normalization": "layer_norm",
        "residual": True,
    })
    assert isinstance(model, GCN)
    assert ModelRegistry.get("gcn") is GCN


def test_gcn_forward_outputs_expected_shape():
    model = GCN(GCNConfig(input_dim=4, output_dim=3, hidden_dim=8, num_layers=2))
    inputs = ModelInput(
        node_features=torch.randn(6, 4),
        edge_index=torch.tensor([
            [0, 1, 1, 2, 3, 4, 4, 5],
            [1, 2, 3, 4, 4, 5, 0, 2],
        ], dtype=torch.long),
    )

    output = model(inputs)

    assert output.predictions.shape == (6, 3)
    assert output.main_prediction is output.predictions


def test_gcn_supports_different_graph_sizes():
    model = GCN(GCNConfig(input_dim=3, output_dim=2, hidden_dim=9, num_layers=2, dropout=0.0))

    for node_count in (4, 8, 13):
        edges = []
        for node in range(node_count - 1):
            edges.append([node, node + 1])
            edges.append([node + 1, node])
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        inputs = ModelInput(node_features=torch.randn(node_count, 3), edge_index=edge_index)
        output = model(inputs)
        assert output.predictions.shape == (node_count, 2)


def test_gcn_rejects_invalid_inputs():
    model = GCN(GCNConfig(input_dim=4, output_dim=2, hidden_dim=6, num_layers=2))

    with pytest.raises(ValueError, match="requires inputs: edge_index"):
        model.validate_inputs(ModelInput(node_features=torch.randn(4, 4)))

    with pytest.raises(ValueError, match="requires inputs: node_features"):
        model.validate_inputs(ModelInput(edge_index=torch.tensor([[0, 1], [1, 0]])))

    with pytest.raises(ValueError, match="edge_index"):
        model.validate_inputs(ModelInput(node_features=torch.randn(4, 4), edge_index=torch.tensor([[0, 1]])))

    with pytest.raises(ValueError, match="input feature"):
        model.validate_inputs(ModelInput(node_features=torch.randn(4, 3), edge_index=torch.tensor([[0, 1], [1, 0]])))


def test_gcn_gradients_and_batch_support():
    model = GCN(GCNConfig(input_dim=3, output_dim=1, hidden_dim=5, num_layers=2, dropout=0.0, residual=True))
    graph_a = ModelInput(
        node_features=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 0]),
    )
    graph_b = ModelInput(
        node_features=torch.randn(5, 3),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3], [1, 2, 3, 4, 0, 0, 1, 2]], dtype=torch.long),
        batch=torch.tensor([1, 1, 1, 1, 1]),
    )

    output_a = model(graph_a)
    output_b = model(graph_b)

    assert output_a.predictions.shape == (4, 1)
    assert output_b.predictions.shape == (5, 1)

    loss = output_a.predictions.sum() + output_b.predictions.sum()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None


def test_gcn_training_decreases_loss():
    torch.manual_seed(0)
    model = GCN(GCNConfig(input_dim=3, output_dim=1, hidden_dim=6, num_layers=2, activation="relu", dropout=0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    coordinates = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=torch.float32)
    features = torch.cat([coordinates, torch.ones((4, 1))], dim=1)
    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 3, 2, 3],
        [1, 0, 2, 0, 3, 1, 3, 2],
    ], dtype=torch.long)
    target = torch.sin(coordinates[:, 0]) + torch.cos(coordinates[:, 1])
    target = target.unsqueeze(-1)

    inputs = ModelInput(node_features=features, edge_index=edge_index, coordinates=coordinates)

    initial_loss = None
    for _ in range(30):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = ((outputs.predictions - target) ** 2).mean()
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()

    final_loss = float(((model(inputs).predictions - target) ** 2).mean().detach())
    assert initial_loss is not None
    assert final_loss < initial_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gcn_runs_on_cuda_when_available():
    model = GCN(GCNConfig(input_dim=3, output_dim=2, hidden_dim=4, num_layers=2, dropout=0.0)).to("cuda")
    inputs = ModelInput(
        node_features=torch.randn(7, 3, device="cuda"),
        edge_index=torch.tensor([
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 0],
        ], dtype=torch.long, device="cuda"),
    )
    outputs = model(inputs)
    assert outputs.predictions.shape == (7, 2)
