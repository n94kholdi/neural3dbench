import pytest
import torch

from models import (
    MeshGraphNet,
    MeshGraphNetConfig,
    ModelInput,
    ModelRegistry,
    create_model,
)
from models.graph.meshgraphnet import MeshGraphNetBlock


def _graph(node_count: int, node_dim: int = 4, edge_dim: int = 3) -> ModelInput:
    source = torch.arange(node_count - 1, dtype=torch.long)
    forward = torch.stack((source, source + 1))
    edge_index = torch.cat((forward, forward.flip(0)), dim=1)
    return ModelInput(
        node_features=torch.randn(node_count, node_dim),
        edge_index=edge_index,
        edge_features=torch.randn(edge_index.shape[1], edge_dim),
    )


def _small_config(**kwargs) -> MeshGraphNetConfig:
    values = {
        "node_input_dim": 4,
        "edge_input_dim": 3,
        "output_dim": 2,
        "latent_dim": 8,
        "processor_steps": 3,
        "mlp_hidden_dim": 12,
        "mlp_layers": 1,
    }
    values.update(kwargs)
    return MeshGraphNetConfig(**values)


def test_meshgraphnet_direct_config_and_registry_construction():
    config = _small_config()
    assert isinstance(MeshGraphNet(config), MeshGraphNet)
    assert create_model(config).config is config
    assert isinstance(create_model(config.to_dict()), MeshGraphNet)
    assert isinstance(ModelRegistry.create("meshgraphnet", config), MeshGraphNet)
    assert ModelRegistry.get("meshgraphnet") is MeshGraphNet


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("node_input_dim", 0),
        ("edge_input_dim", 0),
        ("output_dim", 0),
        ("latent_dim", 0),
        ("processor_steps", 0),
        ("mlp_hidden_dim", 0),
        ("mlp_layers", 0),
        ("activation", "invalid"),
        ("normalization", "batch_norm"),
    ],
)
def test_meshgraphnet_rejects_invalid_config(field, value):
    with pytest.raises(ValueError, match=field):
        _small_config(**{field: value})


def test_encoder_processor_decoder_shapes():
    model = MeshGraphNet(_small_config(processor_steps=4))
    graph = _graph(10)
    nodes, edges = model.encoder(graph.node_features, graph.edge_features)
    assert nodes.shape == (10, 8)
    assert edges.shape == (18, 8)

    processed_nodes, processed_edges = model.processor(
        nodes, edges, graph.edge_index
    )
    assert processed_nodes.shape == nodes.shape
    assert processed_edges.shape == edges.shape
    assert len(model.processor.blocks) == 4
    assert model.processor.blocks[0] is not model.processor.blocks[1]

    predictions = model.decoder(processed_nodes)
    assert predictions.shape == (10, 2)


@pytest.mark.parametrize("node_count", [20, 100, 500])
def test_forward_supports_variable_graph_sizes(node_count):
    model = MeshGraphNet(_small_config(processor_steps=2))
    output = model(_graph(node_count))
    assert output.predictions.shape == (node_count, 2)
    assert output.latent.shape == (node_count, 8)
    assert output.auxiliary["edge_latent"].shape == (2 * (node_count - 1), 8)


def test_message_block_uses_edge_first_sum_aggregation_and_residuals():
    torch.manual_seed(0)
    block = MeshGraphNetBlock(
        4,
        mlp_hidden_dim=6,
        mlp_layers=1,
        activation="relu",
        normalization="none",
        residual=True,
    )
    nodes = torch.randn(3, 4)
    edges = torch.randn(2, 4)
    edge_index = torch.tensor([[0, 2], [1, 1]])

    edge_update = block.edge_mlp(
        torch.cat((nodes[edge_index[0]], nodes[edge_index[1]], edges), dim=-1)
    )
    aggregate = torch.zeros_like(nodes)
    aggregate.index_add_(0, edge_index[1], edge_update)
    node_update = block.node_mlp(torch.cat((nodes, aggregate), dim=-1))

    actual_nodes, actual_edges = block(nodes, edges, edge_index)
    assert torch.allclose(actual_edges, edges + edge_update)
    assert torch.allclose(actual_nodes, nodes + node_update)


def test_predictions_depend_on_edge_features_and_connectivity():
    torch.manual_seed(1)
    model = MeshGraphNet(_small_config()).eval()
    graph = _graph(8)
    baseline = model(graph).predictions

    changed_features = ModelInput(
        node_features=graph.node_features,
        edge_index=graph.edge_index,
        edge_features=graph.edge_features + 2.0,
    )
    changed_connectivity = ModelInput(
        node_features=graph.node_features,
        edge_index=graph.edge_index.flip(0),
        edge_features=graph.edge_features,
    )
    assert not torch.allclose(baseline, model(changed_features).predictions)
    assert not torch.allclose(baseline, model(changed_connectivity).predictions)


def test_gradients_reach_encoder_processor_and_decoder():
    model = MeshGraphNet(_small_config())
    model(_graph(9)).predictions.sum().backward()
    for module in (model.encoder, model.processor, model.decoder):
        assert all(parameter.grad is not None for parameter in module.parameters())


def test_disconnected_graph_batch():
    model = MeshGraphNet(_small_config())
    first = _graph(4)
    second = _graph(5)
    inputs = ModelInput(
        node_features=torch.cat((first.node_features, second.node_features)),
        edge_index=torch.cat((first.edge_index, second.edge_index + 4), dim=1),
        edge_features=torch.cat((first.edge_features, second.edge_features)),
        batch=torch.tensor([0] * 4 + [1] * 5),
    )
    output = model(inputs)
    assert output.predictions.shape == (9, 2)
    assert torch.equal(output.metadata["batch"], inputs.batch)


def test_meshgraphnet_rejects_invalid_inputs():
    model = MeshGraphNet(_small_config())
    valid = _graph(5)

    with pytest.raises(ValueError, match="edge_features"):
        model(ModelInput(node_features=valid.node_features, edge_index=valid.edge_index))
    with pytest.raises(ValueError, match="edge_index"):
        model(ModelInput(node_features=valid.node_features, edge_features=valid.edge_features))
    with pytest.raises(ValueError, match="node_features"):
        model(ModelInput(edge_index=valid.edge_index, edge_features=valid.edge_features))
    with pytest.raises(ValueError, match="edge_index must have shape"):
        model(
            ModelInput(
                node_features=valid.node_features,
                edge_index=valid.edge_index[:1],
                edge_features=valid.edge_features,
            )
        )
    with pytest.raises(ValueError, match="node feature dimension"):
        invalid = _graph(5, node_dim=2)
        model(invalid)
    with pytest.raises(ValueError, match="edge feature dimension"):
        invalid = _graph(5, edge_dim=2)
        model(invalid)
    with pytest.raises(ValueError, match="same number of edges"):
        model(
            ModelInput(
                node_features=valid.node_features,
                edge_index=valid.edge_index,
                edge_features=valid.edge_features[:-1],
            )
        )
    with pytest.raises(ValueError, match="range"):
        invalid_edges = valid.edge_index.clone()
        invalid_edges[0, 0] = 10
        model(
            ModelInput(
                node_features=valid.node_features,
                edge_index=invalid_edges,
                edge_features=valid.edge_features,
            )
        )


def test_empty_edge_graph_is_supported():
    model = MeshGraphNet(_small_config())
    output = model(
        ModelInput(
            node_features=torch.randn(4, 4),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_features=torch.empty((0, 3)),
        )
    )
    assert output.predictions.shape == (4, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_meshgraphnet_runs_on_cuda_when_available():
    model = MeshGraphNet(_small_config()).cuda()
    graph = _graph(7)
    inputs = ModelInput(
        node_features=graph.node_features.cuda(),
        edge_index=graph.edge_index.cuda(),
        edge_features=graph.edge_features.cuda(),
    )
    output = model(inputs)
    assert output.predictions.shape == (7, 2)
    assert output.predictions.is_cuda
