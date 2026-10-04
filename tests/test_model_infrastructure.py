import pytest

from models import BaseNetwork, BaseModelConfig, ModelInput, ModelOutput, ModelRegistry, create_model


class DummyGraphConfig(BaseModelConfig):
    def __init__(self, hidden_dim: int = 8, out_dim: int = 3):
        super().__init__(name="dummy_graph")
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim


class DummyGraphModel(BaseNetwork):
    def __init__(self, config: DummyGraphConfig | None = None):
        super().__init__(config=config or DummyGraphConfig())

    @property
    def required_inputs(self):
        return {"coordinates", "node_features", "edge_index"}

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        return ModelOutput(predictions=inputs.coordinates)


class DummyCoordinateConfig(BaseModelConfig):
    def __init__(self, hidden_dim: int = 16):
        super().__init__(name="dummy_coordinate")
        self.hidden_dim = hidden_dim


class DummyCoordinateModel(BaseNetwork):
    def __init__(self, config: DummyCoordinateConfig | None = None):
        super().__init__(config=config or DummyCoordinateConfig())

    @property
    def required_inputs(self):
        return {"coordinates"}

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        return ModelOutput(predictions=inputs.coordinates)


def test_model_registry_registers_and_creates_models():
    ModelRegistry.register("dummy_graph", DummyGraphModel)
    model = create_model({"name": "dummy_graph", "hidden_dim": 12, "out_dim": 5})

    assert isinstance(model, DummyGraphModel)
    assert model.config.hidden_dim == 12
    assert model.config.out_dim == 5


def test_duplicate_registration_raises():
    ModelRegistry.register("duplicate_guard", DummyGraphModel)

    with pytest.raises(ValueError):
        ModelRegistry.register("duplicate_guard", DummyCoordinateModel)


def test_invalid_registration_raises():
    with pytest.raises(TypeError):
        ModelRegistry.register("bad_model", object)


def test_required_input_validation():
    model = DummyGraphModel()

    inputs = ModelInput(coordinates=[[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        model.validate_inputs(inputs)


def test_optional_inputs_are_allowed():
    model = DummyCoordinateModel()
    inputs = ModelInput(
        coordinates=[[0.0, 0.0]],
        physical_parameters={"diffusion": 1.0},
        global_features={"time": 0.0},
    )

    model.validate_inputs(inputs)
    assert inputs.coordinates.shape == (1, 2)


def test_model_output_behaves_as_common_interface():
    output = ModelOutput(
        predictions={"field": torch_tensor([1.0, 2.0])},
        latent={"hidden": torch_tensor([0.5])},
        auxiliary={"loss_terms": {"data": 0.1}},
        metadata={"source": "unit-test"},
    )

    assert output.predictions["field"].shape == (2,)
    assert output.main_prediction == output.predictions
    assert output.metadata["source"] == "unit-test"


def test_dummy_graph_and_coordinate_models_run():
    graph_model = DummyGraphModel()
    coord_model = DummyCoordinateModel()

    graph_inputs = ModelInput(
        coordinates=[[0.0, 0.0], [1.0, 0.0]],
        node_features=[[1.0], [2.0]],
        edge_index=[[0, 1], [1, 0]],
    )
    coord_inputs = ModelInput(coordinates=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    graph_output = graph_model(graph_inputs)
    coord_output = coord_model(coord_inputs)

    assert graph_output.main_prediction is graph_output.predictions
    assert coord_output.main_prediction is coord_output.predictions


def torch_tensor(values):
    import torch

    return torch.tensor(values)
