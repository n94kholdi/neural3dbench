from __future__ import annotations

import pytest
import torch

from models import FoldingNet, FoldingNetConfig, ModelInput, ModelRegistry, PCN, PCNConfig, create_model

SMALL = {
    "pcn": {"latent_dim": 64, "hidden_dim": 32, "num_coarse": 16, "grid_size": 2},
    "foldingnet": {"latent_dim": 64, "hidden_dim": 32, "grid_size": 5},
}


@pytest.mark.parametrize(("name", "model_cls", "config_cls"), [("pcn", PCN, PCNConfig), ("foldingnet", FoldingNet, FoldingNetConfig)])
def test_registry_and_config(name, model_cls, config_cls):
    assert ModelRegistry.get(name) is model_cls
    model = create_model({"name": name, **SMALL[name]})
    assert isinstance(model, model_cls) and isinstance(model.config, config_cls)
    with pytest.raises(ValueError):
        config_cls(input_dim=2)
    with pytest.raises(ValueError):
        config_cls(grid_size=0)


@pytest.mark.parametrize("name", ["pcn", "foldingnet"])
def test_output_is_new_point_set(name):
    model = create_model({"name": name, **SMALL[name]})
    for num_points in (7, 50):  # output size does not follow the input size
        out = model(ModelInput(coordinates=torch.randn(3, num_points, 3)))
        assert out.predictions.shape == (3, model.config.num_output_points, 3)
    if name == "pcn":
        assert out.auxiliary["coarse"].shape == (3, 16, 3)
    out.predictions.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("name", ["pcn", "foldingnet"])
def test_padding_and_features(name):
    model = create_model({"name": name, **SMALL[name]}).eval()
    points = torch.randn(2, 20, 3)
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[1, 12:] = False
    padded = model(ModelInput(coordinates=points, point_mask=mask)).predictions[1]
    alone = model(ModelInput(coordinates=points[1:, :12])).predictions[0]
    assert torch.allclose(padded, alone, atol=1e-5)

    with_features = create_model({"name": name, "input_dim": 5, **SMALL[name]})
    with_features(ModelInput(coordinates=points, point_features=torch.randn(2, 20, 2)))
    with pytest.raises(ValueError):
        with_features(ModelInput(coordinates=points))
    with pytest.raises(ValueError):
        model(ModelInput(point_features=torch.randn(2, 20, 2)))
