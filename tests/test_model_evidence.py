"""Catalog claims are checked against the papers (evidence) and against the code (measured symmetries)."""

import pytest
import torch

yaml = pytest.importorskip("yaml")

from models import ModelInput, ModelOutput, ModelRegistry, collate_meshes  # noqa: E402
from tools.mcp_advisor import guardrails  # noqa: E402
from tools.mcp_advisor.advisor import OUTPUTS, Advisor  # noqa: E402


@pytest.fixture(scope="module")
def advisor():
    return Advisor()


def _meta(advisor, name):
    return advisor.resolve(name)["metadata"]


def test_every_model_has_paper_evidence_and_outputs(advisor):
    for name, model in advisor.state["models"].items():
        meta = model["metadata"]
        assert set(meta["outputs"]) <= set(OUTPUTS), name
        evidence = meta["evidence"]
        assert evidence["paper"]["url"].startswith(("https://arxiv.org/", "https://openreview.net/")), name
        assert evidence["checked"] and evidence["benchmark_datasets"], name
        for row in evidence["reported"]:
            assert {"dataset", "metric", "value"} <= set(row), name


def _predict(model, x):
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(x if isinstance(x, ModelInput) else ModelInput(coordinates=x))
    return out.predictions if isinstance(out, ModelOutput) else out


@pytest.mark.parametrize("name", ["pointnet", "pointnet2", "pointnext", "dgcnn", "transolver", "mlp", "siren"])
def test_claimed_permutation_symmetry_holds(advisor, name):
    assert "permutation" in _meta(advisor, name)["symmetries"]
    model = ModelRegistry.create(name).eval()
    x, perm = torch.rand(2, 128, 3), torch.randperm(128)
    y, y_perm = _predict(model, x), _predict(model, x[:, perm])
    expected = y[:, perm] if y.ndim == 3 and y.shape[1] == 128 else y
    assert torch.allclose(y_perm, expected, atol=1e-4), name


def _sphere_mesh():
    import math

    rings, segments = 6, 10
    verts = [[0.0, 0.0, 1.0]] + [
        [math.sin(math.pi * i / rings) * math.cos(2 * math.pi * j / segments),
         math.sin(math.pi * i / rings) * math.sin(2 * math.pi * j / segments), math.cos(math.pi * i / rings)]
        for i in range(1, rings) for j in range(segments)
    ] + [[0.0, 0.0, -1.0]]
    faces = [[0, 1 + j, 1 + (j + 1) % segments] for j in range(segments)]
    for i in range(rings - 2):
        for j in range(segments):
            a, b = 1 + i * segments + j, 1 + i * segments + (j + 1) % segments
            faces += [[a, a + segments, b], [b, a + segments, b + segments]]
    last, base = len(verts) - 1, 1 + (rings - 2) * segments
    faces += [[last, base + (j + 1) % segments, base + j] for j in range(segments)]
    return torch.tensor(verts) * torch.tensor([1.0, 1.3, 0.8]), torch.tensor(faces)


@pytest.mark.parametrize("name, config, invariant", [
    ("meshwalker", {}, True),
    ("diffusionnet", {"input_features": "hks"}, True),
    ("diffusionnet", {}, False),  # default xyz input: the catalog must not claim translation
])
def test_mesh_translation_claims(advisor, name, config, invariant):
    verts, faces = _sphere_mesh()
    model = ModelRegistry.create(name, config).eval()
    a = _predict(model, collate_meshes([(verts, faces)]))
    b = _predict(model, collate_meshes([(verts + torch.tensor([3.0, -2.0, 1.0]), faces)]))
    assert torch.allclose(a, b, atol=1e-4) == invariant
    if not config:
        assert ("translation" in _meta(advisor, name)["symmetries"]) == invariant


def test_required_output_drops_models_without_that_head(advisor):
    result = advisor.recommend_architecture("segmentation classification part", "mesh_graph", 2000,
                                            required_output="point_label")
    usable = [r["model"] for r in result["recommendations"] if r["supports_output"]]
    assert result["verdict"] == "fit" and usable[0] == "diffusionnet"
    assert "meshnet" not in usable and "meshwalker" not in usable


def test_no_fit_names_networks_and_datasets(advisor):
    nonrigid = advisor.recommend_architecture("registration", "point_cloud", 1024, required_output="deformation_field")
    assert nonrigid["verdict"] == "no_fit"
    assert nonrigid["suggested_networks"][0]["reference"] == "https://arxiv.org/abs/2205.12796"
    assert nonrigid["suggested_datasets"]
    operator = advisor.recommend_architecture("pde operator", "continuous_coords", 4096, required_output="point_field")
    assert operator["verdict"] == "no_fit" and operator["suggested_networks"][0]["name"] == "DeepONet"
    assert any("fno3d" in alt or "transolver" in alt for alt in operator["convertible_alternatives"])


def test_heat_diffusion_is_not_shape_generation(advisor):
    result = advisor.recommend_architecture("surrogate simulation heat diffusion", "mesh_graph", 2000)
    assert result["verdict"] != "no_fit"


@pytest.mark.parametrize("name", sorted(set(ModelRegistry._registry) - {"pointnet++"}))
def test_training_setup_is_valid_for_every_model(advisor, name):
    from tools.dashboard.trainer import parse_lr_schedule

    result = guardrails.recommend_training_setup(advisor, name, 1024, 8.0, ["permutation", "rotation"])
    setup = result["setup"]
    assert setup["optimizer"] in guardrails.DASHBOARD_OPTIMIZERS and setup["learning_rate"] > 0
    assert setup["batch_size"] >= 1 and 1 <= setup["epochs"] <= 300
    parse_lr_schedule(setup["lr_schedule"])
    assert result["reasons"]


def test_training_setup_follows_the_paper(advisor):
    dgcnn = guardrails.recommend_training_setup(advisor, "dgcnn", 1024, 24.0)["setup"]
    assert (dgcnn["optimizer"], dgcnn["learning_rate"], dgcnn["momentum"]) == ("sgd", 0.1, 0.9)
    assert dgcnn["lr_schedule"] == "cosine:0.001" and dgcnn["batch_size"] == 32
    dcp = guardrails.recommend_training_setup(advisor, "dcp", 1024, 24.0, max_epochs=125)["setup"]
    assert dcp["lr_schedule"] == "multistep:38,75,100:0.1"  # paper milestones 75/150/200 of 250, halved
    rotated = guardrails.recommend_training_setup(advisor, "diffusionnet", 2000, 8.0, ["rotation"])["setup"]
    assert rotated["model_config"] == {"input_features": "hks"} and not rotated["augment_rotation"]
