import asyncio
import textwrap

import pytest

pytest.importorskip("yaml")

from tools.mcp_advisor.advisor import (  # noqa: E402
    CATALOG_PATH,
    REPRESENTATIONS,
    Advisor,
    hardware_tier,
    parse_markdown_doc,
)

EXPECTED_MODELS = {
    "pointnet",
    "pointnet2",
    "gcn",
    "gat",
    "meshgraphnet",
    "adaptive_meshgraphnet",
    "unet3d",
    "fno3d",
    "siren",
    "deep_nurbs",
    "transolver",
}


@pytest.fixture(scope="module")
def advisor():
    advisor = Advisor()
    advisor.refresh()
    return advisor


def test_every_registered_model_is_curated(advisor):
    report = advisor.refresh()
    assert report["import_errors"] == {}
    assert report["uncurated_models"] == [], (
        "Add these models to tools/mcp_advisor/catalog.yaml: " + ", ".join(report["uncurated_models"])
    )
    assert report["catalog_only_models"] == []


def test_list_supported_models_covers_initial_architectures(advisor):
    names = {row["name"] for row in advisor.list_supported_models()}
    assert EXPECTED_MODELS <= names

    voxel = {row["name"] for row in advisor.list_supported_models(representation="voxel")}
    assert voxel == {"unet3d", "fno3d"}
    operators = {row["name"] for row in advisor.list_supported_models(task="pde_operator")}
    assert {"fno3d", "transolver"} <= operators
    assert "pointnet" not in operators


def test_list_supported_models_rejects_unknown_representation(advisor):
    with pytest.raises(ValueError):
        advisor.list_supported_models(representation="hologram")


def test_dossier_merges_registry_docs_and_catalog(advisor):
    dossier = advisor.get_model_dossier("PointNet++")
    assert dossier["name"] == "pointnet2"
    assert "pointnet++" in dossier["aliases"]
    assert dossier["tensor_contract"]["required_inputs"] == ["coordinates"]
    assert any("[B, N, 3]" in line for line in dossier["tensor_contract"]["from_docs"])
    assert dossier["docs"] == ["docs/pointnet2.md"]
    assert dossier["implementation"]["parameter_count_default"] > 0
    assert dossier["merits"] and dossier["demerits"] and dossier["inductive_biases"]

    fno = advisor.get_model_dossier("fno3d")
    assert fno["docs"] == ["docs/fno.md"]
    assert "examples/train_fno3d_synthetic.py" in fno["examples"]
    assert any("arxiv.org/abs/2010.08895" in ref["url"] for ref in fno["references"])


def test_dossier_unknown_model(advisor):
    with pytest.raises(KeyError):
        advisor.get_model_dossier("not_a_model")


def test_resource_footprint_scaling_and_tiers(advisor):
    small = advisor.estimate_resource_footprint("pointnet", 1024)
    assert small["hardware_tier"]["training"] == "<8GB (consumer GPU)"

    dense = advisor.estimate_resource_footprint("dgcnn", 100_000)
    assert dense["hardware_tier"]["training"] == "Multi-GPU / >24GB required"
    assert dense["scaling_if_elements_double"] > 3

    fp32 = advisor.estimate_resource_footprint("unet3d", 64**3, batch_size=2)
    bf16 = advisor.estimate_resource_footprint("unet3d", 64**3, batch_size=2, precision="bf16")
    assert bf16["memory_gb"]["activations_training"] == pytest.approx(
        fp32["memory_gb"]["activations_training"] / 2, rel=0.01
    )
    with pytest.raises(ValueError):
        advisor.estimate_resource_footprint("unet3d", 0)


def test_hardware_tier_boundaries():
    assert hardware_tier(7.9) == "<8GB (consumer GPU)"
    assert hardware_tier(20) == "16-24GB (workstation GPU)"
    assert hardware_tier(30) == "Multi-GPU / >24GB required"


def test_recommendations_respect_input_and_flag_vram(advisor):
    result = advisor.recommend_architecture(
        "Learn the Darcy flow PDE solution operator", "voxel_grid", 64**3, target_metric="accuracy"
    )
    assert result["inferred_tasks"] == ["pde_operator"]
    assert result["recommendations"][0]["model"] == "fno3d"

    graph = advisor.recommend_architecture(
        "transient CFD surrogate rollout with remeshing", "mesh", 2_000_000, dynamic_topology=True
    )
    names = [row["model"] for row in graph["recommendations"]]
    assert "unet3d" not in names[:1]
    mgn = next(row for row in graph["recommendations"] if row["model"] == "meshgraphnet")
    assert any("VRAM saturation" in pitfall for pitfall in mgn["pitfalls"])

    assert result["verdict"] == "fit"


@pytest.mark.parametrize(("query", "capability"), [
    ("scene flow between two lidar frames", "scene_flow"),
    ("generate new shapes with diffusion", "shape_generation"),
    ("instance segmentation of chairs in an office scan", "instance_segmentation"),
    ("rigid registration of two scans", "registration"),
])
def test_uncovered_task_returns_no_fit(advisor, query, capability):
    result = advisor.recommend_architecture(query, "point_cloud", 4096)
    assert result["verdict"] == "no_fit" and result["missing_capability"] == capability
    assert result["suggested_networks"] and all(n["reference"].startswith("https://") for n in result["suggested_networks"])


def test_part_segmentation_query_fits(advisor):
    query = "3D object semantic part segmentation on unordered point clouds to predict point-wise discrete category labels"
    result = advisor.recommend_architecture(query, "point_cloud", 2048)
    assert result["verdict"] == "fit"
    assert result["recommendations"][0]["matched_tasks"] == ["segmentation_classification"]


def test_parse_markdown_doc():
    parsed = parse_markdown_doc(textwrap.dedent("""\
        # Toy Net

        Toy Net mixes points with a toy operator.
        Second line of the summary.

        ## Input contract

        - `coordinates`: `[B, N, 3]`
        - output: `[B, N, 1]`

        ## References

        - [Doe et al., Toy Nets](https://arxiv.org/abs/0000.00000)
        """))
    assert parsed["title"] == "Toy Net"
    assert parsed["summary"] == "Toy Net mixes points with a toy operator. Second line of the summary."
    assert parsed["sections"] == ["Input contract", "References"]
    assert parsed["tensor_contracts"] == ["`coordinates`: `[B, N, 3]`", "output: `[B, N, 1]`"]
    assert parsed["references"] == [{"title": "Doe et al., Toy Nets", "url": "https://arxiv.org/abs/0000.00000"}]


def test_new_registered_model_is_discovered_with_fallback(tmp_path, monkeypatch):
    (tmp_path / "toy_knn_net.py").write_text(textwrap.dedent('''\
        from dataclasses import dataclass

        from torch import nn

        from models import BaseModelConfig, BaseNetwork, ModelRegistry


        @dataclass
        class ToyKnnConfig(BaseModelConfig):
            name: str = "toy_knn_net"
            representation: str = "POINT_CLOUD"
            hidden_dim: int = 32
            num_layers: int = 2
            task: str = "segmentation"


        class ToyKnnNet(BaseNetwork):
            """Toy network aggregating k-NN neighbourhoods."""

            config_class = ToyKnnConfig

            def __init__(self, config=None):
                super().__init__(config=config)
                self.layer = nn.Linear(3, 1)

            @property
            def required_inputs(self):
                return {"coordinates"}

            def forward(self, inputs):
                raise NotImplementedError


        ModelRegistry.register("toy_knn_net", ToyKnnNet, ToyKnnConfig)
        '''))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "toy_knn_net.md").write_text("# Toy kNN Net\n\nInput `[B, N, 3]` points.\n")

    advisor = Advisor(docs_dir=docs, extra_modules=["toy_knn_net"])
    advisor.refresh()
    assert "toy_knn_net" not in {row["name"] for row in advisor.list_supported_models()}

    # Simulate the developer adding the model while the server is running.
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    report = advisor.refresh()
    assert "toy_knn_net" in report["added_since_last_refresh"]
    assert report["uncurated_models"] == ["toy_knn_net"]

    rows = advisor.list_supported_models(representation="point_cloud", task="segmentation_classification")
    assert "toy_knn_net" in {row["name"] for row in rows}

    dossier = advisor.get_model_dossier("toy_knn_net")
    assert dossier["status"] == "uncurated"
    assert dossier["implementation"]["parameter_count_default"] == 4
    assert dossier["tensor_contract"]["from_docs"] == ["Input `[B, N, 3]` points."]
    assert dossier["complexity"]["time"].startswith("O(N^2)")

    estimate = advisor.estimate_resource_footprint("toy_knn_net", 1000)
    assert any("No catalog entry" in warning for warning in estimate["warnings"])


def test_catalog_entries_are_valid():
    advisor = Advisor(catalog_path=CATALOG_PATH)
    for name, entry in advisor._load_catalog().items():
        assert set(entry["representations"]) <= set(REPRESENTATIONS), name
        assert entry["memory_model"]["per_element"] > 0, name
        assert 1 <= entry["speed"] <= 5 and 1 <= entry["expressivity"] <= 5, name


def test_mcp_server_exposes_tools():
    pytest.importorskip("mcp")
    from tools.mcp_advisor import server

    tools = asyncio.run(server.server.list_tools())
    assert {tool.name for tool in tools} == {
        "list_supported_models",
        "get_model_dossier",
        "recommend_architecture",
        "estimate_resource_footprint",
        "refresh_model_catalog",
        "audit_problem_assumptions",
        "check_symmetry_and_invariance",
        "recommend_representation_pipeline",
        "calculate_effective_receptive_field",
        "recommend_batching_strategy",
        "estimate_training_budget",
        "generate_benchmark_config",
        "suggest_ablation_matrix",
    }
    result = asyncio.run(server.server.call_tool("get_model_dossier", {"model_name": "siren"}))
    assert "SIREN" in str(result)


def test_shape_completion_recommends_completion_models(advisor):
    result = advisor.recommend_architecture("point cloud completion from partial scans", "point_cloud", 4096)
    assert result["verdict"] == "fit" and "shape_completion" in result["inferred_tasks"]
    assert {row["model"] for row in result["recommendations"][:2]} == {"pcn", "foldingnet"}


def test_object_detection_recommends_detectors(advisor):
    result = advisor.recommend_architecture("3D object detection of cars in lidar scans", "point_cloud", 20000)
    assert result["verdict"] == "fit" and "object_detection" in result["inferred_tasks"]
    assert result["recommendations"][0]["model"] == "pointpillars"
    assert {row["model"] for row in result["recommendations"][:2]} == {"pointpillars", "votenet"}
