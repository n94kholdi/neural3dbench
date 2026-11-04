import pytest

pytest.importorskip("yaml")

import yaml  # noqa: E402

from tools.mcp_advisor import guardrails as g  # noqa: E402
from tools.mcp_advisor.advisor import Advisor  # noqa: E402


@pytest.fixture(scope="module")
def advisor():
    advisor = Advisor()
    advisor.refresh()
    return advisor


def test_catalog_entries_carry_guardrail_fields(advisor):
    for name, model in advisor.state["models"].items():
        meta = model["metadata"]
        assert set(meta["symmetries"]) <= set(g.SYMMETRIES), name
        assert meta["receptive_field"]["kind"] in ("global", "hops", "conv_grid", "pointwise"), name
        assert meta["resolution_transfer"] in (True, False, "partial"), name
        if key := meta["receptive_field"].get("depth_key"):
            assert key in model["registry"]["config_defaults"], name


def test_audit_flags_topology_resolution_and_bcs(advisor):
    result = g.audit_problem_assumptions(
        advisor, "CFD with remeshing", topology_changes=True, evaluate_at_unseen_resolution=True
    )
    assert "siren" not in result["models_without_errors"]
    assert "gcn" not in result["models_without_errors"]
    assert "transolver" in result["models_without_errors"]

    nurbs = g.audit_problem_assumptions(advisor, "heat", "deep_nurbs", boundary_conditions="dirichlet")
    assert nurbs["audits"]["deep_nurbs"][0]["severity"] == "ok"
    soft = g.audit_problem_assumptions(advisor, "heat", "fno3d", boundary_conditions="dirichlet")
    checks = {f["check"] for f in soft["audits"]["fno3d"]}
    assert {"boundary_conditions", "periodicity"} <= checks
    with pytest.raises(ValueError):
        g.audit_problem_assumptions(advisor, "x", boundary_conditions="robin")


def test_symmetry_vector_targets_need_equivariance(advisor):
    result = g.check_symmetry_and_invariance(advisor, "meshgraphnet", ["rotation", "translation"], "vector")
    verdicts = {v["symmetry"]: v for v in result["verdicts"]}
    assert not verdicts["rotation"]["guaranteed_by_architecture"]
    assert "equivariant" in verdicts["rotation"]["action"]
    assert not verdicts["translation"]["guaranteed_by_architecture"]  # absolute coordinates

    relative = g.check_symmetry_and_invariance(
        advisor, "meshgraphnet", ["translation", "permutation"], uses_absolute_coordinates=False
    )
    assert relative["all_guaranteed"]
    fno = g.check_symmetry_and_invariance(advisor, "fno3d", ["translation"], uses_absolute_coordinates=False)
    assert "append_coordinates" in fno["verdicts"][0]["action"]


def test_pipeline_names_real_converters_and_warns_on_voxelisation(advisor):
    voxel = g.recommend_representation_pipeline(advisor, ".stl", "unet3d")
    assert voxel["target_representation"] == "voxel_grid"
    assert any("volumetric.mesh_to_voxel_grid" in step for step in voxel["steps"])
    assert any("watertight" in w for w in voxel["warnings"])
    assert any("vertices only" in w for w in voxel["warnings"])

    graph = g.recommend_representation_pipeline(advisor, "step", "meshgraphnet")
    assert graph["source"] == "cad" and graph["target_representation"] == "mesh_graph"
    assert any("no CAD reader" in step for step in graph["steps"])

    nurbs = g.recommend_representation_pipeline(advisor, "step", "deep_nurbs")
    assert any("DeepNURBSConfig" in step for step in nurbs["steps"])
    with pytest.raises(ValueError):
        g.recommend_representation_pipeline(advisor, "docx", "unet3d")


def test_receptive_field_warns_when_hops_are_short(advisor):
    short = g.calculate_effective_receptive_field(advisor, "gcn", element_spacing=0.01, interaction_length=1.0)
    assert short["depth"] == 2 and "warning" in short
    deep = g.calculate_effective_receptive_field(advisor, "meshgraphnet", 120, 0.01, 1.0)
    assert "ok" in deep
    unet = g.calculate_effective_receptive_field(advisor, "unet3d")
    assert unet["reach_elements"].startswith("96 voxels")  # k=3, 4 levels
    assert g.calculate_effective_receptive_field(advisor, "transolver")["kind"] == "global"


def test_batching_strategy(advisor):
    small = g.recommend_batching_strategy(advisor, "pointnet", 1024, 24)
    assert small["micro_batch"] == 16 and small["gradient_accumulation_steps"] == 1

    tight = g.recommend_batching_strategy(advisor, "meshgraphnet", 50_000, 24)
    assert 1 <= tight["micro_batch"] < 16
    assert tight["gradient_accumulation_steps"] * tight["micro_batch"] >= 16
    assert "edge_index" in tight["collation"]

    oom = g.recommend_batching_strategy(advisor, "dgcnn", 200_000, 8)
    assert oom["micro_batch"] == 0 and oom["gradient_accumulation_steps"] is None
    assert any("Sub-sample" in note for note in oom["notes"])


def test_training_budget_scales_with_epochs_and_gpu(advisor):
    a100 = g.estimate_training_budget(advisor, "transolver", 32_000, 1000, 100, "A100")
    twice = g.estimate_training_budget(advisor, "transolver", 32_000, 1000, 200, "a100")
    t4 = g.estimate_training_budget(advisor, "transolver", 32_000, 1000, 100, "T4")
    assert twice["total_hours"] == pytest.approx(2 * a100["total_hours"], rel=0.01)
    assert t4["total_hours"] > a100["total_hours"]
    with pytest.raises(ValueError):
        g.estimate_training_budget(advisor, "transolver", 1000, 10, 1, "TPU")


def test_benchmark_config_round_trips(advisor):
    result = g.generate_benchmark_config(advisor, "adaptive_meshgraphnet", "airfoil", 5000)
    config = yaml.safe_load(result["yaml"])
    assert config["model"]["name"] == "meshgraphnet"
    assert config["model"]["config"]["processor_steps"] == 15
    assert "edge_index" in config["data"]["required_inputs"]
    assert "ModelRegistry.create" in result["python"]
    compile(result["python"], "launcher.py", "exec")


def test_ablation_matrix(advisor):
    result = g.suggest_ablation_matrix(advisor, "transolver")
    names = [row["model"] for row in result["contenders"]]
    assert "transolver" not in names and len(names) == len(set(names)) >= 3
    assert "meshgraphnet" in names
    simplest = g.suggest_ablation_matrix(advisor, "gcn")
    assert all(row["role"] != "simplest baseline (same input)" for row in simplest["contenders"])
