"""Task/dataset contract: synthetic labelled data, head sizing, loss, metrics and previews.

Run with ``pytest tools/dashboard/test_tasks.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models import ModelRegistry  # noqa: E402
from tools.dashboard import tasks  # noqa: E402


def test_chair_parts_labels_every_point() -> None:
    coords, labels = tasks.make_dataset("chair_parts", 6, 512)
    assert coords.shape == (6, 512, 3) and labels.shape == (6, 512) and labels.dtype == torch.long
    assert set(labels.unique().tolist()) <= set(range(len(tasks.CHAIR_PARTS)))
    assert {0, 1, 2} <= set(labels.unique().tolist())  # every chair has a seat, a back and legs
    assert coords.norm(dim=-1).max() <= 1.0 + 1e-5
    assert torch.equal(coords, tasks.make_dataset("chair_parts", 6, 512)[0])  # seeded


def test_primitive_shapes_labels() -> None:
    coords, labels = tasks.make_dataset("primitive_shapes", 6, 64)
    assert coords.shape == (6, 64, 3) and labels.tolist() == [0, 1, 2, 0, 1, 2]


def test_choose_dataset_follows_output_and_model() -> None:
    assert tasks.choose_dataset("pointnet", "point_label")[0] == "chair_parts"
    assert tasks.choose_dataset("pointnet", "shape_label")[0] == "primitive_shapes"
    assert tasks.choose_dataset("transolver", "point_field", "pde_operator")[0] == "poisson"
    assert tasks.choose_dataset("siren", "point_field", "field_fitting")[0] == "deformed_sphere"
    name, reason = tasks.choose_dataset("dgcnn", "shape_label", benchmarks=["ModelNet40"])
    assert name == "primitive_shapes" and "synthetic stand-in" in reason and "modelnet40" in reason
    # Mesh models only get mesh datasets; classification-only mesh models get nothing per point.
    assert set(tasks.compatible_datasets("meshnet", "shape_label")) == {"modelnet10_mesh", "shrec11"}
    assert tasks.compatible_datasets("meshnet", "point_label") == []


@pytest.mark.parametrize("model_name", ["pointnet", "pointnet2", "dgcnn", "pointnext", "mlp"])
def test_heads_match_task_shapes(model_name: str) -> None:
    coords = tasks.make_dataset("chair_parts", 2, 64)[0]
    for dataset, shape in (("chair_parts", (2, 64, 4)), ("primitive_shapes", (2, 3)), ("deformed_sphere", (2, 64, 1))):
        ds = tasks.DATASETS[dataset]
        config = tasks.head_config(model_name, ds)
        if model_name == "pointnext":
            config["sample_counts"] = (32, 16, 8, 4)
        model = ModelRegistry.create(model_name, config).eval()
        with torch.no_grad():
            assert tasks.forward(ds.task, model, coords).shape == shape, (model_name, dataset)



def test_input_mismatch() -> None:
    for name in ("pointnet", "dgcnn", "mlp", "transolver"):
        assert tasks.input_mismatch(name) is None
    for name in ("gcn", "unet3d", "gino", "deep_nurbs"):
        assert tasks.input_mismatch(name)
    assert "voxel_fields" in tasks.input_mismatch("fno3d")
    assert "detector" in tasks.input_mismatch("votenet", "chair_parts")
    assert "poses" in tasks.input_mismatch("dcp", "chair_parts")


def test_loss_and_metrics() -> None:
    target = torch.tensor([[0, 0, 1, 2]])
    perfect = torch.nn.functional.one_hot(target, 3).float() * 10
    assert tasks.metrics(tasks.PART_SEGMENTATION, perfect, target, 3) == {"accuracy": 1.0, "miou": 1.0}
    assert tasks.loss(tasks.PART_SEGMENTATION, perfect, target) < 1e-3
    # predict everything as class 0: IoU 0.5 for class 0, 0 for classes 1 and 2
    assert tasks.mean_iou(torch.zeros(4, dtype=torch.long), target[0], 3) == pytest.approx(0.5 / 3)
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert tasks.metrics(tasks.SHAPE_CLASSIFICATION, logits, torch.tensor([0, 2])) == {"accuracy": 0.5}


def test_evaluation_figures() -> None:
    pytest.importorskip("plotly")
    coords, labels = tasks.make_dataset("chair_parts", 1, 128)
    logits = torch.nn.functional.one_hot(labels[0], 4).float()
    figure = tasks.evaluation_figure(tasks.DATASETS["chair_parts"], coords[0], labels[0], logits)
    assert "accuracy 100.0%" in figure.layout.title.text
    assert {t.name for t in figure.data} >= set(tasks.CHAIR_PARTS) | {"misclassified"}
    figure = tasks.evaluation_figure(tasks.DATASETS["primitive_shapes"], coords[0], torch.tensor(1), torch.tensor([0.0, 3.0, 0.0]))
    assert "True: cube" in figure.layout.title.text and "correct" in figure.layout.title.text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_partial_shapes_and_chamfer() -> None:
    partial, full = tasks.make_dataset("partial_shapes", 4, 128)
    assert partial.shape == full.shape == (4, 128, 3)
    assert tasks.chamfer_distance(full, full).item() == pytest.approx(0.0, abs=1e-5)
    assert tasks.chamfer_distance(partial, full).item() > 0.01  # a side is missing
    pred = torch.zeros(4, 10, 3, requires_grad=True)
    tasks.chamfer_distance(pred, full).backward()
    assert torch.isfinite(pred.grad).all()
    assert tasks.choose_dataset("pcn", "point_set")[0] == "partial_shapes"
    assert tasks.choose_dataset("pointpillars", "boxes", benchmarks=["KITTI"]) [0] is None
    assert "KITTI" in tasks.choose_dataset("pointpillars", "boxes", benchmarks=["KITTI"])[1]
    assert tasks.choose_dataset("dcp", "rigid_pose")[0] is None


def test_completion_mismatch() -> None:
    assert tasks.input_mismatch("pcn", "partial_shapes") is None
    assert tasks.input_mismatch("foldingnet", "chair_parts")
    assert tasks.input_mismatch("pointnet", "partial_shapes")
    assert tasks.input_mismatch("pointnet", "chair_parts") is None
