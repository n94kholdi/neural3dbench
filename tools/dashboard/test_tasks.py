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


def test_dataset_for_problem() -> None:
    query = "3D object semantic part segmentation on unordered point clouds to predict point-wise labels"
    assert tasks.dataset_for_problem("segmentation_classification", query) == "chair_parts"
    assert tasks.dataset_for_problem("segmentation_classification", "shape classification") == "primitive_shapes"
    assert tasks.dataset_for_problem("pde_operator") == "poisson"
    assert tasks.dataset_for_problem("field_fitting") == "deformed_sphere"


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
