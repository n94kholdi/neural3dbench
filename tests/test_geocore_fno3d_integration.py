from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = WORKSPACE_ROOT / "neural3dbench" / "examples"
GEOCORE_ROOT = WORKSPACE_ROOT / "geometry-py" / "geocore"
for source_root in (EXAMPLES_ROOT, GEOCORE_ROOT):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import geocore as gc  # noqa: E402
from geocore_fno3d_integration import adapt_voxel_grid, run  # noqa: E402
from train_fno3d_geocore_heat import (  # noqa: E402
    CHANNEL_NAMES,
    create_problem,
    heat_response,
    make_dataset,
)


def test_saved_geocore_voxel_grid_runs_through_fno3d(tmp_path):
    run(tmp_path, resolution=8)
    raw = gc.load(str(tmp_path / "voxel_grid.npz"))
    assert str(raw["representation"]) == "VOXEL_GRID"
    assert raw["voxel_fields"].shape == (1, 4, 8, 8, 8)
    assert raw["channel_names"].tolist() == list(CHANNEL_NAMES)
    assert (tmp_path / "fno3d_smoke.pt").is_file()


def test_voxel_grid_adapter_preserves_contract_metadata_and_axis_order():
    raw = create_problem(resolution=10, seed=3)
    model_input = adapt_voxel_grid(raw)
    assert model_input.representation == "GRID"
    assert tuple(model_input.voxel_fields.shape) == (1, 4, 10, 10, 10)
    assert model_input.voxel_fields.dtype == torch.float32
    assert model_input.metadata["representation"] == "VOXEL_GRID"
    assert model_input.metadata["channel_names"] == list(CHANNEL_NAMES)
    extent = np.asarray(model_input.metadata["bounds"])[1] - np.asarray(
        model_input.metadata["bounds"]
    )[0]
    spacing = np.asarray(model_input.metadata["spacing"])
    np.testing.assert_allclose(extent, spacing * np.array([10, 10, 10]), rtol=1e-5)


def test_heat_dataset_is_finite_nontrivial_and_shape_aligned():
    inputs, targets = make_dataset(samples=3, resolution=8, seed=20)
    assert inputs.shape == (3, 4, 8, 8, 8)
    assert targets.shape == (3, 1, 8, 8, 8)
    assert torch.isfinite(inputs).all() and torch.isfinite(targets).all()
    assert torch.count_nonzero(inputs[:, 0]) > 0
    assert torch.count_nonzero(targets) > 0
    torch.testing.assert_close(targets, heat_response(inputs))
