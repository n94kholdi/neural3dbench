from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = WORKSPACE_ROOT / "neural3dbench" / "examples" / "geocore"
GEOCORE_ROOT = WORKSPACE_ROOT / "geometry-py" / "geocore"
for source_root in (EXAMPLES_ROOT, GEOCORE_ROOT):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import geocore as gc  # noqa: E402
from geocore_pointnet_integration import run  # noqa: E402
from train_pointnet_geocore_shapes import adapt_cloud, create_cloud, make_geometries  # noqa: E402


def test_saved_geocore_cloud_runs_through_pointnet(tmp_path):
    run(tmp_path)
    raw = gc.load(str(tmp_path / "point_cloud.npz"))
    assert str(raw["representation"]) == "POINT_CLOUD"
    assert raw["coordinates"].shape == (1, 96, 3)
    assert raw["point_features"].shape == (1, 96, 3)


def test_training_cloud_has_canonical_coordinate_only_contract():
    cloud = create_cloud(make_geometries(resolution=0.2)[1], 48, seed=4)
    model_input = adapt_cloud(cloud)
    assert model_input.representation == "POINT_CLOUD"
    assert tuple(model_input.coordinates.shape) == (1, 48, 3)
    assert model_input.point_features is None
    assert np.isfinite(model_input.coordinates.numpy()).all()
    assert model_input.metadata["normalization_id"].startswith("centroid_unit_sphere:")
