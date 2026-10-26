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
from geocore_gino_integration import create_sample, run  # noqa: E402


def test_geocore_gino_source_to_query_round_trip(tmp_path):
    run(tmp_path)
    raw = gc.load(str(tmp_path / "point_cloud.npz"))
    assert str(raw["representation"]) == "POINT_CLOUD"
    assert raw["coordinates"].shape == (1, 48, 3)
    assert raw["point_features"].shape == (1, 48, 1)
    assert raw["query_coordinates"].shape == (1, 32, 3)
    assert np.all((raw["coordinates"] >= 0.0) & (raw["coordinates"] <= 1.0))
    assert np.all((raw["query_coordinates"] >= 0.0) & (raw["query_coordinates"] <= 1.0))


def test_geocore_sample_has_matching_finite_source_and_query_fields():
    cloud, target = create_sample(source_count=20, query_count=13, seed=5)
    assert cloud["point_features"].shape == (1, 20, 1)
    assert cloud["query_coordinates"].shape == (1, 13, 3)
    assert target.shape == (1, 13, 1)
    assert np.isfinite(cloud["point_features"]).all()
    assert np.isfinite(target).all()