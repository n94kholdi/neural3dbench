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
from geocore_deep_nurbs_integration import (  # noqa: E402
    adapt_parametric_surface,
    create_problem,
    run,
)
from models import create_model  # noqa: E402
from train_deep_nurbs_geocore_field import manufactured_target  # noqa: E402


def test_saved_geocore_parametric_surface_runs_through_deep_nurbs(tmp_path):
    run(tmp_path, n_points=16)
    raw = gc.load(str(tmp_path / "parametric_surface.npz"))
    assert str(raw["representation"]) == "PARAMETRIC_SURFACE"
    assert raw["parametric_coordinates"].shape == (1, 32, 2)
    assert (tmp_path / "deep_nurbs_smoke.pt").is_file()


def test_payload_builds_geometry_config_and_runtime_input():
    raw = create_problem(n_points=32, seed=4, include_boundary=8)
    config, model_input = adapt_parametric_surface(raw)
    assert config.degree == tuple(raw["degree"])
    assert config.output_dim == raw["lifting_coefficients"].shape[-1]
    assert model_input.representation == "PARAMETRIC_SURFACE"
    assert model_input.coordinates.shape == (1, 40, 2)
    assert model_input.metadata["representation"] == "PARAMETRIC_SURFACE"

    model = create_model(config).eval()
    with torch.no_grad():
        output = model(model_input)
    np.testing.assert_allclose(
        output.metadata["physical_coordinates"].numpy(),
        raw["physical_coordinates"],
        rtol=2e-5,
        atol=2e-6,
    )


def test_manufactured_field_and_model_ansatz_have_exact_boundary_trace():
    raw = create_problem(n_points=25, seed=2, method="grid", include_boundary=0)
    config, model_input = adapt_parametric_surface(raw)
    output = create_model(config).eval()(model_input)
    target = manufactured_target(
        output.metadata["physical_coordinates"], output.metadata["admissible_field"]
    )
    uv = model_input.coordinates
    boundary = ((uv == 0.0) | (uv == 1.0)).any(dim=-1)
    torch.testing.assert_close(target[boundary], torch.zeros_like(target[boundary]))
    torch.testing.assert_close(
        output.predictions[boundary], torch.zeros_like(output.predictions[boundary])
    )
    assert torch.count_nonzero(target[~boundary]) > 0
