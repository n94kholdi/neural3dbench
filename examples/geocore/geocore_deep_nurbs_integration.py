"""End-to-end geocore PARAMETRIC_SURFACE -> Deep NURBS smoke test.

Run from the workspace root:

    .venv/bin/python neural3dbench/examples/geocore/geocore_deep_nurbs_integration.py

The check saves and reloads the interchange file, constructs both the model
configuration and runtime input from it, and verifies CPU gradients and a
checkpoint round trip.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
NEURAL3DBENCH_ROOT = WORKSPACE_ROOT / "neural3dbench"
GEOCORE_ROOT = WORKSPACE_ROOT / "geometry-py" / "geocore"
for source_root in (NEURAL3DBENCH_ROOT, GEOCORE_ROOT):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import geocore as gc  # noqa: E402
from models import (  # noqa: E402
    DeepNURBSConfig,
    ParametricSurfaceRepresentationAdapter,
    create_model,
)


METADATA_KEYS = (
    "schema_version",
    "representation",
    "coordinate_system",
    "units",
    "normalization_id",
)


def scalar_or_list(value: np.ndarray):
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def metadata_from(raw: dict) -> dict:
    return {
        key: scalar_or_list(raw[key])
        for key in METADATA_KEYS
        if key in raw
    }


def create_surface() -> gc.NURBSSurface:
    """Create a curved, non-folding quarter-annulus PDE domain."""
    return gc.NURBSSurface.annulus(
        0.65, 1.35, theta0=0.0, theta1=0.5 * math.pi, degree_r=2
    ).refine(5, 4)


def create_problem(
    n_points: int = 128,
    seed: int = 0,
    *,
    method: str = "sobol",
    include_boundary: int = 16,
) -> dict:
    problem = dict(
        gc.parametric_surface(
            create_surface(),
            output_dim=1,
            dirichlet=0.0,
            n_points=n_points,
            method=method,
            include_boundary=include_boundary,
            units="normalized",
            rng=seed,
        )
    )
    gc.validate(problem)
    return problem


def adapt_parametric_surface(raw: dict, **network_overrides):
    """Build Deep NURBS configuration and ModelInput from one payload."""
    if str(raw["representation"]) != "PARAMETRIC_SURFACE":
        raise ValueError("Expected a geocore PARAMETRIC_SURFACE payload.")
    metadata = metadata_from(raw)
    network_options = {
        "hidden_dim": 24,
        "num_hidden_layers": 2,
        "activation": "tanh",
        "metadata": metadata,
    }
    network_options.update(network_overrides)
    config = DeepNURBSConfig.from_parametric_surface(raw, **network_options)
    model_input = ParametricSurfaceRepresentationAdapter().adapt(
        {**raw, "metadata": metadata}
    )
    return config, model_input


def run(output_dir: Path, n_points: int = 128) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    problem = create_problem(n_points=n_points)
    npz_path = Path(gc.save(str(output_dir / "parametric_surface.npz"), problem))

    raw = gc.load(str(npz_path))
    gc.validate(raw)
    config, model_input = adapt_parametric_surface(raw)

    torch.manual_seed(0)
    model = create_model(config).cpu()
    output = model(model_input)
    expected_points = n_points + 16
    assert output.predictions.shape == (1, expected_points, 1)
    assert torch.isfinite(output.predictions).all()
    output.predictions.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    checkpoint_path = output_dir / "deep_nurbs_smoke.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict()},
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = create_model(checkpoint["config"]).cpu().eval()
    restored.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        expected = model(model_input).predictions
        actual = restored(model_input).predictions
    torch.testing.assert_close(actual, expected)

    physical = output.metadata["physical_coordinates"].detach().numpy()
    np.testing.assert_allclose(
        physical, raw["physical_coordinates"], rtol=2e-5, atol=2e-6
    )
    print("PASS: geocore PARAMETRIC_SURFACE and Deep NURBS work together")
    print(f"  interchange file: {npz_path}")
    print(f"  UV / physical shape: {tuple(model_input.coordinates.shape)} / {physical.shape}")
    print(f"  control net: {tuple(raw['control_points'].shape[:2])}")
    print(f"  checkpoint round trip: {checkpoint_path}")
    print("  CPU forward and backward passes: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep artifacts here (default: temporary directory).",
    )
    args = parser.parse_args()
    if args.num_points < 4:
        parser.error("num-points must be at least 4")
    if args.output_dir is not None:
        run(args.output_dir.resolve(), args.num_points)
        return
    with tempfile.TemporaryDirectory(prefix="geocore_deep_nurbs_") as temp_dir:
        run(Path(temp_dir), args.num_points)


if __name__ == "__main__":
    main()
