"""End-to-end geocore ``VOXEL_GRID`` -> internal ``GRID`` -> FNO3D check.

Run from the workspace root:

    .venv/bin/python neural3dbench/examples/geocore/geocore_fno3d_integration.py

The randomly initialized model verifies the saved interchange contract,
adapter, FNO3D forward/backward path, and checkpoint round trip.
"""

from __future__ import annotations

import argparse
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
from models import FNO3DConfig, GridRepresentationAdapter, create_model  # noqa: E402


METADATA_KEYS = (
    "schema_version",
    "representation",
    "coordinate_system",
    "units",
    "normalization_id",
    "channel_names",
    "origin",
    "spacing",
    "bounds",
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


def adapt_voxel_grid(raw: dict):
    """Translate canonical geocore ``VOXEL_GRID`` to neural3dbench ``GRID``."""
    if str(raw["representation"]) != "VOXEL_GRID":
        raise ValueError("Expected a geocore VOXEL_GRID payload.")
    return GridRepresentationAdapter().adapt(
        {**raw, "metadata": metadata_from(raw)}
    )


def create_problem(resolution: int = 12) -> dict:
    geometry = gc.Box((0.0, 0.0, 0.0), (1.3, 0.9, 0.7), rounding=0.12)

    def source(points: np.ndarray) -> np.ndarray:
        offset = points - np.asarray((-0.25, 0.0, 0.0))
        return np.exp(-18.0 * np.sum(offset * offset, axis=1))

    grid = dict(
        gc.voxel_grid(
            geometry,
            shape=resolution,
            bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
            channels=("occupancy", "sdf", "boundary", "source"),
            fields={"source": source},
            normalization="unit_cube",
        )
    )
    gc.validate(grid, input_channels=4)
    return grid


def run(output_dir: Path, resolution: int = 12) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    problem = create_problem(resolution)
    npz_path = Path(gc.save(str(output_dir / "voxel_grid.npz"), problem))

    # Deliberately adapt the reloaded file, not the in-memory generation result.
    raw = gc.load(str(npz_path))
    gc.validate(raw, input_channels=4)
    model_input = adapt_voxel_grid(raw)
    assert model_input.representation == "GRID"

    config = FNO3DConfig(
        input_channels=4,
        output_channels=1,
        width=4,
        num_layers=2,
        modes=(2, 2, 2),
        projection_channels=8,
    )
    torch.manual_seed(0)
    model = create_model(config).cpu()
    output = model(model_input)
    assert output.predictions.shape == (1, 1, resolution, resolution, resolution)
    assert torch.isfinite(output.predictions).all()
    output.predictions.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    checkpoint_path = output_dir / "fno3d_smoke.pt"
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

    print("PASS: geocore VOXEL_GRID and FNO3D work together")
    print(f"  interchange file: {npz_path}")
    print(f"  external / internal representation: VOXEL_GRID / {model_input.representation}")
    print(f"  voxel shape: {tuple(model_input.voxel_fields.shape)}")
    print(f"  channels: {metadata_from(raw)['channel_names']}")
    print(f"  checkpoint round trip: {checkpoint_path}")
    print("  CPU forward and backward passes: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep artifacts here (default: temporary directory).",
    )
    args = parser.parse_args()
    if args.resolution < 4:
        parser.error("resolution must be at least 4")
    if args.output_dir is not None:
        run(args.output_dir.resolve(), args.resolution)
        return
    with tempfile.TemporaryDirectory(prefix="geocore_fno3d_") as temp_dir:
        run(Path(temp_dir), args.resolution)


if __name__ == "__main__":
    main()
