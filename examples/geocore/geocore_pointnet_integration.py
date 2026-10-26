"""End-to-end geocore -> ModelInput -> PointNet smoke test.

Run from any directory with:

    python neural3dbench/examples/geocore/geocore_pointnet_integration.py

The model is randomly initialized, so this example verifies the serialized
point-cloud contract, adapter, inference, and gradients rather than accuracy.
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
from models import (  # noqa: E402
    PointCloudRepresentationAdapter,
    PointNetConfig,
    create_model,
)


METADATA_KEYS = (
    "schema_version",
    "representation",
    "coordinate_system",
    "units",
    "normalization_id",
    "point_feature_names",
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


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use optional surface normals to exercise both PointNet input tensors.
    geometry = gc.Box((0.0, 0.0, 0.0), (2.0, 1.5, 1.0), rounding=0.1)
    cloud = geometry.output(
        "POINT_CLOUD",
        n_points=96,
        mode="surface",
        features=("nx", "ny", "nz"),
        normalization="unit_sphere",
        resolution=0.12,
        rng=0,
    )
    assert cloud.validate(input_dim=6) == "POINT_CLOUD"
    npz_path = Path(cloud.save(str(output_dir) + "/"))

    # Reload the interchange file so the test cannot accidentally depend on
    # behavior of the in-memory geocore result.
    raw = gc.load(str(npz_path))
    gc.validate(raw, input_dim=6)
    model_input = PointCloudRepresentationAdapter().adapt(
        {**raw, "metadata": metadata_from(raw)}
    )

    config = PointNetConfig(
        input_dim=6,
        output_dim=3,
        task="classification",
        global_dim=64,
        dropout=0.0,
    )
    torch.manual_seed(0)
    model = create_model(config).eval()
    output = model(model_input)

    assert output.predictions.shape == (1, 3)
    assert output.latent.shape == (1, config.global_dim)
    assert torch.isfinite(output.predictions).all()
    output.predictions.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    print("PASS: geocore and PointNet work together")
    print(f"  interchange file: {npz_path}")
    print(f"  point cloud shape: {tuple(model_input.coordinates.shape)}")
    print(f"  point features: {metadata_from(raw)['point_feature_names']}")
    print(f"  prediction shape: {tuple(output.predictions.shape)}")
    print("  forward and backward passes: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep point_cloud.npz in this directory (default: temporary directory).",
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        run(args.output_dir.resolve())
        return
    with tempfile.TemporaryDirectory(prefix="geocore_pointnet_") as temp_dir:
        run(Path(temp_dir))


if __name__ == "__main__":
    main()
