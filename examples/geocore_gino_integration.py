"""End-to-end geocore -> GINO source-to-query smoke test.

Run from the workspace root:

    .venv/bin/python neural3dbench/examples/geocore_gino_integration.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
NEURAL3DBENCH_ROOT = WORKSPACE_ROOT / "neural3dbench"
GEOCORE_ROOT = WORKSPACE_ROOT / "geometry-py" / "geocore"
for source_root in (NEURAL3DBENCH_ROOT, GEOCORE_ROOT):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import geocore as gc  # noqa: E402
from models import GINOConfig, PointCloudRepresentationAdapter, create_model  # noqa: E402


METADATA_KEYS = (
    "schema_version",
    "representation",
    "coordinate_system",
    "units",
    "normalization_id",
)


def metadata_from(raw: dict) -> dict:
    metadata = {}
    for key in METADATA_KEYS:
        value = np.asarray(raw[key])
        metadata[key] = value.item() if value.ndim == 0 else value.tolist()
    return metadata


def make_geometry(seed: int):
    rng = np.random.default_rng(seed)
    if seed % 2:
        sizes = rng.uniform(1.2, 1.7, 3)
        return gc.Box((0.0, 0.0, 0.0), tuple(sizes), rounding=0.08)
    return gc.Sphere((0.0, 0.0, 0.0), float(rng.uniform(0.65, 0.85)))


def gaussian_field(points: np.ndarray, center: np.ndarray, sharpness: float) -> np.ndarray:
    offset = points - center
    return np.exp(-sharpness * np.sum(offset * offset, axis=-1)).astype(np.float32)


def create_sample(
    source_count: int = 48,
    query_count: int = 32,
    seed: int = 0,
) -> tuple[dict, np.ndarray]:
    """Build one geocore point-to-query sample and its analytic target field."""
    rng = np.random.default_rng(seed)
    geometry = gc.Geometry(make_geometry(seed), resolution=0.15)
    physical_queries = geometry.sample_volume(query_count, rng=seed + 1)
    cloud = dict(
        geometry.output(
            "POINT_CLOUD",
            n_points=source_count,
            mode="volume",
            features=None,
            normalization="unit_box_01",
            queries=physical_queries,
            rng=seed,
        )
    )

    center = rng.uniform(0.2, 0.8, 3)
    sharpness = float(rng.uniform(10.0, 24.0))
    source_values = gaussian_field(cloud["coordinates"][0], center, sharpness)
    target_values = gaussian_field(cloud["query_coordinates"][0], center, sharpness)
    cloud["point_features"] = source_values[None, :, None]
    cloud["point_feature_names"] = np.asarray(["source_value"], dtype=np.str_)
    gc.validate(cloud)
    assert cloud["point_features"].shape[-1] == 1
    return cloud, target_values[None, :, None]


def adapt_cloud(raw: dict):
    return PointCloudRepresentationAdapter().adapt(
        {**raw, "metadata": metadata_from(raw)}
    )


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cloud, _ = create_sample()
    cloud_path = Path(gc.save(str(output_dir / "point_cloud.npz"), cloud))

    raw = gc.load(str(cloud_path))
    gc.validate(raw)
    model_input = adapt_cloud(raw)
    assert model_input.representation == "POINT_CLOUD"
    assert tuple(model_input.coordinates.shape) == (1, 48, 3)
    assert tuple(model_input.point_features.shape) == (1, 48, 1)
    assert tuple(model_input.query_coordinates.shape) == (1, 32, 3)

    torch.manual_seed(0)
    config = GINOConfig(
        input_dim=1,
        output_dim=1,
        latent_resolution=(6, 6, 6),
        width=8,
        modes=(2, 2, 2),
        num_layers=2,
        kernel_radius=0.6,
        kernel_hidden_dim=16,
        kernel_chunk_size=128,
        kernel_max_pairs=8192,
    )
    model = create_model(config)
    output = model(model_input)
    assert tuple(output.predictions.shape) == (1, 32, 1)
    assert torch.isfinite(output.predictions).all()
    output.predictions.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    print("PASS: geocore and GINO work together")
    print(f"  interchange file: {cloud_path}")
    print(f"  source coordinates: {tuple(model_input.coordinates.shape)}")
    print(f"  source features: {tuple(model_input.point_features.shape)}")
    print(f"  query coordinates: {tuple(model_input.query_coordinates.shape)}")
    print(f"  predictions: {tuple(output.predictions.shape)}")
    print("  forward and backward passes: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        run(args.output_dir.resolve())
        return
    with tempfile.TemporaryDirectory(prefix="geocore_gino_") as temp_dir:
        run(Path(temp_dir))


if __name__ == "__main__":
    main()