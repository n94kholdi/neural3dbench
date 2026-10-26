"""Preview target, predicted, and error fields from the geocore GAT example."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402


def render(result_path: Path, output_path: Path) -> None:
    with np.load(result_path, allow_pickle=False) as archive:
        coordinates = archive["coordinates"]
        target = archive["target"]
        prediction = archive["prediction"]
    error = np.abs(prediction - target)
    triangulation = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1])
    fields = (
        ("Target field", target, "viridis", float(target.min()), float(target.max())),
        ("GAT prediction", prediction, "viridis", float(target.min()), float(target.max())),
        ("Absolute error", error, "magma", 0.0, max(float(error.max()), 1e-8)),
    )

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160, constrained_layout=True)
    for axis, (title, values, cmap, minimum, maximum) in zip(axes, fields):
        image = axis.tripcolor(
            triangulation,
            values,
            shading="gouraud",
            cmap=cmap,
            vmin=minimum,
            vmax=maximum,
        )
        axis.triplot(triangulation, color="white", linewidth=0.2, alpha=0.45)
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.set_xlabel("normalized x")
        axis.set_ylabel("normalized y")
        figure.colorbar(image, ax=axis, shrink=0.85)

    mse = float(np.mean((prediction - target) ** 2))
    mae = float(error.mean())
    figure.suptitle(f"Geocore MESH_GRAPH -> GAT | MSE {mse:.6f} | MAE {mae:.6f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gat_geocore_field"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    result_path = output_dir / "gat_field_result.npz"
    if not result_path.is_file():
        raise FileNotFoundError(f"Expected {result_path}; run training first.")
    preview_path = output_dir / "gat_geocore_field_preview.png"
    render(result_path, preview_path)
    print(f"GAT field preview: {preview_path}")


if __name__ == "__main__":
    main()