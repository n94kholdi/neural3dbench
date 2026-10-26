"""Render the canonical geocore voxel grid and FNO3D validation slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def colorize(values: np.ndarray, lower: float | None = None, upper: float | None = None) -> Image.Image:
    values = np.asarray(values, dtype=np.float32)
    lower = float(values.min()) if lower is None else lower
    upper = float(values.max()) if upper is None else upper
    scaled = np.clip((values - lower) / max(upper - lower, 1e-8), 0.0, 1.0)
    # A compact blue -> cyan -> yellow heat map without a plotting dependency.
    red = np.clip(2.2 * scaled - 0.35, 0.0, 1.0)
    green = np.clip(2.0 * scaled, 0.0, 1.0) * np.clip(2.0 - 1.2 * scaled, 0.0, 1.0)
    blue = np.clip(1.25 - 1.7 * scaled, 0.0, 1.0)
    rgb = (255.0 * np.stack((red, green, blue), axis=-1)).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def middle_planes(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth, height, width = volume.shape
    return (
        volume[depth // 2],
        volume[:, height // 2, :],
        volume[:, :, width // 2],
    )


def render_rows(rows: list[tuple[str, np.ndarray]], path: Path, title: str) -> None:
    tile = 250
    left = 235
    top = 105
    gap = 18
    width = left + 3 * tile + 4 * gap
    height = top + len(rows) * (tile + gap) + 25
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 28), title, anchor="ma", font=font(23, bold=True), fill="#101828")
    for column, label in enumerate(("XY (middle Z)", "XZ (middle Y)", "YZ (middle X)")):
        draw.text((left + gap + column * (tile + gap) + tile // 2, 76), label, anchor="mm", font=font(16, bold=True), fill="#475467")

    for row_index, (label, volume) in enumerate(rows):
        y = top + row_index * (tile + gap)
        draw.text((left - 15, y + tile // 2), label, anchor="rm", font=font(17, bold=True), fill="#344054")
        lower, upper = float(volume.min()), float(volume.max())
        for column, plane in enumerate(middle_planes(volume)):
            image = colorize(plane, lower, upper).resize((tile, tile), Image.Resampling.NEAREST)
            x = left + gap + column * (tile + gap)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + tile, y + tile), outline="#98a2b3", width=2)
    canvas.save(path)


def render_canonical(grid: dict[str, np.ndarray], path: Path) -> None:
    names = grid["channel_names"].tolist()
    fields = grid["voxel_fields"][0]
    render_rows(
        [(str(name), fields[index]) for index, name in enumerate(names)],
        path,
        "Canonical geocore VOXEL_GRID  •  D,H,W = Z,Y,X",
    )


def render_validation(result: dict[str, np.ndarray], path: Path) -> None:
    inputs = result["inputs"][0]
    target = result["targets"][0, 0]
    prediction = result["predictions"][0, 0]
    error = np.abs(prediction - target)
    mse = float(result["validation_mse"])
    mae = float(result["validation_mae"])
    render_rows(
        [
            ("occupancy", inputs[0]),
            ("heat source", inputs[3]),
            ("target", target),
            ("FNO3D", prediction),
            ("absolute error", error),
        ],
        path,
        f"geocore VOXEL_GRID → FNO3D heat  •  MSE {mse:.5f}  •  MAE {mae:.5f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fno3d_geocore_heat"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    grid_path = output_dir / "voxel_grid.npz"
    result_path = output_dir / "validation_result.npz"
    if not grid_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(
            f"Expected {grid_path} and {result_path}; run training first."
        )

    grid_preview = output_dir / "voxel_grid_preview.png"
    result_preview = output_dir / "fno3d_heat_preview.png"
    render_canonical(load_npz(grid_path), grid_preview)
    render_validation(load_npz(result_path), result_preview)
    print(f"Voxel-grid preview: {grid_preview}")
    print(f"FNO3D preview: {result_preview}")


if __name__ == "__main__":
    main()
