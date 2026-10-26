"""Render the geocore NURBS domain and Deep NURBS validation field."""

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


def colorize(values: np.ndarray, lower: float, upper: float) -> Image.Image:
    scaled = np.clip((values - lower) / max(upper - lower, 1e-12), 0.0, 1.0)
    red = np.clip(2.2 * scaled - 0.35, 0.0, 1.0)
    green = np.clip(2.0 * scaled, 0.0, 1.0) * np.clip(2.0 - 1.2 * scaled, 0.0, 1.0)
    blue = np.clip(1.25 - 1.7 * scaled, 0.0, 1.0)
    rgb = (255.0 * np.stack((red, green, blue), axis=-1)).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def render_surface(surface: dict[str, np.ndarray], path: Path) -> None:
    canvas = Image.new("RGB", (760, 700), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (380, 25),
        "Canonical geocore PARAMETRIC_SURFACE",
        anchor="ma",
        font=font(23, bold=True),
        fill="#101828",
    )
    physical = surface["physical_coordinates"][0]
    control = surface["control_points"]
    all_points = np.concatenate((physical, control.reshape(-1, 2)))
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    span = np.maximum(upper - lower, 1e-9)

    def project(points):
        normalized = (np.asarray(points) - lower) / span
        return np.column_stack((80 + 600 * normalized[:, 0], 625 - 540 * normalized[:, 1]))

    samples = project(physical)
    for x, y in samples:
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill="#84adff")
    for axis in (0, 1):
        count = control.shape[1 - axis]
        for index in range(count):
            line = control[:, index] if axis == 0 else control[index, :]
            draw.line([tuple(point) for point in project(line)], fill="#d92d20", width=3)
    for x, y in project(control.reshape(-1, 2)):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#7a271a")
    draw.text((380, 665), "blue: sampled physical points   red: NURBS control net", anchor="mm", font=font(16), fill="#344054")
    canvas.save(path)


def render_result(result: dict[str, np.ndarray], path: Path) -> None:
    resolution = int(result["validation_resolution"])
    target = result["targets"][0, :, 0].reshape(resolution, resolution)
    prediction = result["predictions"][0, :, 0].reshape(resolution, resolution)
    error = np.abs(prediction - target)
    field_limit = max(abs(float(target.min())), abs(float(target.max())))
    panels = (
        ("Manufactured target", target, -field_limit, field_limit),
        ("Deep NURBS", prediction, -field_limit, field_limit),
        ("Absolute error", error, 0.0, max(float(error.max()), 1e-12)),
    )
    tile, gap, top = 260, 25, 130
    canvas = Image.new("RGB", (3 * tile + 4 * gap, 445), "white")
    draw = ImageDraw.Draw(canvas)
    mse = float(result["validation_mse"])
    rel = float(result["validation_relative_l2"])
    draw.text(
        (canvas.width // 2, 25),
        "geocore PARAMETRIC_SURFACE -> Deep NURBS",
        anchor="ma",
        font=font(21, bold=True),
        fill="#101828",
    )
    draw.text(
        (canvas.width // 2, 60),
        f"MSE {mse:.2e}   relative L2 {rel:.4f}",
        anchor="ma",
        font=font(17),
        fill="#475467",
    )
    for index, (label, values, lower, upper) in enumerate(panels):
        x = gap + index * (tile + gap)
        draw.text((x + tile // 2, 98), label, anchor="mm", font=font(17, bold=True), fill="#344054")
        image = colorize(values.T, lower, upper).resize((tile, tile), Image.Resampling.NEAREST)
        canvas.paste(image, (x, top))
        draw.rectangle((x, top, x + tile, top + tile), outline="#98a2b3", width=2)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/deep_nurbs_geocore_field"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    surface_path = output_dir / "parametric_surface.npz"
    result_path = output_dir / "validation_result.npz"
    if not surface_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(
            f"Expected {surface_path} and {result_path}; run training first."
        )

    surface_preview = output_dir / "parametric_surface_preview.png"
    result_preview = output_dir / "deep_nurbs_field_preview.png"
    render_surface(load_npz(surface_path), surface_preview)
    render_result(load_npz(result_path), result_preview)
    print(f"Parametric-surface preview: {surface_preview}")
    print(f"Deep NURBS preview: {result_preview}")


if __name__ == "__main__":
    main()
