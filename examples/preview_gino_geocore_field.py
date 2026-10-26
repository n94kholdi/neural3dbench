"""Render source values and GINO query predictions from saved training output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def project(points: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    azimuth = np.deg2rad(42.0)
    elevation = np.deg2rad(24.0)
    rotate_z = np.array(
        [[np.cos(azimuth), -np.sin(azimuth), 0.0],
         [np.sin(azimuth), np.cos(azimuth), 0.0],
         [0.0, 0.0, 1.0]]
    )
    rotate_x = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(elevation), -np.sin(elevation)],
         [0.0, np.sin(elevation), np.cos(elevation)]]
    )
    rotated = (points - points.mean(axis=0)) @ (rotate_z @ rotate_x).T
    extent = float(np.ptp(rotated[:, :2], axis=0).max())
    scale = 0.72 * min(right - left, bottom - top) / max(extent, 1e-6)
    xy = rotated[:, :2] * scale
    xy[:, 0] += (right - left) / 2
    xy[:, 1] = (bottom - top) / 2 - xy[:, 1]
    return xy + np.asarray([left, top])


def colorize(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    stops = np.asarray([[45, 76, 145], [36, 166, 154], [245, 210, 74]], dtype=float)
    scaled = np.clip((values - lower) / max(upper - lower, 1e-8), 0.0, 1.0)
    position = scaled * (len(stops) - 1)
    lower_index = np.minimum(position.astype(int), len(stops) - 2)
    fraction = (position - lower_index)[:, None]
    colors = stops[lower_index] * (1.0 - fraction) + stops[lower_index + 1] * fraction
    return colors.astype(np.uint8)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    values: np.ndarray,
    box: tuple[int, int, int, int],
    title: str,
    value_range: tuple[float, float],
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=10, fill="#f5f7fa", outline="#d0d5dd", width=2)
    draw.text((left + 16, top + 12), title, font=font(19, bold=True), fill="#182230")
    plot_box = (left + 15, top + 48, right - 15, bottom - 15)
    projected = project(points, plot_box)
    colors = colorize(values, *value_range)
    order = np.argsort(points[:, 2])
    for index in order:
        x, y = projected[index]
        color = tuple(int(channel) for channel in colors[index])
        radius = 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/gino_geocore_field")
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    result_path = output_dir / "validation_result.npz"
    if not result_path.is_file():
        raise FileNotFoundError(f"Expected {result_path}; run GINO training first.")

    with np.load(result_path, allow_pickle=False) as archive:
        source = archive["source_coordinates"][0]
        source_values = archive["source_values"][0, :, 0]
        queries = archive["query_coordinates"][0]
        target = archive["targets"][0, :, 0]
        prediction = archive["predictions"][0, :, 0]
        mse = float(archive["validation_mse"])
    error = np.abs(prediction - target)

    width, height, margin, gap = 1500, 900, 34, 22
    panel_width = (width - 2 * margin - gap) // 2
    panel_height = (height - 2 * margin - gap) // 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (width // 2, 22),
        f"geocore POINT_CLOUD -> GINO source-to-query field | MSE {mse:.5f}",
        anchor="ma",
        font=font(24, bold=True),
        fill="#101828",
    )
    boxes = []
    for row in range(2):
        for column in range(2):
            left = margin + column * (panel_width + gap)
            top = 65 + row * (panel_height + gap)
            boxes.append((left, top, left + panel_width, top + panel_height))
    draw_panel(draw, source, source_values, boxes[0], "Source values", (0.0, 1.0))
    draw_panel(draw, queries, target, boxes[1], "Query target", (0.0, 1.0))
    draw_panel(draw, queries, prediction, boxes[2], "GINO prediction", (0.0, 1.0))
    draw_panel(draw, queries, error, boxes[3], "Absolute error", (0.0, max(float(error.max()), 1e-6)))
    preview_path = output_dir / "gino_field_preview.png"
    canvas.save(preview_path)
    print(f"GINO field preview: {preview_path}")


if __name__ == "__main__":
    main()