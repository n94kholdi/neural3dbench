"""Render saved geocore point clouds and PointNet validation predictions."""

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


def project(points: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return an isometric projection and depth order for a point cloud."""
    azimuth = np.deg2rad(42.0)
    elevation = np.deg2rad(24.0)
    rotation_z = np.array(
        [[np.cos(azimuth), -np.sin(azimuth), 0.0],
         [np.sin(azimuth), np.cos(azimuth), 0.0],
         [0.0, 0.0, 1.0]]
    )
    rotation_x = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(elevation), -np.sin(elevation)],
         [0.0, np.sin(elevation), np.cos(elevation)]]
    )
    rotated = (points - points.mean(axis=0)) @ (rotation_z @ rotation_x).T
    extent = float(np.ptp(rotated[:, :2], axis=0).max())
    scale = 0.72 * min(width, height) / max(extent, 1e-6)
    xy = rotated[:, :2] * scale
    xy[:, 0] += width / 2
    xy[:, 1] = height / 2 - xy[:, 1]
    return xy, np.argsort(rotated[:, 2])


def draw_cloud(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    projected, order = project(points, right - left, bottom - top)
    z = points[:, 2]
    z = (z - z.min()) / max(float(z.max() - z.min()), 1e-6)
    draw.rounded_rectangle(box, radius=18, fill="#f6f8fb", outline="#d0d5dd", width=2)
    for index in order:
        x = left + projected[index, 0]
        y = top + projected[index, 1]
        value = float(z[index])
        color = (
            int(38 + 210 * value),
            int(75 + 90 * (1.0 - abs(2.0 * value - 1.0))),
            int(170 - 105 * value),
        )
        radius = 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def render_validation(result: dict[str, np.ndarray], path: Path) -> None:
    points = result["coordinates"]
    targets = result["targets"]
    predictions = result["predictions"]
    probabilities = result["probabilities"]
    class_names = result["class_names"].tolist()

    selected = []
    for label in range(len(class_names)):
        matches = np.flatnonzero(targets == label)
        if len(matches):
            selected.append(int(matches[0]))

    width, height = 1600, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (width // 2, 28),
        "geocore POINT_CLOUD  →  PointNet",
        anchor="ma",
        font=font(30, bold=True),
        fill="#101828",
    )
    accuracy = float(np.mean(targets == predictions))
    draw.text((width // 2, 70), f"validation accuracy {accuracy:.1%}", anchor="ma", font=font(20), fill="#475467")
    margin, gap = 50, 24
    panel_width = (width - 2 * margin - gap * (len(selected) - 1)) // len(selected)
    for plot_index, sample_index in enumerate(selected):
        left = margin + plot_index * (panel_width + gap)
        xyz = points[sample_index]
        draw_cloud(draw, xyz, (left, 112, left + panel_width, 450))
        target = int(targets[sample_index])
        prediction = int(predictions[sample_index])
        confidence = float(probabilities[sample_index, prediction])
        title_color = "#157f3b" if target == prediction else "#b42318"
        draw.text(
            (left + panel_width // 2, 470),
            f"true: {class_names[target]}  •  predicted: {class_names[prediction]}",
            anchor="ma", font=font(17, bold=True), fill=title_color,
        )
        draw.text(
            (left + panel_width // 2, 508),
            f"confidence {confidence:.1%}",
            anchor="ma", font=font(16), fill="#475467",
        )
    image.save(path)


def render_canonical(cloud: dict[str, np.ndarray], path: Path) -> None:
    xyz = cloud["coordinates"][0]
    width, height = 760, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (width // 2, 32),
        "Canonical geocore POINT_CLOUD",
        anchor="ma",
        font=font(28, bold=True),
        fill="#101828",
    )
    draw.text(
        (width // 2, 74),
        f"{len(xyz)} points  •  normalization: per-sample unit sphere",
        anchor="ma", font=font(17), fill="#475467",
    )
    draw_cloud(draw, xyz, (100, 125, 660, 555))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pointnet_geocore_shapes"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    cloud_path = output_dir / "point_cloud.npz"
    result_path = output_dir / "validation_result.npz"
    if not cloud_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(
            f"Expected {cloud_path} and {result_path}; run training first."
        )

    cloud_preview = output_dir / "point_cloud_preview.png"
    result_preview = output_dir / "pointnet_shapes_preview.png"
    render_canonical(load_npz(cloud_path), cloud_preview)
    render_validation(load_npz(result_path), result_preview)
    print(f"Point-cloud preview: {cloud_preview}")
    print(f"PointNet preview: {result_preview}")


if __name__ == "__main__":
    main()
