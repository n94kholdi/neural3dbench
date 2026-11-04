"""Plotly figures for point clouds, surface meshes, predicted-vs-true fields and part labels.

Inputs may be NumPy arrays, CPU torch tensors or nested lists. Samples above
``MAX_RENDER_ELEMENTS`` points/vertices are downsampled for browser rendering.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MAX_RENDER_ELEMENTS = 30_000
_SEED = 0
_COLORSCALE = "Viridis"
_ERROR_COLORSCALE = "Inferno"
_CAMERA = {"eye": {"x": 1.5, "y": 1.5, "z": 1.0}}


def _coords(array: Any, name: str) -> np.ndarray:
    coords = np.asarray(array, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3]; got {coords.shape}.")
    return coords


def _faces(array: Any, num_vertices: int) -> np.ndarray:
    faces = np.asarray(array, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape [F, 3]; got {faces.shape}.")
    if faces.size and (faces.min() < 0 or faces.max() >= num_vertices):
        raise ValueError("faces index outside the vertex array.")
    return faces


def _scalar(values: Any, n: int, name: str) -> np.ndarray:
    """Return a per-element scalar of shape [N]; multi-channel inputs [N, C] use the L2 norm."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim == 2:
        array = np.linalg.norm(array, axis=1)
    if array.shape != (n,):
        raise ValueError(f"{name} must have shape [N] or [N, C] with N={n}; got {np.asarray(values).shape}.")
    return array


def downsample_points(n: int, limit: int = MAX_RENDER_ELEMENTS) -> np.ndarray:
    """Sorted random indices of at most ``limit`` elements out of ``n``."""
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(_SEED).choice(n, size=limit, replace=False))


def downsample_mesh(faces: np.ndarray, num_vertices: int, limit: int = MAX_RENDER_ELEMENTS) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertex_index, reindexed_faces)`` keeping at most ``limit`` vertices.

    shortcut: random face subset (limit // 3 faces) leaves holes; switch to quadric decimation if previews look sparse.
    """
    if num_vertices <= limit:
        return np.arange(num_vertices), faces
    keep = downsample_points(len(faces), limit // 3)
    vertex_index, inverse = np.unique(faces[keep], return_inverse=True)
    return vertex_index, inverse.reshape(-1, 3)


def _scene() -> dict[str, Any]:
    return {"aspectmode": "data", "camera": _CAMERA, "dragmode": "orbit"}


def plot_point_cloud(coords: Any, features: Any | None = None, title: str = "Point Cloud") -> go.Figure:
    """Scatter3d of ``coords`` [N, 3], colored by ``features`` [N] / [N, C] or by the z coordinate."""
    points = _coords(coords, "coords")
    color = _scalar(features, len(points), "features") if features is not None else points[:, 2]
    total = len(points)
    index = downsample_points(total)
    points, color = points[index], color[index]

    figure = go.Figure(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker={"size": 2, "color": color, "colorscale": _COLORSCALE, "colorbar": {"title": "value" if features is not None else "z"}},
        )
    )
    shown = f" ({len(index):,} of {total:,} shown)" if len(index) < total else ""
    figure.update_layout(title=title + shown, scene=_scene(), margin={"l": 0, "r": 0, "t": 40, "b": 0})
    return figure


def _mesh_trace(vertices: np.ndarray, faces: np.ndarray | None, field: np.ndarray | None, **kwargs: Any) -> go.Mesh3d:
    trace: dict[str, Any] = {"x": vertices[:, 0], "y": vertices[:, 1], "z": vertices[:, 2], "flatshading": False}
    if faces is not None:
        trace.update(i=faces[:, 0], j=faces[:, 1], k=faces[:, 2])
    else:
        trace["alphahull"] = 0  # no connectivity: render the convex hull
    if field is not None:
        trace.update(intensity=field, colorscale=_COLORSCALE)
    else:
        trace["color"] = "lightsteelblue"
    trace.update(kwargs)
    return go.Mesh3d(**trace)


def plot_mesh(vertices: Any, faces: Any | None = None, field: Any | None = None, title: str = "Surface Mesh") -> go.Figure:
    """Mesh3d of ``vertices`` [N, 3] and triangle ``faces`` [F, 3], colored by a scalar ``field`` [N]."""
    verts = _coords(vertices, "vertices")
    values = _scalar(field, len(verts), "field") if field is not None else None
    tris = _faces(faces, len(verts)) if faces is not None else None

    if tris is not None:
        index, tris = downsample_mesh(tris, len(verts))
    else:
        index = downsample_points(len(verts))
    verts = verts[index]
    values = values[index] if values is not None else None

    figure = go.Figure(_mesh_trace(verts, tris, values, colorbar={"title": "field"} if values is not None else None))
    figure.update_layout(title=title, scene=_scene(), margin={"l": 0, "r": 0, "t": 40, "b": 0})
    return figure


def plot_field_comparison(
    vertices: Any,
    faces: Any | None,
    gt_field: Any,
    pred_field: Any,
    metric_name: str = "Pressure",
) -> go.Figure:
    """1x3 subplots: ground truth, prediction and absolute error |y_true - y_pred|.

    Ground truth and prediction share one color range. All three scenes start from
    the same camera; Plotly has no cross-scene camera linking without custom JS.
    """
    verts = _coords(vertices, "vertices")
    gt = _scalar(gt_field, len(verts), "gt_field")
    pred = _scalar(pred_field, len(verts), "pred_field")
    tris = _faces(faces, len(verts)) if faces is not None else None

    if tris is not None:
        index, tris = downsample_mesh(tris, len(verts))
    else:
        index = downsample_points(len(verts))
    verts, gt, pred = verts[index], gt[index], pred[index]
    error = np.abs(gt - pred)
    low, high = float(min(gt.min(), pred.min())), float(max(gt.max(), pred.max()))

    figure = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}] * 3],
        subplot_titles=(f"Ground truth {metric_name}", f"Predicted {metric_name}", f"|Error| {metric_name}"),
        horizontal_spacing=0.02,
    )
    shared = {"cmin": low, "cmax": high}
    figure.add_trace(_mesh_trace(verts, tris, gt, colorbar={"title": metric_name, "x": 0.30, "len": 0.8}, **shared), 1, 1)
    figure.add_trace(_mesh_trace(verts, tris, pred, colorbar={"title": metric_name, "x": 0.64, "len": 0.8}, **shared), 1, 2)
    figure.add_trace(
        _mesh_trace(verts, tris, error, colorscale=_ERROR_COLORSCALE, colorbar={"title": "|err|", "x": 0.98, "len": 0.8}), 1, 3
    )
    scene = _scene()
    figure.update_layout(
        scene=scene,
        scene2=scene,
        scene3=scene,
        title=f"{metric_name}: MAE {error.mean():.4g}, max {error.max():.4g}",
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        uirevision="field-comparison",
    )
    return figure


_PART_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22")


def class_color(index: int, class_colors: Any = ()) -> str:
    """The dataset's color for class ``index``, else the default palette (cycles after 8)."""
    return class_colors[index] if index < len(class_colors) else _PART_COLORS[index % len(_PART_COLORS)]


def plot_labelled_points(coords: Any, labels: Any, class_names: Any = (), class_colors: Any = (), title: str = "") -> go.Figure:
    """Scatter3d with one color and legend entry per class id in ``labels`` [N]."""
    points = _coords(coords, "coords")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    index = downsample_points(len(points))
    points, labels = points[index], labels[index]
    figure = go.Figure()
    for c in np.unique(labels):
        p = points[labels == c]
        name = class_names[c] if c < len(class_names) else f"class {c}"
        figure.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name=name,
                                      marker={"size": 2, "color": class_color(int(c), class_colors)}))
    figure.update_layout(title=title, scene=_scene(), legend={"itemsizing": "constant"}, margin={"l": 0, "r": 0, "t": 40, "b": 0})
    return figure


def plot_segmentation_comparison(
    coords: Any, gt_labels: Any, pred_labels: Any, class_names: Any = (), class_colors: Any = ()
) -> go.Figure:
    """1x3 subplots: ground-truth parts, predicted parts and misclassified points (red), one color per part.

    ``class_colors`` (hex per class) overrides the default palette, which cycles after 8 parts.
    """
    points = _coords(coords, "coords")
    gt = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    pred = np.asarray(pred_labels, dtype=np.int64).reshape(-1)
    if gt.shape != (len(points),) or pred.shape != (len(points),):
        raise ValueError(f"labels must have shape [N] with N={len(points)}; got {gt.shape} and {pred.shape}.")
    index = downsample_points(len(points))
    points, gt, pred = points[index], gt[index], pred[index]
    names = list(class_names) or [f"part {c}" for c in range(int(max(gt.max(), pred.max())) + 1)]

    figure = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3], horizontal_spacing=0.02,
                           subplot_titles=("Ground-truth parts", "Predicted parts", "Misclassified points (red)"))

    def scatter(mask: np.ndarray, color: str, name: str, show: bool) -> go.Scatter3d:
        p = points[mask]
        return go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", marker={"size": 2, "color": color},
                            name=name, legendgroup=name, showlegend=show)

    for c, name in enumerate(names):
        color = class_color(c, class_colors)
        figure.add_trace(scatter(gt == c, color, name, True), 1, 1)
        figure.add_trace(scatter(pred == c, color, name, False), 1, 2)
    wrong = gt != pred
    figure.add_trace(scatter(~wrong, "lightgray", "correct", True), 1, 3)
    figure.add_trace(scatter(wrong, "red", "misclassified", True), 1, 3)

    ious = [((gt == c) & (pred == c)).sum() / union for c in range(len(names)) if (union := ((gt == c) | (pred == c)).sum())]
    scene = _scene()
    figure.update_layout(
        scene=scene, scene2=scene, scene3=scene,
        title=f"Point accuracy {1 - wrong.mean():.1%}, mIoU {np.mean(ious):.1%}",
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        uirevision="segmentation-comparison",
    )
    return figure


def plot_completion_comparison(partial: Any, target: Any, pred: Any) -> go.Figure:
    """1x3 subplots: partial input ``[N, 3]``, complete ground truth ``[N, 3]`` and predicted completion ``[M, 3]``."""
    clouds = [_coords(partial, "partial"), _coords(target, "target"), _coords(pred, "pred")]
    figure = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3], horizontal_spacing=0.02,
                           subplot_titles=("Partial input", "Ground truth", "Predicted completion"))
    for col, (points, color) in enumerate(zip(clouds, ("gray", "seagreen", "royalblue")), 1):
        points = points[downsample_points(len(points))]
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                                      marker={"size": 2, "color": color}, showlegend=False), 1, col)
    scene = _scene()
    figure.update_layout(scene=scene, scene2=scene, scene3=scene, title="Shape completion",
                         margin={"l": 0, "r": 0, "t": 60, "b": 0}, uirevision="completion-comparison")
    return figure
