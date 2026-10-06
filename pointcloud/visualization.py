from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def _scalar_colors(values: Any, point_count: int) -> torch.Tensor:
    values = torch.as_tensor(values).detach().cpu()
    if values.shape[0] != point_count:
        raise ValueError("Every visualization field must have one value per point.")
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    if values.ndim == 2:
        return torch.linalg.vector_norm(values, dim=1)
    raise ValueError("Fields must have shape [N], [N, 1], or [N, D].")


def plot_point_cloud(
    points: Any,
    *,
    labels: Any = None,
    ground_truth: Any = None,
    prediction: Any = None,
    error: Any = None,
    predicted_class: int | str | None = None,
    title: str = "Point cloud",
    show: bool = True,
):
    """Plot geometry and optional point fields using a lazy Matplotlib import."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Point-cloud plotting requires `pip install -e '.[viz]'`."
        ) from exc

    xyz = torch.as_tensor(points).detach().cpu()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points must have shape [N, 3].")
    fields: dict[str, Any] = {}
    if labels is not None:
        fields["Labels"] = labels
    if ground_truth is not None:
        fields["Ground truth"] = ground_truth
    if prediction is not None:
        fields["Prediction"] = prediction
    if error is not None:
        fields["Error"] = error
    elif ground_truth is not None and prediction is not None:
        fields["Absolute error"] = (
            torch.as_tensor(prediction) - torch.as_tensor(ground_truth)
        ).abs()

    panels: Mapping[str, Any] = fields or {"Input": None}
    figure = plt.figure(figsize=(5 * len(panels), 4))
    for index, (name, values) in enumerate(panels.items(), start=1):
        axis = figure.add_subplot(1, len(panels), index, projection="3d")
        colors = None if values is None else _scalar_colors(values, xyz.shape[0])
        scatter = axis.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=8)
        if colors is not None:
            figure.colorbar(scatter, ax=axis, shrink=0.65)
        axis.set_title(name)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
    suffix = "" if predicted_class is None else f" — class: {predicted_class}"
    figure.suptitle(title + suffix)
    figure.tight_layout()
    if show:
        plt.show()
    return figure
