from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _scalar_volume(value: Any, channel: int) -> torch.Tensor:
    volume = torch.as_tensor(value).detach().cpu()
    if volume.ndim == 5:
        if volume.shape[0] != 1:
            raise ValueError("Visualize one batch item at a time.")
        volume = volume[0]
    if volume.ndim == 4:
        if not 0 <= channel < volume.shape[0]:
            raise ValueError(f"channel must be in [0, {volume.shape[0] - 1}].")
        volume = volume[channel]
    if volume.ndim != 3:
        raise ValueError("Volumes must have shape [D,H,W], [C,D,H,W], or [1,C,D,H,W].")
    return volume


def plot_volume_slices(
    input_volume: Any,
    *,
    ground_truth: Any = None,
    prediction: Any = None,
    error: Any = None,
    channel: int = 0,
    slice_indices: Sequence[int] | None = None,
    title: str = "Volumetric prediction",
    show: bool = True,
):
    """Plot XY, XZ, and YZ slices for input, target, prediction, and error."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Volume plotting requires `pip install -e '.[viz]'`.") from exc

    fields: dict[str, torch.Tensor] = {"Input": _scalar_volume(input_volume, channel)}
    if ground_truth is not None:
        fields["Ground truth"] = _scalar_volume(ground_truth, channel)
    if prediction is not None:
        fields["Prediction"] = _scalar_volume(prediction, channel)
    if error is not None:
        fields["Absolute error"] = _scalar_volume(error, channel)
    elif ground_truth is not None and prediction is not None:
        fields["Absolute error"] = (fields["Prediction"] - fields["Ground truth"]).abs()

    shapes = {tuple(volume.shape) for volume in fields.values()}
    if len(shapes) != 1:
        raise ValueError("All visualization volumes must have the same spatial shape.")
    depth, height, width = next(iter(shapes))
    if slice_indices is None:
        z_index, y_index, x_index = depth // 2, height // 2, width // 2
    else:
        if len(slice_indices) != 3:
            raise ValueError("slice_indices must be (z, y, x).")
        z_index, y_index, x_index = (int(value) for value in slice_indices)
        if not (0 <= z_index < depth and 0 <= y_index < height and 0 <= x_index < width):
            raise ValueError("slice_indices are outside the volume.")

    figure, axes = plt.subplots(len(fields), 3, figsize=(12, 3.5 * len(fields)), squeeze=False)
    planes: Mapping[str, Any] = {
        "XY": lambda volume: volume[z_index],
        "XZ": lambda volume: volume[:, y_index, :],
        "YZ": lambda volume: volume[:, :, x_index],
    }
    for row, (field_name, volume) in enumerate(fields.items()):
        for column, (plane_name, extract) in enumerate(planes.items()):
            image = axes[row, column].imshow(extract(volume), origin="lower", cmap="viridis")
            axes[row, column].set_title(f"{field_name} — {plane_name}")
            figure.colorbar(image, ax=axes[row, column], shrink=0.75)
    figure.suptitle(title)
    figure.tight_layout()
    if show:
        plt.show()
    return figure
