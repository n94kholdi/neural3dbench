from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from models.types import ModelInput


def _resolution_tuple(resolution: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(resolution, int):
        result = (resolution,) * 3
    else:
        result = tuple(int(value) for value in resolution)
        if len(result) != 3:
            raise ValueError("resolution must be an int or a (D, H, W) sequence.")
    if any(value <= 0 for value in result):
        raise ValueError("Every resolution dimension must be positive.")
    return result


def voxelize_points(
    points: Any,
    *,
    resolution: int | Sequence[int] = 32,
    features: Any = None,
    point_mask: Any = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    include_occupancy: bool = True,
) -> torch.Tensor:
    """Voxelize points and mean-reduce optional features into ``[B,C,D,H,W]``.

    Coordinates use conventional XYZ order, while tensor spatial axes use DHW
    (Z, Y, X). Points outside explicit bounds are ignored rather than clipped.
    """

    points = torch.as_tensor(points)
    unbatched = points.ndim == 2
    if unbatched:
        points = points.unsqueeze(0)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [N, 3] or [B, N, 3].")
    if not points.is_floating_point():
        points = points.float()
    batch_size, point_count, _ = points.shape
    depth, height, width = _resolution_tuple(resolution)

    feature_tensor = None
    if features is not None:
        feature_tensor = torch.as_tensor(features, device=points.device)
        if unbatched and feature_tensor.ndim == 2:
            feature_tensor = feature_tensor.unsqueeze(0)
        if feature_tensor.ndim != 3 or feature_tensor.shape[:2] != (batch_size, point_count):
            raise ValueError("features must have shape [N, F] or [B, N, F].")
        feature_tensor = feature_tensor.to(dtype=points.dtype)
    if not include_occupancy and feature_tensor is None:
        raise ValueError("features are required when include_occupancy is false.")

    if point_mask is None:
        mask = torch.ones((batch_size, point_count), dtype=torch.bool, device=points.device)
    else:
        mask = torch.as_tensor(point_mask, device=points.device)
        if unbatched and mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != (batch_size, point_count) or mask.dtype != torch.bool:
            raise ValueError("point_mask must be boolean with shape [N] or [B, N].")

    if bounds is None:
        if not mask.any():
            raise ValueError("Cannot infer bounds without at least one valid point.")
        valid_points = points[mask]
        lower = valid_points.amin(dim=0)
        upper = valid_points.amax(dim=0)
    else:
        lower = torch.as_tensor(bounds[0], device=points.device, dtype=points.dtype)
        upper = torch.as_tensor(bounds[1], device=points.device, dtype=points.dtype)
        if lower.shape != (3,) or upper.shape != (3,) or torch.any(upper <= lower):
            raise ValueError("bounds must contain increasing XYZ lower/upper triples.")
        mask = mask & ((points >= lower) & (points <= upper)).all(dim=-1)

    extent = (upper - lower).clamp_min(torch.finfo(points.dtype).eps)
    normalized = (points - lower) / extent
    xyz_scale = points.new_tensor((width - 1, height - 1, depth - 1))
    xyz_indices = (normalized * xyz_scale).floor().long()
    xyz_indices[..., 0].clamp_(0, width - 1)
    xyz_indices[..., 1].clamp_(0, height - 1)
    xyz_indices[..., 2].clamp_(0, depth - 1)

    feature_channels = 0 if feature_tensor is None else feature_tensor.shape[-1]
    channels = feature_channels + int(include_occupancy)
    volume = points.new_zeros((batch_size, channels, depth * height * width))
    for batch_index in range(batch_size):
        valid = mask[batch_index]
        indices = xyz_indices[batch_index, valid]
        linear = indices[:, 2] * (height * width) + indices[:, 1] * width + indices[:, 0]
        counts = points.new_zeros(depth * height * width)
        counts.index_add_(0, linear, torch.ones_like(linear, dtype=points.dtype))
        channel_offset = 0
        if include_occupancy:
            volume[batch_index, 0] = (counts > 0).to(points.dtype)
            channel_offset = 1
        if feature_tensor is not None:
            sums = points.new_zeros((feature_channels, depth * height * width))
            sums.index_add_(1, linear, feature_tensor[batch_index, valid].transpose(0, 1))
            volume[batch_index, channel_offset:] = sums / counts.clamp_min(1).unsqueeze(0)
    return volume.view(batch_size, channels, depth, height, width)


def point_cloud_to_voxel_grid(
    points: Any,
    *,
    resolution: int | Sequence[int] = 32,
    features: Any = None,
    point_mask: Any = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    include_occupancy: bool = True,
) -> ModelInput:
    """Adapt a point cloud to the common structured-grid model contract."""

    volume = voxelize_points(
        points,
        resolution=resolution,
        features=features,
        point_mask=point_mask,
        bounds=bounds,
        include_occupancy=include_occupancy,
    )
    return ModelInput(
        voxel_fields=volume,
        representation="GRID",
        metadata={"resolution": tuple(volume.shape[-3:]), "source": "point_cloud"},
    )


def mesh_to_voxel_grid(
    mesh: Any,
    *,
    resolution: int | Sequence[int] = 32,
    feature_names: Sequence[str] = (),
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    include_occupancy: bool = True,
) -> ModelInput:
    """Voxelize mesh vertices and selected node fields without touching topology."""

    if not hasattr(mesh, "vertices"):
        raise TypeError("mesh must expose a vertices tensor.")
    points = torch.as_tensor(mesh.vertices)
    if points.ndim != 2 or points.shape[1] not in {2, 3}:
        raise ValueError("mesh vertices must have shape [N, 2] or [N, 3].")
    if points.shape[1] == 2:
        points = torch.nn.functional.pad(points, (0, 1))
    fields = []
    node_data = getattr(mesh, "node_data", {})
    for name in feature_names:
        if name not in node_data:
            raise KeyError(f"Mesh node_data does not contain '{name}'.")
        field = torch.as_tensor(node_data[name], device=points.device)
        if field.ndim == 1:
            field = field.unsqueeze(-1)
        if field.ndim != 2 or field.shape[0] != points.shape[0]:
            raise ValueError(f"node_data['{name}'] must have shape [N] or [N, F].")
        fields.append(field.to(dtype=points.dtype))
    features = torch.cat(fields, dim=-1) if fields else None
    result = point_cloud_to_voxel_grid(
        points,
        resolution=resolution,
        features=features,
        bounds=bounds,
        include_occupancy=include_occupancy,
    )
    result.metadata.update({"source": "mesh_vertices", "feature_names": list(feature_names)})
    return result
