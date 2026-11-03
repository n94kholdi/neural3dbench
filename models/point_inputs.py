from __future__ import annotations

import torch

from .types import ModelInput


def validate_point_cloud_inputs(
    inputs: ModelInput, feature_dim: int, model_name: str
) -> None:
    """Check the shared ``[B, N, 3]`` point-cloud contract used by point models."""

    points = inputs.coordinates
    points = points if isinstance(points, torch.Tensor) else torch.as_tensor(points)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(
            f"{model_name} coordinates must have shape [B, N, 3], "
            f"got {tuple(points.shape)}."
        )
    if not points.is_floating_point():
        raise TypeError("coordinates must use a floating-point dtype.")
    if points.shape[1] < 1:
        raise ValueError("Each point cloud must contain at least one point.")

    if inputs.point_features is None:
        if feature_dim:
            raise ValueError(
                f"{model_name} expects {feature_dim} additional point features."
            )
    else:
        features = torch.as_tensor(inputs.point_features)
        expected = (*points.shape[:2], feature_dim)
        if tuple(features.shape) != expected:
            raise ValueError(
                f"point_features must have shape {expected}, got {tuple(features.shape)}."
            )
        if not features.is_floating_point():
            raise TypeError("point_features must use a floating-point dtype.")

    if inputs.point_mask is not None:
        mask = torch.as_tensor(inputs.point_mask)
        if tuple(mask.shape) != tuple(points.shape[:2]):
            raise ValueError(
                f"point_mask must have shape {tuple(points.shape[:2])}, "
                f"got {tuple(mask.shape)}."
            )
        if mask.dtype != torch.bool:
            raise TypeError("point_mask must use a boolean dtype.")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every point cloud must contain at least one valid point.")


def prepare_point_cloud_inputs(
    inputs: ModelInput, parameter: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Move points, optional features, and a full validity mask to the model device."""

    points = torch.as_tensor(
        inputs.coordinates, device=parameter.device, dtype=parameter.dtype
    )
    features = None
    if inputs.point_features is not None:
        features = torch.as_tensor(
            inputs.point_features, device=parameter.device, dtype=parameter.dtype
        )
    mask = (
        torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        if inputs.point_mask is None
        else torch.as_tensor(inputs.point_mask, device=points.device, dtype=torch.bool)
    )
    return points, features, mask
