from __future__ import annotations

import torch


def pointwise_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    point_mask: torch.Tensor | None = None,
    loss: str = "mse",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Masked MSE, MAE, or sample-mean relative L2 for point fields."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [B, N, C].")
    if point_mask is None:
        point_mask = torch.ones(
            prediction.shape[:2], dtype=torch.bool, device=prediction.device
        )
    if point_mask.shape != prediction.shape[:2] or point_mask.dtype != torch.bool:
        raise ValueError("point_mask must be boolean with shape [B, N].")
    if not torch.all(point_mask.any(dim=1)):
        raise ValueError("Every sample must contain at least one valid point.")

    valid = point_mask.unsqueeze(-1)
    difference = prediction - target
    name = loss.lower()
    if name == "mse":
        return difference.square().masked_select(valid.expand_as(difference)).mean()
    if name == "mae":
        return difference.abs().masked_select(valid.expand_as(difference)).mean()
    if name in {"relative_l2", "rel_l2"}:
        valid_values = valid.to(dtype=prediction.dtype)
        numerator = (difference.square() * valid_values).sum(dim=(1, 2)).sqrt()
        denominator = (target.square() * valid_values).sum(dim=(1, 2)).sqrt()
        return (numerator / denominator.clamp_min(eps)).mean()
    raise ValueError("loss must be 'mse', 'mae', or 'relative_l2'.")
