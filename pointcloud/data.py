from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from models.types import ModelInput


@dataclass
class PointCloudSample:
    points: torch.Tensor
    features: torch.Tensor | None = None
    target: torch.Tensor | int | None = None
    metadata: dict[str, Any] | None = None


def normalize_points(
    points: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    """Center point clouds and scale each to its furthest-point radius."""

    points = torch.as_tensor(points)
    if points.ndim not in {2, 3} or points.shape[-1] != 3:
        raise ValueError("points must have shape [N, 3] or [B, N, 3].")
    point_axis = 0 if points.ndim == 2 else 1
    center = points.mean(dim=point_axis, keepdim=True)
    centered = points - center
    radius = torch.linalg.vector_norm(centered, dim=-1).amax(
        dim=point_axis, keepdim=True
    )
    return centered / radius.clamp_min(eps).unsqueeze(-1)


def sample_points(
    points: torch.Tensor,
    num_points: int,
    features: torch.Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Randomly sample one unbatched cloud, using replacement when needed."""

    points = torch.as_tensor(points)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape [N, 3].")
    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    if points.shape[0] == 0:
        raise ValueError("Cannot sample an empty point cloud.")
    if features is not None:
        features = torch.as_tensor(features)
        if features.ndim != 2 or features.shape[0] != points.shape[0]:
            raise ValueError("features must have shape [N, F].")

    if num_points <= points.shape[0]:
        indices = torch.randperm(
            points.shape[0], generator=generator, device=points.device
        )[:num_points]
    else:
        indices = torch.randint(
            points.shape[0],
            (num_points,),
            generator=generator,
            device=points.device,
        )
    sampled_features = None if features is None else features[indices]
    return points[indices], sampled_features


def _as_sample(item: PointCloudSample | Mapping[str, Any]) -> PointCloudSample:
    if isinstance(item, PointCloudSample):
        return item
    if not isinstance(item, Mapping):
        raise TypeError("Each batch item must be PointCloudSample or a mapping.")
    points = item.get("points", item.get("coordinates"))
    if points is None:
        raise ValueError("Each point-cloud sample requires points or coordinates.")
    return PointCloudSample(
        points=torch.as_tensor(points),
        features=(
            None
            if item.get("features", item.get("point_features")) is None
            else torch.as_tensor(item.get("features", item.get("point_features")))
        ),
        target=item.get("target"),
        metadata=item.get("metadata"),
    )


def collate_point_clouds(
    samples: Sequence[PointCloudSample | Mapping[str, Any]],
    *,
    num_points: int | None = None,
    normalize: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[ModelInput, torch.Tensor | list[Any] | None]:
    """Batch clouds by optional sampling or zero-padding with a validity mask."""

    if not samples:
        raise ValueError("Cannot collate an empty sequence.")
    converted = [_as_sample(item) for item in samples]
    feature_presence = [sample.features is not None for sample in converted]
    if any(feature_presence) and not all(feature_presence):
        raise ValueError("Either all samples or no samples must provide features.")

    processed: list[PointCloudSample] = []
    for sample in converted:
        points = torch.as_tensor(sample.points)
        features = sample.features
        if num_points is not None:
            points, features = sample_points(
                points, num_points, features, generator=generator
            )
        if normalize:
            points = normalize_points(points)
        processed.append(
            PointCloudSample(points, features, sample.target, sample.metadata)
        )

    maximum = max(sample.points.shape[0] for sample in processed)
    batch_size = len(processed)
    points = processed[0].points.new_zeros((batch_size, maximum, 3))
    mask = torch.zeros((batch_size, maximum), dtype=torch.bool, device=points.device)
    features = None
    if all(feature_presence):
        feature_dim = int(processed[0].features.shape[1])  # type: ignore[union-attr]
        features = processed[0].features.new_zeros(  # type: ignore[union-attr]
            (batch_size, maximum, feature_dim)
        )

    for index, sample in enumerate(processed):
        count = sample.points.shape[0]
        points[index, :count] = sample.points
        mask[index, :count] = True
        if features is not None and sample.features is not None:
            if sample.features.shape != (count, features.shape[-1]):
                raise ValueError("All point features must have the same feature dimension.")
            features[index, :count] = sample.features

    targets = [sample.target for sample in processed]
    target_batch: torch.Tensor | list[Any] | None
    if all(target is None for target in targets):
        target_batch = None
    else:
        try:
            target_batch = torch.stack([torch.as_tensor(target) for target in targets])
        except RuntimeError:
            target_batch = targets

    model_input = ModelInput(
        coordinates=points,
        point_features=features,
        point_mask=mask,
        representation="POINT_CLOUD",
        metadata={"samples": [sample.metadata or {} for sample in processed]},
    )
    return model_input, target_batch


def mesh_to_point_cloud(
    mesh: Any,
    *,
    feature_names: Sequence[str] = (),
    num_points: int | None = None,
    normalize: bool = False,
    generator: torch.Generator | None = None,
) -> ModelInput:
    """Extract mesh vertices and selected node fields without changing the mesh."""

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
        value = torch.as_tensor(node_data[name], device=points.device)
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        if value.ndim != 2 or value.shape[0] != points.shape[0]:
            raise ValueError(f"node_data['{name}'] must have shape [N] or [N, F].")
        fields.append(value.to(dtype=points.dtype))
    features = torch.cat(fields, dim=-1) if fields else None

    if num_points is not None:
        points, features = sample_points(
            points, num_points, features, generator=generator
        )
    if normalize:
        points = normalize_points(points)
    return ModelInput(
        coordinates=points.unsqueeze(0),
        point_features=None if features is None else features.unsqueeze(0),
        point_mask=torch.ones((1, points.shape[0]), dtype=torch.bool, device=points.device),
        representation="POINT_CLOUD",
        metadata={"source": "mesh_vertices", "feature_names": list(feature_names)},
    )
