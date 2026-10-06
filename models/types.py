from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ModelCapabilities:
    representation: str | None = None
    requires: set[str] = field(default_factory=set)
    supports: set[str] = field(default_factory=set)
    variable_geometry: bool = False
    variable_node_count: bool = False


@dataclass
class ModelInput:
    coordinates: Any = None
    node_features: Any = None
    edge_index: Any = None
    edge_features: Any = None
    point_features: Any = None
    point_mask: Any = None
    grid: Any = None
    voxel_fields: Any = None
    boundary_conditions: Any = None
    boundary_mask: Any = None
    boundary_types: Any = None
    material_properties: Any = None
    physical_parameters: Any = None
    time: Any = None
    global_features: Any = None
    batch: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    representation: str | None = None

    def __post_init__(self) -> None:
        for field_name in [
            "coordinates",
            "node_features",
            "edge_index",
            "edge_features",
            "point_features",
            "point_mask",
            "grid",
            "voxel_fields",
            "boundary_conditions",
            "boundary_mask",
            "boundary_types",
            "material_properties",
            "physical_parameters",
            "time",
            "global_features",
            "batch",
        ]:
            value = getattr(self, field_name)
            if value is None or isinstance(value, (torch.Tensor, dict, str, bytes)):
                continue
            if isinstance(value, (list, tuple)) or hasattr(value, "shape"):
                try:
                    setattr(self, field_name, torch.as_tensor(value))
                except (TypeError, ValueError):
                    pass

    def available_fields(self) -> set[str]:
        return {
            field_name
            for field_name in self.__dataclass_fields__
            if getattr(self, field_name) is not None
        }

    def has(self, field_name: str) -> bool:
        return field_name in self.available_fields()

    def to_dict(self) -> dict[str, Any]:
        data = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value is not None:
                data[field_name] = value
        return data


@dataclass
class ModelOutput:
    predictions: Any
    latent: Any = None
    auxiliary: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def main_prediction(self) -> Any:
        return self.predictions

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": self.predictions,
            "latent": self.latent,
            "auxiliary": self.auxiliary,
            "metadata": self.metadata,
        }
