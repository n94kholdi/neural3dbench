from __future__ import annotations

from typing import Any, Mapping

from .types import ModelInput


class RepresentationAdapter:
    """Transforms raw problem data into a model-ready input container."""

    def adapt(self, raw_data: Any, **kwargs) -> ModelInput:
        raise NotImplementedError

    __call__ = adapt


class GraphRepresentationAdapter(RepresentationAdapter):
    def adapt(self, raw_data: Any, **kwargs) -> ModelInput:
        if isinstance(raw_data, Mapping):
            payload = dict(raw_data)
        else:
            payload = {"coordinates": raw_data}

        return ModelInput(
            coordinates=payload.get("coordinates"),
            node_features=payload.get("node_features"),
            edge_index=payload.get("edge_index"),
            edge_features=payload.get("edge_features"),
            metadata=payload.get("metadata", {}),
            representation="GRAPH",
            **kwargs,
        )


class CoordinateRepresentationAdapter(RepresentationAdapter):
    def adapt(self, raw_data: Any, **kwargs) -> ModelInput:
        if isinstance(raw_data, Mapping):
            payload = dict(raw_data)
        else:
            payload = {"coordinates": raw_data}

        return ModelInput(
            coordinates=payload.get("coordinates"),
            physical_parameters=payload.get("physical_parameters"),
            global_features=payload.get("global_features"),
            metadata=payload.get("metadata", {}),
            representation="COORDINATES",
            **kwargs,
        )


class PointCloudRepresentationAdapter(RepresentationAdapter):
    """Adapt point arrays or mappings to the PointNet input contract.

    Mappings may use either ``coordinates``/``point_features`` or the more
    domain-friendly aliases ``points``/``features``.
    """

    def adapt(self, raw_data: Any, **kwargs) -> ModelInput:
        if isinstance(raw_data, Mapping):
            payload = dict(raw_data)
        else:
            payload = {"coordinates": raw_data}

        return ModelInput(
            coordinates=payload.get("coordinates", payload.get("points")),
            point_features=payload.get("point_features", payload.get("features")),
            point_mask=payload.get("point_mask", payload.get("mask")),
            metadata=payload.get("metadata", {}),
            representation="POINT_CLOUD",
            **kwargs,
        )


class GridRepresentationAdapter(RepresentationAdapter):
    def adapt(self, raw_data: Any, **kwargs) -> ModelInput:
        if isinstance(raw_data, Mapping):
            payload = dict(raw_data)
        else:
            payload = {"grid": raw_data, "voxel_fields": raw_data}

        return ModelInput(
            grid=payload.get("grid"),
            voxel_fields=payload.get("voxel_fields", payload.get("grid")),
            material_properties=payload.get("material_properties"),
            boundary_mask=payload.get("boundary_mask", payload.get("geometry_mask")),
            metadata=payload.get("metadata", {}),
            representation="GRID",
            **kwargs,
        )
