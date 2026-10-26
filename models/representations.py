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
            query_coordinates=payload.get(
                "query_coordinates", kwargs.pop("query_coordinates", None)
            ),
            metadata=payload.get("metadata", {}),
            representation="POINT_CLOUD",
            **kwargs,
        )


class ParametricSurfaceRepresentationAdapter(RepresentationAdapter):
    """Adapt canonical ``PARAMETRIC_SURFACE`` samples to ``ModelInput``.

    The NURBS definition belongs in :class:`DeepNURBSConfig`; only sampled
    parametric coordinates are runtime model input.
    """

    def adapt(self, raw_data: Any, **kwargs) -> ModelInput:
        if isinstance(raw_data, Mapping):
            payload = dict(raw_data)
        else:
            payload = {"parametric_coordinates": raw_data}

        representation = payload.get("representation")
        if representation is not None and str(representation) != "PARAMETRIC_SURFACE":
            raise ValueError("Expected a PARAMETRIC_SURFACE payload.")
        return ModelInput(
            coordinates=payload.get(
                "parametric_coordinates", payload.get("coordinates")
            ),
            metadata=payload.get("metadata", {}),
            representation="PARAMETRIC_SURFACE",
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
