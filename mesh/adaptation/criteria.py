from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from mesh.types import TriangularMesh

from .base import AdaptationCriterion


@dataclass
class LocalVariationCriterion(AdaptationCriterion):
    """Maximum edge-wise field gradient magnitude in each triangle."""

    field: str = "state"
    distance_epsilon: float = 1e-12

    def compute(
        self,
        mesh: TriangularMesh,
        state: Mapping[str, torch.Tensor],
        predictions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del predictions
        if self.field not in state:
            raise KeyError(f"adaptation field '{self.field}' is absent from state.")
        values = torch.as_tensor(state[self.field])
        if values.shape[0] != mesh.num_nodes:
            raise ValueError(f"state['{self.field}'] must have one value per mesh node.")
        if values.ndim == 1:
            values = values[:, None]
        tri = mesh.triangles.to(values.device)
        coordinates = mesh.vertices.to(values.device)
        pairs = ((0, 1), (1, 2), (2, 0))
        variations = []
        for first, second in pairs:
            difference = torch.linalg.vector_norm(
                values[tri[:, first]] - values[tri[:, second]], dim=-1
            )
            distance = torch.linalg.vector_norm(
                coordinates[tri[:, first]] - coordinates[tri[:, second]], dim=-1
            )
            variations.append(difference / distance.clamp_min(self.distance_epsilon))
        return torch.stack(variations, dim=1).amax(dim=1)

