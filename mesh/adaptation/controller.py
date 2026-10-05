from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from mesh.types import TransferMode, TriangularMesh

from .base import AdaptationCriterion, BaseRemesher, RemeshResult, StateTransfer
from .config import MeshAdaptationConfig, MeshMode


@dataclass
class AdaptationOutcome:
    mesh: TriangularMesh
    state: dict[str, torch.Tensor]
    remesh: RemeshResult | None


class MeshAdaptationController:
    def __init__(
        self,
        config: MeshAdaptationConfig,
        criterion: AdaptationCriterion,
        remesher: BaseRemesher,
        state_transfer: StateTransfer,
        field_modes: Mapping[str, TransferMode] | None = None,
    ) -> None:
        self.config = config
        self.criterion = criterion
        self.remesher = remesher
        self.state_transfer = state_transfer
        self.field_modes = dict(field_modes or {})

    def should_adapt(self, completed_steps: int) -> bool:
        return (
            self.config.mode is MeshMode.ADAPTIVE
            and completed_steps % self.config.adapt_every == 0
        )

    def adapt(
        self,
        completed_steps: int,
        mesh: TriangularMesh,
        state: Mapping[str, torch.Tensor],
        predictions: torch.Tensor | None = None,
    ) -> AdaptationOutcome:
        state_dict = dict(state)
        if not self.should_adapt(completed_steps):
            return AdaptationOutcome(mesh, state_dict, None)
        indicators = self.criterion.compute(mesh, state, predictions)
        result = self.remesher.adapt(mesh, state, indicators, self.config)
        if not result.changed:
            return AdaptationOutcome(mesh, state_dict, result)
        new_state = self.state_transfer.transfer(
            mesh, result.mesh, state, result.mapping, self.field_modes
        )
        return AdaptationOutcome(result.mesh, new_state, result)

