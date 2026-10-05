from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from mesh.types import NodeMapping, TransferMode, TriangularMesh

from .config import MeshAdaptationConfig


@dataclass
class RemeshResult:
    mesh: TriangularMesh
    mapping: NodeMapping
    changed: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class AdaptationCriterion(ABC):
    @abstractmethod
    def compute(
        self,
        mesh: TriangularMesh,
        state: Mapping[str, torch.Tensor],
        predictions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one non-negative importance value per element."""


class BaseRemesher(ABC):
    @abstractmethod
    def adapt(
        self,
        mesh: TriangularMesh,
        state: Mapping[str, torch.Tensor],
        indicators: torch.Tensor,
        config: MeshAdaptationConfig,
    ) -> RemeshResult:
        """Return a geometrically valid mesh and old-to-new correspondence."""


class StateTransfer(ABC):
    @abstractmethod
    def transfer(
        self,
        old_mesh: TriangularMesh,
        new_mesh: TriangularMesh,
        state: Mapping[str, torch.Tensor],
        mapping: NodeMapping,
        field_modes: Mapping[str, TransferMode] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Transfer node state to the new discretization."""

