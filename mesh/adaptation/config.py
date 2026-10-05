from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MeshMode(str, Enum):
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


@dataclass
class MeshAdaptationConfig:
    """Configuration for the currently supported deterministic 2D refiner."""

    mode: MeshMode = MeshMode.FIXED
    adapt_every: int = 1
    criterion_field: str = "state"
    refinement_threshold: float = 1.0
    max_nodes: int | None = None
    max_elements: int | None = None
    max_refinement_level: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.mode, MeshMode):
            self.mode = MeshMode(self.mode)
        if self.adapt_every < 1:
            raise ValueError("adapt_every must be at least 1.")
        if self.refinement_threshold < 0:
            raise ValueError("refinement_threshold must be non-negative.")
        if self.max_nodes is not None and self.max_nodes < 3:
            raise ValueError("max_nodes must be at least 3.")
        if self.max_elements is not None and self.max_elements < 1:
            raise ValueError("max_elements must be positive.")
        if self.max_refinement_level < 0:
            raise ValueError("max_refinement_level must be non-negative.")

