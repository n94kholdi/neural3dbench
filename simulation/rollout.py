from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

from mesh import (
    BarycentricStateTransfer,
    LocalVariationCriterion,
    MeshAdaptationConfig,
    MeshAdaptationController,
    TriangleCentroidRemesher,
    TriangularMesh,
)
from models.types import ModelInput, ModelOutput


State = Mapping[str, torch.Tensor]
StateUpdater = Callable[[TriangularMesh, State, ModelOutput], dict[str, torch.Tensor]]


@dataclass
class RolloutFrame:
    step: int
    model_input: ModelInput
    model_output: ModelOutput
    mesh_after_step: TriangularMesh
    state_after_step: dict[str, torch.Tensor]
    remesh_metadata: dict = field(default_factory=dict)


@dataclass
class RolloutResult:
    frames: list[RolloutFrame]
    final_mesh: TriangularMesh
    final_state: dict[str, torch.Tensor]


class ReplaceState:
    """Interpret model predictions as the next value of one state field."""

    def __init__(self, field: str = "state") -> None:
        self.field = field

    def __call__(
        self, mesh: TriangularMesh, state: State, output: ModelOutput
    ) -> dict[str, torch.Tensor]:
        del mesh
        updated = dict(state)
        updated[self.field] = output.predictions
        return updated


class MeshSimulationRunner:
    """External rollout orchestration for fixed and adaptive meshes.

    Topology decisions are discrete and outside autograd. Model evaluation and
    barycentric arithmetic remain ordinary differentiable PyTorch operations.
    """

    def __init__(
        self,
        model,
        graph_builder,
        *,
        mesh_adaptation: MeshAdaptationConfig | None = None,
        state_updater: StateUpdater | None = None,
        adaptation_controller: MeshAdaptationController | None = None,
    ) -> None:
        self.model = model
        self.graph_builder = graph_builder
        self.mesh_adaptation = mesh_adaptation or MeshAdaptationConfig()
        self.state_updater = state_updater or ReplaceState(
            self.mesh_adaptation.criterion_field
        )
        self.adaptation_controller = adaptation_controller or MeshAdaptationController(
            self.mesh_adaptation,
            LocalVariationCriterion(self.mesh_adaptation.criterion_field),
            TriangleCentroidRemesher(),
            BarycentricStateTransfer(),
        )

    def run(
        self,
        initial_mesh: TriangularMesh,
        initial_state: State,
        *,
        steps: int,
    ) -> RolloutResult:
        if steps < 1:
            raise ValueError("steps must be at least 1.")
        mesh = initial_mesh
        state = dict(initial_state)
        frames: list[RolloutFrame] = []
        for step in range(steps):
            model_input = self.graph_builder(mesh, state)
            output = self.model(model_input)
            updated_state = self.state_updater(mesh, state, output)
            outcome = self.adaptation_controller.adapt(
                step + 1, mesh, updated_state, output.predictions
            )
            mesh, state = outcome.mesh, outcome.state
            frames.append(
                RolloutFrame(
                    step=step,
                    model_input=model_input,
                    model_output=output,
                    mesh_after_step=mesh,
                    state_after_step=state,
                    remesh_metadata=(outcome.remesh.metadata if outcome.remesh else {}),
                )
            )
        return RolloutResult(frames, mesh, state)

