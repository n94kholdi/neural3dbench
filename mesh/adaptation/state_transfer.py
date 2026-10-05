from __future__ import annotations

from typing import Mapping

import torch

from mesh.types import NodeMapping, TransferMode, TriangularMesh

from .base import StateTransfer


class BarycentricStateTransfer(StateTransfer):
    """Barycentric transfer for continuous data and discrete parent selection."""

    def transfer(
        self,
        old_mesh: TriangularMesh,
        new_mesh: TriangularMesh,
        state: Mapping[str, torch.Tensor],
        mapping: NodeMapping,
        field_modes: Mapping[str, TransferMode] | None = None,
    ) -> dict[str, torch.Tensor]:
        modes = field_modes or {}
        transferred: dict[str, torch.Tensor] = {}
        for name, raw_values in state.items():
            values = torch.as_tensor(raw_values)
            if values.shape[0] != old_mesh.num_nodes:
                raise ValueError(f"state['{name}'] must have one value per old mesh node.")
            parents = mapping.parent_indices.to(values.device)
            weights = mapping.weights.to(device=values.device, dtype=torch.get_default_dtype())
            mode = modes.get(name, TransferMode.CONTINUOUS)
            gathered = values[parents]
            if mode is TransferMode.CONTINUOUS:
                if not values.is_floating_point():
                    raise TypeError(
                        f"continuous state field '{name}' must use a floating-point dtype."
                    )
                value_weights = weights.to(values.dtype)
                while value_weights.ndim < gathered.ndim:
                    value_weights = value_weights.unsqueeze(-1)
                transferred[name] = (gathered * value_weights).sum(dim=1)
            elif mode is TransferMode.CATEGORICAL:
                selected = weights.argmax(dim=1)
                rows = torch.arange(parents.shape[0], device=values.device)
                transferred[name] = gathered[rows, selected]
            else:
                selected = weights.argmax(dim=1)
                rows = torch.arange(parents.shape[0], device=values.device)
                result = gathered[rows, selected].clone()
                is_original = weights.amax(dim=1) == 1
                result[~is_original] = 0
                transferred[name] = result
        return transferred
