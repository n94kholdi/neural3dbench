from __future__ import annotations

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    """A SIREN linear layer followed by a scaled sine activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        omega_0: float = 30.0,
        is_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        if omega_0 <= 0:
            raise ValueError("omega_0 must be positive.")
        self.in_features = in_features
        self.omega_0 = float(omega_0)
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    @property
    def weight_bound(self) -> float:
        if self.is_first:
            return 1.0 / self.in_features
        return math.sqrt(6.0 / self.in_features) / self.omega_0

    def reset_parameters(self) -> None:
        # Deliberately initialize only weights. The official implementation
        # leaves nn.Linear's default bias initialization intact.
        with torch.no_grad():
            self.linear.weight.uniform_(-self.weight_bound, self.weight_bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(inputs))
