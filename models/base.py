from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from .configs import BaseModelConfig
from .types import ModelCapabilities, ModelInput, ModelOutput


class BaseNetwork(nn.Module, ABC):
    """Stable interface every model must satisfy."""

    config_class = BaseModelConfig

    def __init__(self, config: BaseModelConfig | None = None):
        super().__init__()
        self.config = config or self.config_class()

    @property
    @abstractmethod
    def required_inputs(self) -> set[str]:
        raise NotImplementedError

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            representation=self.config.representation,
            requires=set(self.required_inputs),
            supports=set(),
            variable_geometry=False,
            variable_node_count=False,
        )

    @property
    def supported_representations(self) -> set[str]:
        rep = self.capabilities.representation
        return {rep} if rep is not None else set()

    def validate_inputs(self, inputs: ModelInput) -> bool:
        if not isinstance(inputs, ModelInput):
            raise TypeError(f"Expected ModelInput, got {type(inputs).__name__}.")

        required = set(self.required_inputs)
        available = inputs.available_fields()
        missing = required - available
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ValueError(
                f"{self.__class__.__name__} requires inputs: {missing_fields}. "
                f"Available: {sorted(available)}."
            )

        rep = self.capabilities.representation
        if rep is not None and inputs.representation is not None and inputs.representation != rep:
            raise ValueError(
                f"Model expects representation '{rep}' but received '{inputs.representation}'."
            )

        return True

    @abstractmethod
    def forward(self, inputs: ModelInput) -> ModelOutput:
        raise NotImplementedError

    def predict(self, inputs: ModelInput) -> ModelOutput:
        return self(inputs)

    @staticmethod
    def _ensure_tensor(value: Any, *, name: str = "tensor") -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value, dtype=torch.float32)
