from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BaseModelConfig:
    name: str = "base_model"
    model_type: str | None = None
    representation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BaseModelConfig":
        if data is None:
            return cls()
        payload = dict(data)
        signature = inspect.signature(cls.__init__)
        accepted = set(signature.parameters)

        for key in ["name", "model_type", "representation", "metadata"]:
            if key not in accepted:
                payload.pop(key, None)

        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DummyConfig(BaseModelConfig):
    hidden_dim: int = 8
    out_dim: int = 1
