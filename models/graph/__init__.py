from __future__ import annotations

from models.registry import ModelRegistry

from .base import BaseGraphNetwork
from .gcn import GCN, GCNConfig

ModelRegistry.register("gcn", GCN, GCNConfig)

__all__ = ["BaseGraphNetwork", "GCN", "GCNConfig"]
