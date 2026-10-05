from __future__ import annotations

from models.registry import ModelRegistry

from .base import BaseGraphNetwork
from .gat import GAT, GATConfig
from .gcn import GCN, GCNConfig

ModelRegistry.register("gcn", GCN, GCNConfig)
ModelRegistry.register("gat", GAT, GATConfig)

__all__ = ["BaseGraphNetwork", "GAT", "GATConfig", "GCN", "GCNConfig"]
