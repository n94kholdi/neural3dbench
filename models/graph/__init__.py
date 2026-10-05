from __future__ import annotations

from models.registry import ModelRegistry

from .base import BaseGraphNetwork
from .gat import GAT, GATConfig
from .gcn import GCN, GCNConfig
from .meshgraphnet import MeshGraphNet, MeshGraphNetConfig

ModelRegistry.register("gcn", GCN, GCNConfig)
ModelRegistry.register("gat", GAT, GATConfig)
ModelRegistry.register("meshgraphnet", MeshGraphNet, MeshGraphNetConfig)

__all__ = [
    "BaseGraphNetwork",
    "GAT",
    "GATConfig",
    "GCN",
    "GCNConfig",
    "MeshGraphNet",
    "MeshGraphNetConfig",
]
