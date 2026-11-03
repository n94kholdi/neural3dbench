from models.registry import ModelRegistry

from .components import EdgeConv, knn
from .model import DGCNN, DGCNNConfig

ModelRegistry.register("dgcnn", DGCNN, DGCNNConfig)

__all__ = ["DGCNN", "DGCNNConfig", "EdgeConv", "knn"]
