from models.registry import ModelRegistry

from .model import FoldingNet, FoldingNetConfig

ModelRegistry.register("foldingnet", FoldingNet, FoldingNetConfig)

__all__ = ["FoldingNet", "FoldingNetConfig"]
