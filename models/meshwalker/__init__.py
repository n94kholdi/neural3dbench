from models.registry import ModelRegistry

from .model import MeshWalker, MeshWalkerConfig, random_walks

ModelRegistry.register("meshwalker", MeshWalker, MeshWalkerConfig)

__all__ = ["MeshWalker", "MeshWalkerConfig", "random_walks"]
