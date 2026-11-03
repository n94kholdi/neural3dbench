from models.registry import ModelRegistry

from .model import MLP, MLPConfig

ModelRegistry.register("mlp", MLP, MLPConfig)

__all__ = ["MLP", "MLPConfig"]
