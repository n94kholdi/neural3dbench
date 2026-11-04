from models.registry import ModelRegistry

from .model import PCN, PCNConfig

ModelRegistry.register("pcn", PCN, PCNConfig)

__all__ = ["PCN", "PCNConfig"]
