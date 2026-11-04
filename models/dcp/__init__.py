from models.registry import ModelRegistry

from .model import DCP, DCPConfig

ModelRegistry.register("dcp", DCP, DCPConfig)

__all__ = ["DCP", "DCPConfig"]
