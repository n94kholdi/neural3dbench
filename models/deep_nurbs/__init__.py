from models.registry import ModelRegistry

from .model import DeepNURBS, DeepNURBSConfig

ModelRegistry.register("deep_nurbs", DeepNURBS, DeepNURBSConfig)

__all__ = ["DeepNURBS", "DeepNURBSConfig"]