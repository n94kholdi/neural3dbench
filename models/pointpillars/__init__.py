from models.registry import ModelRegistry

from .model import PointPillars, PointPillarsConfig

ModelRegistry.register("pointpillars", PointPillars, PointPillarsConfig)

__all__ = ["PointPillars", "PointPillarsConfig"]
