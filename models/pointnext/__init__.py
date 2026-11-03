from models.registry import ModelRegistry

from .components import (
    InvResMLP,
    PointNeXtEncoder,
    PointNeXtSegmentationHead,
    PointNeXtSetAbstraction,
)
from .model import PointNeXt, PointNeXtConfig

ModelRegistry.register("pointnext", PointNeXt, PointNeXtConfig)

__all__ = [
    "InvResMLP",
    "PointNeXt",
    "PointNeXtConfig",
    "PointNeXtEncoder",
    "PointNeXtSegmentationHead",
    "PointNeXtSetAbstraction",
]
