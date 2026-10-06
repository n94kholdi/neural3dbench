from models.registry import ModelRegistry

from .components import (
    PointNetClassificationHead,
    PointNetEncoder,
    PointNetSegmentationHead,
    SharedPointMLP,
    TransformNet,
    feature_transform_regularizer,
)
from .model import PointNet, PointNetConfig

ModelRegistry.register("pointnet", PointNet, PointNetConfig)

__all__ = [
    "PointNet",
    "PointNetClassificationHead",
    "PointNetConfig",
    "PointNetEncoder",
    "PointNetSegmentationHead",
    "SharedPointMLP",
    "TransformNet",
    "feature_transform_regularizer",
]
