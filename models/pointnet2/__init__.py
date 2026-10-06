from models.registry import ModelRegistry

from .components import (
    FeaturePropagation,
    PointNet2Encoder,
    PointNet2SegmentationHead,
    SetAbstraction,
    farthest_point_sample,
    query_ball_point,
)
from .model import PointNet2, PointNet2Config

ModelRegistry.register("pointnet2", PointNet2, PointNet2Config)
ModelRegistry.register("pointnet++", PointNet2, PointNet2Config)

__all__ = [
    "FeaturePropagation",
    "PointNet2",
    "PointNet2Config",
    "PointNet2Encoder",
    "PointNet2SegmentationHead",
    "SetAbstraction",
    "farthest_point_sample",
    "query_ball_point",
]
