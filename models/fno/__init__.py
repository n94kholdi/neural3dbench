from models.registry import ModelRegistry

from .components import FourierOperatorBlock3d, ProjectionHead3d, normalized_coordinate_grid
from .model import FNO3D, FNO3DConfig
from .spectral import SpectralConv3d

ModelRegistry.register("fno3d", FNO3D, FNO3DConfig)

__all__ = [
    "FNO3D",
    "FNO3DConfig",
    "FourierOperatorBlock3d",
    "ProjectionHead3d",
    "SpectralConv3d",
    "normalized_coordinate_grid",
]
