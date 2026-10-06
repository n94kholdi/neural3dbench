from models.registry import ModelRegistry

from .components import (
    ConvBlock3D,
    DecoderBlock3D,
    EncoderBlock3D,
    VolumetricOutputHead,
)
from .model import UNet3D, UNet3DConfig

ModelRegistry.register("unet3d", UNet3D, UNet3DConfig)

__all__ = [
    "ConvBlock3D",
    "DecoderBlock3D",
    "EncoderBlock3D",
    "UNet3D",
    "UNet3DConfig",
    "VolumetricOutputHead",
]
