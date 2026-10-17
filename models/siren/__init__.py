from models.registry import ModelRegistry

from .layers import SineLayer
from .model import SIREN, SIRENConfig

ModelRegistry.register("siren", SIREN, SIRENConfig)

__all__ = ["SineLayer", "SIREN", "SIRENConfig"]
