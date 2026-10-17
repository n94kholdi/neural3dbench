from models.registry import ModelRegistry

from .block import TransolverBlock
from .components import FeedForward, InputEmbedding, OutputHead
from .model import Transolver, TransolverConfig
from .physics_attention import PhysicsAttention

ModelRegistry.register("transolver", Transolver, TransolverConfig)

__all__ = [
    "FeedForward",
    "InputEmbedding",
    "OutputHead",
    "PhysicsAttention",
    "Transolver",
    "TransolverBlock",
    "TransolverConfig",
]
