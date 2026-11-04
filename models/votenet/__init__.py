from models.registry import ModelRegistry

from .model import VoteNet, VoteNetConfig

ModelRegistry.register("votenet", VoteNet, VoteNetConfig)

__all__ = ["VoteNet", "VoteNetConfig"]
