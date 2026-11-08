from models.registry import ModelRegistry

from .model import FaceKernelCorrelation, FaceRotateConvolution, MeshConvolution, MeshNet, MeshNetConfig

ModelRegistry.register("meshnet", MeshNet, MeshNetConfig)

__all__ = ["FaceKernelCorrelation", "FaceRotateConvolution", "MeshConvolution", "MeshNet", "MeshNetConfig"]
