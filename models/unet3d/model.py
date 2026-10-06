from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.types import ModelCapabilities, ModelInput, ModelOutput

from .components import DecoderBlock3D, EncoderBlock3D, VolumetricOutputHead


@dataclass
class UNet3DConfig(BaseModelConfig):
    """Configuration for dense prediction on structured 3D volumes."""

    input_channels: int = 1
    output_channels: int = 1
    base_channels: int = 32
    levels: int = 4
    channel_multiplier: int = 2
    kernel_size: int = 3
    normalization: str = "batch"
    activation: str = "relu"
    downsampling: str = "max_pool"
    upsampling: str = "transpose_conv"
    representation: str = "GRID"

    def __post_init__(self) -> None:
        self.name = "unet3d"
        self.model_type = "unet3d"
        self.representation = "GRID"
        self.normalization = self.normalization.lower()
        self.activation = self.activation.lower()
        self.downsampling = self.downsampling.lower()
        self.upsampling = self.upsampling.lower()
        for field_name in ("input_channels", "output_channels", "base_channels"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if self.levels < 2:
            raise ValueError("levels must be at least 2.")
        if self.channel_multiplier < 1:
            raise ValueError("channel_multiplier must be at least 1.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if self.normalization not in {"batch", "instance", "group", "none"}:
            raise ValueError("normalization must be batch, instance, group, or none.")
        if self.activation not in {"relu", "leaky_relu", "elu", "gelu", "silu"}:
            raise ValueError("Unsupported activation.")
        if self.downsampling not in {"max_pool", "strided_conv"}:
            raise ValueError("downsampling must be max_pool or strided_conv.")
        if self.upsampling not in {"transpose_conv", "interpolate"}:
            raise ValueError("upsampling must be transpose_conv or interpolate.")


class UNet3D(BaseNetwork):
    """Encoder-decoder 3D U-Net producing raw voxel-wise logits or values."""

    config_class = UNet3DConfig

    def __init__(self, config: UNet3DConfig | None = None):
        super().__init__(config=config or UNet3DConfig())
        channels = [
            self.config.base_channels * self.config.channel_multiplier**level
            for level in range(self.config.levels)
        ]
        encoders: list[nn.Module] = []
        in_channels = self.config.input_channels
        for index, out_channels in enumerate(channels):
            encoders.append(
                EncoderBlock3D(
                    in_channels,
                    out_channels,
                    kernel_size=self.config.kernel_size,
                    normalization=self.config.normalization,
                    activation=self.config.activation,
                    downsampling=(
                        self.config.downsampling
                        if index < self.config.levels - 1
                        else None
                    ),
                )
            )
            in_channels = out_channels
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(
            DecoderBlock3D(
                channels[index],
                channels[index - 1],
                kernel_size=self.config.kernel_size,
                normalization=self.config.normalization,
                activation=self.config.activation,
                upsampling=self.config.upsampling,
            )
            for index in range(self.config.levels - 1, 0, -1)
        )
        self.head = VolumetricOutputHead(channels[0], self.config.output_channels)

    @property
    def required_inputs(self) -> set[str]:
        return {"voxel_fields"}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            representation="GRID",
            requires={"voxel_fields"},
            supports={"boundary_mask", "material_properties"},
            variable_geometry=False,
            variable_node_count=False,
        )

    def validate_inputs(self, inputs: ModelInput) -> bool:
        super().validate_inputs(inputs)
        volume = self._ensure_tensor(inputs.voxel_fields, name="voxel_fields")
        if volume.ndim != 5:
            raise ValueError(
                "UNet3D voxel_fields must have shape [B, C, D, H, W], "
                f"got {tuple(volume.shape)}."
            )
        if volume.shape[1] != self.config.input_channels:
            raise ValueError(
                f"UNet3D expects {self.config.input_channels} input channels, "
                f"got {volume.shape[1]}."
            )
        if not volume.is_floating_point():
            raise TypeError("voxel_fields must use a floating-point dtype.")
        minimum_size = 2 ** (self.config.levels - 1)
        if any(size < minimum_size for size in volume.shape[-3:]):
            raise ValueError(
                f"Each spatial dimension must be at least {minimum_size} for "
                f"{self.config.levels} levels."
            )
        return True

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        parameter = next(self.parameters())
        hidden = torch.as_tensor(
            inputs.voxel_fields, device=parameter.device, dtype=parameter.dtype
        )
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            skip, hidden = encoder(hidden)
            skips.append(skip)

        bottleneck = skips[-1]
        hidden = bottleneck
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            hidden = decoder(hidden, skip)
        predictions = self.head(hidden)
        return ModelOutput(
            predictions=predictions,
            latent=bottleneck,
            metadata={"spatial_shape": tuple(predictions.shape[-3:])},
        )
