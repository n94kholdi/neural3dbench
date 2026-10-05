from __future__ import annotations

import inspect
from typing import Any, get_args, get_origin, Union

from .base import BaseNetwork
from .configs import BaseModelConfig


class ModelRegistry:
    _registry: dict[str, type[BaseNetwork]] = {}
    _config_types: dict[str, type[BaseModelConfig]] = {}

    @classmethod
    def register(cls, name: str, model_class: type[BaseNetwork], config_class: type[BaseModelConfig] | None = None) -> None:
        if not isinstance(model_class, type) or not issubclass(model_class, BaseNetwork):
            raise TypeError("Registered model must be a subclass of BaseNetwork.")
        if name in cls._registry:
            raise ValueError(f"Model '{name}' is already registered.")

        if config_class is None:
            config_class = cls._infer_config_class(model_class)

        cls._registry[name] = model_class
        cls._config_types[name] = config_class

    @classmethod
    def get(cls, name: str) -> type[BaseNetwork]:
        try:
            return cls._registry[name]
        except KeyError as exc:
            raise KeyError(f"Unknown model '{name}'. Available models: {sorted(cls._registry)}") from exc

    @classmethod
    def get_config_class(cls, name: str) -> type[BaseModelConfig]:
        return cls._config_types.get(name, BaseModelConfig)

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | BaseModelConfig | None = None, **kwargs: Any) -> BaseNetwork:
        model_class = cls.get(name)
        config_type = cls.get_config_class(name)

        if config is None:
            config = kwargs
        if isinstance(config, dict):
            if not config:
                config = {}
            config = config_type.from_dict(config)
        elif config is not None and not isinstance(config, BaseModelConfig):
            raise TypeError("Model config must be a BaseModelConfig or dict.")

        return model_class(config=config)

    @classmethod
    def _infer_config_class(cls, model_class: type[BaseNetwork]) -> type[BaseModelConfig]:
        try:
            signature = inspect.signature(model_class.__init__)
        except (TypeError, ValueError):
            return BaseModelConfig

        config_param = signature.parameters.get("config")
        if config_param is None:
            return BaseModelConfig

        annotation = config_param.annotation
        if annotation in (inspect._empty, Any):
            return BaseModelConfig

        origin = get_origin(annotation)
        if origin in (Union, getattr(__import__("types"), "UnionType", None)):
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if args:
                annotation = args[0]

        if isinstance(annotation, type) and issubclass(annotation, BaseModelConfig):
            return annotation

        return BaseModelConfig


def create_model(config: dict[str, Any] | BaseModelConfig | str | None = None, **kwargs: Any) -> BaseNetwork:
    if isinstance(config, BaseNetwork):
        return config
    if isinstance(config, str):
        return ModelRegistry.create(config, **kwargs)
    if config is None:
        raise ValueError("A model name or config must be provided.")
    if isinstance(config, BaseModelConfig):
        model_name = config.name or config.model_type
        if not model_name:
            raise ValueError("Model config must include a model name.")
        return ModelRegistry.create(model_name, config=config, **kwargs)
    if not isinstance(config, dict):
        raise TypeError("Config must be a dict, BaseModelConfig, or model name string.")

    model_name = config.get("name")
    if not model_name:
        raise ValueError("Model config must include a 'name' field.")
    return ModelRegistry.create(model_name, config=config, **kwargs)
