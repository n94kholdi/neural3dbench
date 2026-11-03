"""Dump ModelRegistry contents as JSON on stdout.

Run in a fresh interpreter by ``Advisor.refresh`` so that models added or edited
after the MCP server started are picked up without reloading modules in-process.

Usage: python _introspect.py REPO_ROOT [EXTRA_MODULE ...]
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import pkgutil
import sys
import traceback


def _first_registration(registry: dict[str, type]) -> dict[type, str]:
    canonical: dict[type, str] = {}
    for name, cls in registry.items():
        canonical.setdefault(cls, name)
    return canonical


def _instantiate(registry_cls, name: str):
    import torch

    # Meta tensors avoid allocating weights; some models read values in __init__.
    try:
        with torch.device("meta"):
            return registry_cls.create(name)
    except Exception:
        return registry_cls.create(name)


def main(argv: list[str]) -> dict:
    repo_root, extra_modules = argv[0], argv[1:]
    sys.path.insert(0, repo_root)
    import_errors: dict[str, str] = {}

    import models
    from models import ModelRegistry

    # Import every subpackage so models registered outside models/__init__.py are seen.
    for info in pkgutil.walk_packages(models.__path__, "models."):
        try:
            importlib.import_module(info.name)
        except Exception as exc:
            import_errors[info.name] = f"{type(exc).__name__}: {exc}"
    for module_name in extra_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            import_errors[module_name] = f"{type(exc).__name__}: {exc}"

    registry = ModelRegistry._registry
    canonical = _first_registration(registry)
    entries: dict[str, dict] = {}
    for name, cls in registry.items():
        if canonical[cls] != name:
            entries[canonical[cls]]["aliases"].append(name)
            continue
        config_cls = ModelRegistry.get_config_class(name)
        entry = {
            "class_name": cls.__name__,
            "module": cls.__module__,
            "source_file": inspect.getsourcefile(cls),
            "docstring": inspect.cleandoc(cls.__doc__) if cls.__doc__ else "",
            "config_class": config_cls.__name__,
            "config_docstring": inspect.cleandoc(config_cls.__doc__) if config_cls.__doc__ else "",
            "aliases": [],
            "config_defaults": {},
            "parameter_count": None,
            "required_inputs": [],
            "supported_inputs": [],
            "representation": None,
            "instantiation_error": None,
        }
        try:
            config = config_cls()
            if dataclasses.is_dataclass(config):
                entry["config_defaults"] = json.loads(json.dumps(dataclasses.asdict(config), default=str))
            entry["representation"] = getattr(config, "representation", None)
            model = _instantiate(ModelRegistry, name)
            entry["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
            entry["required_inputs"] = sorted(model.required_inputs)
            capabilities = model.capabilities
            entry["supported_inputs"] = sorted(capabilities.supports)
            entry["representation"] = capabilities.representation or entry["representation"]
        except Exception as exc:
            entry["instantiation_error"] = f"{type(exc).__name__}: {exc}"
        entries[name] = entry

    return {"models": entries, "import_errors": import_errors}


if __name__ == "__main__":
    # Keep stray prints from model modules out of the JSON channel.
    json_out, sys.stdout = sys.stdout, sys.stderr
    try:
        result = main(sys.argv[1:])
    except Exception:
        result = {"models": {}, "import_errors": {"models": traceback.format_exc()}}
    json_out.write(json.dumps(result))
