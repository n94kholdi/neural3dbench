# Neural3DBench agent rules

## Adding a 3D architecture

When implementing a new 3D architecture in `neural3dbench`:

1. Subclass `models.BaseNetwork` (with a `BaseModelConfig` dataclass) and register it with
   `ModelRegistry.register("<model_name>", Model, ModelConfig)` in the model package `__init__.py`.
   Give the model and config classes docstrings; the advisor reads them.
2. Create the corresponding markdown documentation in `docs/<model_name>.md`, including an
   input/output contract with shapes such as `[B, N, 3]` and reference links.
3. Add an entry to `tools/mcp_advisor/catalog.yaml` keyed by the registry name, specifying its
   representations, tasks, complexity, memory model, merits, demerits and inductive biases
   (field reference at the top of the file).
4. Verify discovery with the `tools/mcp_advisor/` test suite:
   `pytest tests/test_advisor_mcp.py`. It fails while any registered model has no catalog entry.

## Conventions

- Follow `docs/adding_a_network.md` for the model, configuration and registry pattern.
- Run `pytest` before committing.
