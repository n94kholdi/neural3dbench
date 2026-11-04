# neural3d-advisor

`neural3d-advisor` is a [Model Context Protocol](https://modelcontextprotocol.io)
server that helps choose and size 3D architectures in Neural3DBench. It explains
inductive biases and trade-offs, estimates GPU memory, and ranks models for a
problem description.

It discovers models at runtime instead of relying on a fixed list:

| Source | What it provides |
| --- | --- |
| `models.ModelRegistry` | Registered names, aliases (`pointnet++`), classes, docstrings, default configs, parameter counts, required/optional `ModelInput` fields |
| `docs/<model_name>.md` | Summary, section headings, tensor contracts (`[B, N, 3]` lines), reference links |
| `examples/**/*.py` | Runnable scripts whose file name contains the model name or whose source uses the model class |
| `catalog.yaml` | Curated metadata: paradigm, inductive biases, merits, demerits, Big-O complexity, memory model, speed and expressivity scores |

A model registered without a catalog entry is still listed. It is marked
`uncurated` and gets conservative defaults derived from its config (width,
depth, task, representation) and docstrings, including a quadratic-memory
warning when its docs mention k-NN.

Registry introspection runs in a fresh Python subprocess, so
`refresh_model_catalog` picks up new or edited model packages, docs and catalog
entries without restarting the server.

## Install

From the repository root, in the environment that has `torch` installed:

```bash
python -m pip install -e ".[advisor]"
```

This installs `mcp` and `pyyaml`. The server works with `mcp` 1.x (`FastMCP`)
and 2.x (`MCPServer`).

Run it manually to check that it starts (it waits for a client on stdio):

```bash
python -m tools.mcp_advisor            # from the repository root
python /abs/path/to/Neural3DBench/tools/mcp_advisor/server.py   # from anywhere
```

## Tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `list_supported_models` | `representation=None`, `task=None` | Models with status, representations, tasks, complexity |
| `get_model_dossier` | `model_name` | Paradigm, inductive biases, merits, demerits, tensor contract, complexity, default config, docs, references, examples |
| `recommend_architecture` | `problem_description`, `input_format`, `max_elements_per_sample`, `dynamic_topology=False`, `target_metric="accuracy"` | `verdict` (`fit`, `weak_fit`, `no_fit`); ranked models with score, matched tasks, rationale, pitfalls, estimated VRAM and hardware tier. On `no_fit`: `missing_capability`, `reason` and `suggested_networks` (name, why, paper) |
| `estimate_resource_footprint` | `model_name`, `num_elements`, `batch_size=1`, `precision="fp32"` | Training and inference memory, hardware tier, scaling when N doubles, assumptions |
| `refresh_model_catalog` | none | Sync report: added/removed models, uncurated models, models without docs, import errors |
| `audit_problem_assumptions` | `problem_description`, `model_name=None`, `boundary_conditions="none"`, `topology_changes=False`, `geometry_varies=False`, `evaluate_at_unseen_resolution=False` | Per-model findings (BC enforcement, topology, per-instance fitting, resolution transfer, FFT periodicity) and the models without errors |
| `check_symmetry_and_invariance` | `model_name`, `required_symmetries`, `target_kind="scalar"`, `uses_absolute_coordinates=True` | Per-symmetry verdict (permutation, translation, rotation) and the fix; vector/tensor targets are told they need equivariance |
| `recommend_representation_pipeline` | `raw_format`, `target_model`, `watertight=None`, `voxel_resolution=64` | Conversion steps using the repository converters and adapters, with voxelisation and connectivity warnings |
| `calculate_effective_receptive_field` | `model_name`, `depth=None`, `element_spacing=None`, `interaction_length=None` | Reach in hops/voxels and physical units; warning when it is shorter than the coupling length |
| `recommend_batching_strategy` | `model_name`, `num_elements`, `vram_budget_gb`, `target_effective_batch=16`, `variable_size=True` | Micro-batch, gradient accumulation, fp32/bf16 choice, collation (block-diagonal graphs, padded points) |
| `estimate_training_budget` | `model_name`, `num_elements`, `dataset_size`, `epochs`, `target_gpu="A100"`, `precision="fp32"` | Wall-clock hours with a 0.5x-4x range (H100, A100, RTX4090, V100, T4) |
| `generate_benchmark_config` | `model_name`, `dataset_name`, `num_elements`, `hardware_target="A100"`, `dataset_size=1000`, `epochs=100` | YAML config and a `ModelRegistry.create()` launcher with batch, precision and optimizer filled in |
| `suggest_ablation_matrix` | `primary_model`, `max_contenders=4` | Baselines with the role each plays (simpler model, other interaction mechanism, other representation) |

The guardrail tools read four extra catalog fields: `symmetries`, `receptive_field`,
`resolution_transfer` and the optional `boundary_conditions`. Training time treats a
step as memory-bandwidth bound and is not calibrated on real GPUs; time a pilot epoch.

Vocabulary:

- Representations: `point_cloud`, `mesh_graph`, `voxel_grid`, `continuous_coords`.
  Aliases such as `mesh`, `voxel`, `points` and the repository codes
  (`POINT_CLOUD`, `GRAPH`, `GRID`, `PARAMETRIC_SURFACE`) are accepted.
- Tasks: `surrogate_simulation`, `pde_operator`, `segmentation_classification`,
  `field_fitting`.
- Precision: `fp32`, `fp16`, `bf16`, `amp`.
- Hardware tiers: `<8GB`, `16-24GB`, `Multi-GPU / >24GB`.

### How the memory estimate works

Training peak in GB is the sum of three terms:

- Adam state: 16 bytes per parameter. The parameter count comes from the
  default config.
- Activations: `B * (a*N + q*N^2 + c) * bytes_per_float * 1.2`.
- CUDA context: 0.5 GB.

The constants `a`, `q` and `c` are the `memory_model` of each catalog entry,
in floats per sample. They were calibrated by counting autograd-saved tensors in
a forward/backward pass at the default config. Graphs used about 6 directed
edges per node. Inference keeps a quarter of the activations.

These figures are estimates, not measurements of your workload. Check them
with `torch.cuda.max_memory_allocated()`. For voxel models, `num_elements` is
the voxel count (`R^3`).

### How recommendations are ranked

Each model gets a score from these terms:

| Signal | Score |
| --- | --- |
| Native input format | +3 |
| Input usable after conversion (`also_accepts`, e.g. voxelization) | +1 |
| Input format not supported | model excluded |
| Task inferred from the description matches | +2 |
| Inferred task does not match | -1 |
| Catalog keywords found in the description | up to +1.5 |
| `dynamic_topology=True` and the model handles it | +1.5 |
| `dynamic_topology=True` and the model needs a fixed geometry | -2 |
| `0.5 * speed` or `0.5 * expressivity`, from `target_metric` | +0.5 to +2.5 |
| Estimated training peak above 24 GB | -3, with a VRAM pitfall |
| Estimated training peak of 8-24 GB | -1 |

## Register with Claude Desktop

Edit `claude_desktop_config.json`:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "neural3d-advisor": {
      "command": "/abs/path/to/Neural3DBench/.venv/bin/python",
      "args": ["/abs/path/to/Neural3DBench/tools/mcp_advisor/server.py"]
    }
  }
}
```

Use the Python interpreter of the environment where `torch` and the `advisor`
extra are installed, then restart Claude Desktop.

## Register with Cursor

Create `.cursor/mcp.json` in the repository (or `~/.cursor/mcp.json` for all projects):

```json
{
  "mcpServers": {
    "neural3d-advisor": {
      "command": "${workspaceFolder}/.venv/bin/python",
      "args": ["${workspaceFolder}/tools/mcp_advisor/server.py"]
    }
  }
}
```

## Register with Claude Code

```bash
claude mcp add neural3d-advisor -- /abs/path/to/.venv/bin/python /abs/path/to/Neural3DBench/tools/mcp_advisor/server.py
```

## Adding a model

1. Register the model in `ModelRegistry` and write `docs/<model_name>.md`.
   It now appears as `uncurated` after `refresh_model_catalog`.
2. Add a `catalog.yaml` entry keyed by the registry name. The field reference
   is at the top of the file. To calibrate `memory_model.per_element`, count the
   bytes saved for backward at two input sizes with
   `torch.autograd.graph.saved_tensors_hooks`.
3. Run `pytest tests/test_advisor_mcp.py`. It fails while any registered model
   is uncurated.

Variants that reuse a registered class can be added as catalog-only entries
with `registry_name` (see `adaptive_meshgraphnet`).

Models registered outside the `models` package can be included with a
comma-separated list of modules to import:

```bash
NEURAL3D_ADVISOR_EXTRA_MODULES=my_lab.models,other.pkg python tools/mcp_advisor/server.py
```
