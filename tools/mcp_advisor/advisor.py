"""Advisor core: merges live ModelRegistry introspection, docs/ and catalog.yaml.

This module has no torch or MCP dependency. Registry introspection runs in a
subprocess (``_introspect.py``) so ``refresh()`` sees models added after start-up.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = Path(__file__).with_name("catalog.yaml")
INTROSPECT_SCRIPT = Path(__file__).with_name("_introspect.py")

REPRESENTATIONS = ("point_cloud", "mesh_graph", "voxel_grid", "continuous_coords")
TASKS = ("surrogate_simulation", "pde_operator", "segmentation_classification", "field_fitting", "shape_completion", "object_detection",
         "registration")
REQUIRED_CATALOG_FIELDS = (
    "display_name",
    "paradigm",
    "representations",
    "tasks",
    "element",
    "inductive_biases",
    "merits",
    "demerits",
    "complexity",
    "memory_model",
    "speed",
    "expressivity",
    "dynamic_topology",
)

# Repository representation codes (BaseModelConfig.representation) -> advisor vocabulary.
_REPO_REPRESENTATIONS = {
    "POINT_CLOUD": "point_cloud",
    "GRAPH": "mesh_graph",
    "MESH_GRAPH": "mesh_graph",
    "MESH": "mesh_graph",
    "GRID": "voxel_grid",
    "VOXEL_GRID": "voxel_grid",
    "PARAMETRIC_SURFACE": "continuous_coords",
}
_REPRESENTATION_ALIASES = {
    **{code.lower(): rep for code, rep in _REPO_REPRESENTATIONS.items()},
    **{rep: rep for rep in REPRESENTATIONS},
    "points": "point_cloud",
    "pointcloud": "point_cloud",
    "mesh": "mesh_graph",
    "graph": "mesh_graph",
    "voxel": "voxel_grid",
    "voxels": "voxel_grid",
    "grid": "voxel_grid",
    "coords": "continuous_coords",
    "coordinates": "continuous_coords",
    "implicit": "continuous_coords",
}
_TASK_KEYWORDS = {
    "segmentation_classification": ("classif", "segment", "label", "recogni", "parts"),
    "pde_operator": ("pde", "operator", "solver", "darcy", "navier", "poisson", "parametric", "super-resolution"),
    "surrogate_simulation": ("simulat", "surrogate", "cfd", "fea", "fem", "rollout", "transient", "dynamics", "time-step"),
    "field_fitting": ("pinn", "implicit", "sdf", "fit", "neural field", "reconstruct"),
    "shape_completion": ("complet", "partial scan", "missing part", "upsampl", "densif"),
    "object_detection": ("detect", "bounding box", "3d box", "localiz"),
    "registration": ("registr", "align two", "alignment", "pose estimat", "icp", "correspondence"),
}

# Capabilities no catalog task covers. A catalog entry may list one of these keys under ``tasks``
# once a model implements it; until then the advisor answers "no_fit" instead of a wrong model.
MISSING_CAPABILITIES = {
    "instance_segmentation": {
        "keywords": ("instance",),
        "needs": "A separate id per object instance, not only per class: offset/embedding head and clustering.",
        "suggested_datasets": ["ScanNet (instance labels)", "S3DIS"],
        "suggested_networks": [
            {"name": "PointGroup", "why": "Semantic + center-offset branches, clustered into instances.", "reference": "https://arxiv.org/abs/2004.01658"},
        ],
    },
    "shape_generation": {
        # "diffusion model", not "diffusion": heat diffusion and DiffusionNet are not generation.
        "keywords": ("generat", "synthes", "diffusion model", "sample new shape"),
        "needs": "A generative model over point sets (flow, diffusion or GAN), trained without per-point targets.",
        "suggested_datasets": ["ShapeNet"],
        "suggested_networks": [
            {"name": "PointFlow", "why": "Continuous normalizing flow over shapes and points.", "reference": "https://arxiv.org/abs/1906.12320"},
            {"name": "LION", "why": "Latent point diffusion; current quality baseline.", "reference": "https://arxiv.org/abs/2210.06978"},
        ],
    },
    "scene_flow": {
        "keywords": ("scene flow", "motion estimat", "optical flow"),
        "needs": "Two frames in, a 3D motion vector per point of frame 1 out: a cross-frame correlation layer.",
        "suggested_datasets": ["FlyingThings3D", "KITTI scene flow"],
        "suggested_networks": [
            {"name": "FlowNet3D", "why": "PointNet++ with a flow-embedding layer between frames.", "reference": "https://arxiv.org/abs/1806.01411"},
        ],
    },
}

# What a model's heads predict (catalog ``outputs``); problem_schema answers set ``required_output``.
OUTPUTS = {
    "shape_label": "one class per shape",
    "point_label": "a class per point / vertex / voxel",
    "point_field": "a continuous field per point (regression)",
    "point_set": "a new point set (completion / reconstruction)",
    "boxes": "3D bounding boxes",
    "rigid_pose": "a rigid transform between two clouds",
    "deformation_field": "a non-rigid displacement per point between two clouds",
}
# Networks and datasets to add when no registered model gives the required output for a task.
# Keyed by (task, output) or (task, output, representation); checked against the papers on 2026-11-09.
GAP_SUGGESTIONS: dict[tuple[str, ...], dict[str, list]] = {
    ("registration", "deformation_field"): {
        "networks": [{"name": "Neural Deformation Pyramid (NDP)", "why": "Hierarchical MLP motion field for non-rigid point cloud registration (NeurIPS 2022).", "reference": "https://arxiv.org/abs/2205.12796"}],
        "datasets": ["4DMatch / 4DLoMatch"],
    },
    ("pde_operator", "point_field", "continuous_coords"): {
        "networks": [{"name": "DeepONet", "why": "Branch net for the input function, trunk net for query coordinates (Nature Machine Intelligence 2021).", "reference": "https://arxiv.org/abs/1910.03193"}],
        "datasets": ["Darcy flow", "Burgers"],
    },
    ("shape_completion", "point_set", "voxel_grid"): {
        "networks": [{"name": "3D-EPN", "why": "Volumetric encoder-predictor CNN for shape completion (CVPR 2017).", "reference": "https://arxiv.org/abs/1612.00101"}],
        "datasets": ["ShapeNet (3D-EPN partial scans)"],
    },
    ("object_detection", "boxes", "voxel_grid"): {
        "networks": [{"name": "VoxelNet", "why": "Voxel feature encoding + 3D convolutions + region proposal network (CVPR 2018).", "reference": "https://arxiv.org/abs/1711.06396"}],
        "datasets": ["KITTI"],
    },
    ("segmentation_classification", "shape_label", "voxel_grid"): {
        "networks": [{"name": "3D ShapeNets", "why": "Volumetric CNN shape classifier (CVPR 2015).", "reference": "https://arxiv.org/abs/1406.5670"}],
        "datasets": ["ModelNet40 (voxelized)"],
    },
}

BYTES_PER_FLOAT = {"fp32": 4, "fp16": 2, "bf16": 2, "amp": 2}
# Adam training state per parameter: weights, gradients, two moments (fp32).
TRAIN_BYTES_PER_PARAM = 16
# Allocator fragmentation and temporary buffers on top of stored activations.
ACTIVATION_OVERHEAD = 1.2
# Inference keeps only a few live activations instead of the whole autograd graph.
INFERENCE_ACTIVATION_FRACTION = 0.25
CUDA_CONTEXT_GB = 0.5
GB = 1024**3
_TIERS = ((8, "<8GB (consumer GPU)"), (24, "16-24GB (workstation GPU)"))

_SHAPE = re.compile(r"\[(?:B|N|E|Q)\b[^\]\n]*\]")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def normalize_representation(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _REPRESENTATION_ALIASES:
        raise ValueError(f"Unknown representation '{value}'. Use one of {list(REPRESENTATIONS)}.")
    return _REPRESENTATION_ALIASES[key]


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def parse_markdown_doc(text: str) -> dict[str, Any]:
    """Extract title, summary, section headings, tensor contracts and references."""
    lines = text.splitlines()
    title = next((line[2:].strip() for line in lines if line.startswith("# ")), None)

    summary_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("```"):
            if summary_lines:
                break
            continue
        if not stripped:
            if summary_lines:
                break
            continue
        summary_lines.append(stripped)

    contracts: list[str] = []
    for line in lines:
        if _SHAPE.search(line):
            entry = re.sub(r"\s+", " ", line.strip().lstrip("-* ").strip())
            if entry and entry not in contracts:
                contracts.append(entry)

    references: list[dict[str, str]] = []
    for label, url in _LINK.findall(text):
        if not any(ref["url"] == url for ref in references):
            references.append({"title": re.sub(r"\s+", " ", label), "url": url})
    for line in lines:
        if "et al." in line and not _LINK.search(line):
            references.append({"title": line.strip(), "url": ""})

    return {
        "title": title,
        "summary": " ".join(summary_lines),
        "sections": [line.lstrip("#").strip() for line in lines if line.startswith("## ")],
        "tensor_contracts": contracts[:16],
        "references": references[:10],
    }


def _fallback_entry(name: str, info: dict[str, Any], doc: dict[str, Any] | None) -> dict[str, Any]:
    """Conservative metadata for a registered model that has no catalog entry."""
    defaults = info.get("config_defaults", {})
    rep = _REPO_REPRESENTATIONS.get(str(info.get("representation") or "").upper())
    task = str(defaults.get("task", "")).lower()
    tasks = ["segmentation_classification"] if task in ("classification", "segmentation") else []

    width = next(
        (defaults[key] for key in ("hidden_dim", "latent_dim", "width", "base_channels", "embedding_dim")
         if isinstance(defaults.get(key), int)),
        128,
    )
    depth = next(
        (defaults[key] for key in ("num_layers", "processor_steps", "num_hidden_layers", "levels")
         if isinstance(defaults.get(key), int)),
        len(defaults["widths"]) if isinstance(defaults.get("widths"), list) else 4,
    )
    text = " ".join([info.get("docstring", ""), info.get("config_docstring", ""), (doc or {}).get("summary", "")])
    uses_knn = bool(re.search(r"k-?nn|nearest neighbo", text, re.IGNORECASE))
    first_line = (info.get("docstring") or "").split("\n")[0] or f"{info.get('class_name', name)} (no docstring)"

    return {
        "display_name": info.get("class_name", name),
        "paradigm": first_line,
        "representations": [rep] if rep else [],
        "also_accepts": {},
        "tasks": tasks,
        "element": "element",
        "inductive_biases": ["Not curated - add an entry to tools/mcp_advisor/catalog.yaml."],
        "merits": ["Not curated - see the model docstring and docs."],
        "demerits": ["Not curated - estimates below are generic defaults."],
        "complexity": {
            "time": "O(N^2) suspected (k-NN detected in docs)" if uses_knn else "unknown (assumed O(N))",
            "memory": "O(N^2) suspected" if uses_knn else "unknown (assumed O(N))",
            "notes": f"Generic estimate from width={width}, depth={depth} in the default config.",
        },
        # ~4 stored tensors (pre/post norm, activation, residual) per layer per element.
        "memory_model": {"per_element": 4 * width * depth, "per_element_sq": 1 if uses_knn else 0, "per_sample": 0},
        "speed": 3,
        "expressivity": 3,
        "dynamic_topology": "partial",
        "keywords": [],
    }


class Advisor:
    def __init__(
        self,
        repo_root: Path | str = REPO_ROOT,
        catalog_path: Path | str = CATALOG_PATH,
        docs_dir: Path | str | None = None,
        examples_dir: Path | str | None = None,
        extra_modules: tuple[str, ...] | list[str] | None = None,
        python: str = sys.executable,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.catalog_path = Path(catalog_path)
        self.docs_dir = Path(docs_dir) if docs_dir else self.repo_root / "docs"
        self.examples_dir = Path(examples_dir) if examples_dir else self.repo_root / "examples"
        if extra_modules is None:
            extra_modules = [m for m in os.environ.get("NEURAL3D_ADVISOR_EXTRA_MODULES", "").split(",") if m]
        self.extra_modules = list(extra_modules)
        self.python = python
        self._state: dict[str, Any] | None = None

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> dict[str, Any]:
        if self._state is None:
            self.refresh()
        return self._state

    def _introspect_registry(self) -> dict[str, Any]:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(self.repo_root), env.get("PYTHONPATH")]))
        proc = subprocess.run(
            [self.python, str(INTROSPECT_SCRIPT), str(self.repo_root), *self.extra_modules],
            capture_output=True,
            text=True,
            env=env,
            cwd=self.repo_root,
            timeout=300,
        )
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {"models": {}, "import_errors": {"introspection": proc.stderr[-4000:] or "no output"}}

    def _load_catalog(self) -> dict[str, dict[str, Any]]:
        with self.catalog_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        catalog = data.get("models") or {}
        for name, entry in catalog.items():
            missing = [key for key in REQUIRED_CATALOG_FIELDS if key not in entry]
            if missing:
                raise ValueError(f"catalog.yaml entry '{name}' is missing fields: {missing}")
            bad = set(entry["representations"]) - set(REPRESENTATIONS)
            if bad:
                raise ValueError(f"catalog.yaml entry '{name}' has unknown representations {sorted(bad)}")
        return catalog

    def _find_docs(self, names: list[str], explicit: list[str]) -> list[Path]:
        candidates = [self.repo_root / path for path in explicit]
        candidates += [self.docs_dir / f"{name}.md" for name in names]
        found: list[Path] = []
        for path in candidates:
            if path.is_file() and path not in found:
                found.append(path)
        return found

    def _find_examples(self, names: list[str], class_name: str | None, explicit: list[str]) -> list[str]:
        found = [path for path in explicit if (self.repo_root / path).exists()]
        if not self.examples_dir.is_dir():
            return found
        class_pattern = re.compile(rf"\b{re.escape(class_name)}\b") if class_name else None
        for path in sorted(self.examples_dir.rglob("*.py")):
            tokens = set(path.stem.lower().split("_"))
            matched = any(name.lower() in tokens for name in names)
            if not matched and class_pattern is not None:
                matched = bool(class_pattern.search(path.read_text(encoding="utf-8", errors="ignore")))
            rel = _relative(path, self.repo_root)
            if matched and rel not in found:
                found.append(rel)
        return found

    def refresh(self) -> dict[str, Any]:
        """Re-scan ModelRegistry, docs/ and catalog.yaml; return a sync report."""
        previous = set(self._state["models"]) if self._state else set()
        introspection = self._introspect_registry()
        registry: dict[str, dict[str, Any]] = introspection.get("models", {})
        catalog = self._load_catalog()

        models: dict[str, dict[str, Any]] = {}
        for name in list(registry) + [n for n in catalog if n not in registry]:
            entry = catalog.get(name)
            registry_name = (entry or {}).get("registry_name", name)
            info = registry.get(registry_name)
            if entry is not None and registry_name not in registry:
                # Catalog entry whose model is no longer (or not yet) registered.
                status = "catalog_only"
            else:
                status = "curated" if entry is not None else "uncurated"

            module_pkg = info["module"].split(".")[1] if info and info["module"].count(".") >= 1 else None
            names = [name, *(info or {}).get("aliases", []), *([module_pkg] if module_pkg else [])]
            docs = self._find_docs(names, (entry or {}).get("docs", []))
            parsed_docs = [parse_markdown_doc(path.read_text(encoding="utf-8")) for path in docs]
            primary_doc = parsed_docs[0] if parsed_docs else None
            metadata = entry if entry is not None else _fallback_entry(name, info or {}, primary_doc)

            models[name] = {
                "name": name,
                "status": status,
                "registry_name": registry_name if info else None,
                "aliases": (info or {}).get("aliases", []),
                "registry": info,
                "metadata": metadata,
                "docs": [_relative(path, self.repo_root) for path in docs],
                "parsed_docs": parsed_docs,
                "examples": self._find_examples(
                    names, (info or {}).get("class_name"), (entry or {}).get("examples", [])
                ),
            }

        self._state = {"models": models, "import_errors": introspection.get("import_errors", {})}
        current = set(models)
        return {
            "model_count": len(models),
            "registered": sorted(registry),
            "added_since_last_refresh": sorted(current - previous) if previous else [],
            "removed_since_last_refresh": sorted(previous - current),
            "uncurated_models": sorted(n for n, m in models.items() if m["status"] == "uncurated"),
            "catalog_only_models": sorted(
                n for n, m in models.items() if m["status"] == "catalog_only"
            ),
            "models_without_docs": sorted(n for n, m in models.items() if not m["docs"]),
            "import_errors": self._state["import_errors"],
        }

    def resolve(self, model_name: str) -> dict[str, Any]:
        models = self.state["models"]
        key = model_name.strip().lower()
        for name, model in models.items():
            if key == name.lower() or key in (alias.lower() for alias in model["aliases"]):
                return model
            if key == model["metadata"]["display_name"].lower():
                return model
        raise KeyError(f"Unknown model '{model_name}'. Available: {sorted(models)}")

    # ------------------------------------------------------------------ tools

    def list_supported_models(self, representation: str | None = None, task: str | None = None) -> list[dict]:
        representation = normalize_representation(representation)
        rows = []
        for name, model in self.state["models"].items():
            meta = model["metadata"]
            if representation and representation not in meta["representations"]:
                continue
            if task and task.lower() not in meta["tasks"]:
                continue
            rows.append({
                "name": name,
                "display_name": meta["display_name"],
                "aliases": model["aliases"],
                "status": model["status"],
                "representations": meta["representations"],
                "tasks": meta["tasks"],
                "complexity": meta["complexity"]["time"],
                "paradigm": meta["paradigm"],
            })
        return rows

    def get_model_dossier(self, model_name: str) -> dict[str, Any]:
        model = self.resolve(model_name)
        meta = model["metadata"]
        info = model["registry"] or {}
        doc = model["parsed_docs"][0] if model["parsed_docs"] else {}
        contracts = [c for parsed in model["parsed_docs"][:1] for c in parsed["tensor_contracts"]]
        return {
            "name": model["name"],
            "display_name": meta["display_name"],
            "status": model["status"],
            "aliases": model["aliases"],
            "paradigm": meta["paradigm"],
            "representations": meta["representations"],
            "also_accepts": meta.get("also_accepts", {}),
            "tasks": meta["tasks"],
            "inductive_biases": meta["inductive_biases"],
            "merits": meta["merits"],
            "demerits": meta["demerits"],
            "complexity": meta["complexity"],
            "dynamic_topology": meta["dynamic_topology"],
            "outputs": meta.get("outputs"),
            "evidence": meta.get("evidence"),
            "tensor_contract": {
                "required_inputs": info.get("required_inputs", []),
                "optional_inputs": info.get("supported_inputs", []),
                "repository_representation": info.get("representation"),
                "from_docs": contracts,
            },
            "implementation": {
                "class": info.get("class_name"),
                "module": info.get("module"),
                "source_file": _relative(Path(info["source_file"]), self.repo_root) if info.get("source_file") else None,
                "config_class": info.get("config_class"),
                "default_config": info.get("config_defaults", {}),
                "parameter_count_default": info.get("parameter_count"),
                "docstring": info.get("docstring", ""),
            },
            "docs": model["docs"],
            "doc_summary": doc.get("summary", ""),
            "doc_sections": doc.get("sections", []),
            "references": [ref for parsed in model["parsed_docs"] for ref in parsed["references"]][:10],
            "examples": model["examples"],
        }

    def estimate_resource_footprint(
        self,
        model_name: str,
        num_elements: int,
        batch_size: int = 1,
        precision: str = "fp32",
    ) -> dict[str, Any]:
        if num_elements <= 0 or batch_size <= 0:
            raise ValueError("num_elements and batch_size must be positive.")
        precision = precision.lower()
        if precision not in BYTES_PER_FLOAT:
            raise ValueError(f"precision must be one of {sorted(BYTES_PER_FLOAT)}.")

        model = self.resolve(model_name)
        meta = model["metadata"]
        mem = meta["memory_model"]
        n = float(num_elements)
        floats_per_sample = mem["per_element"] * n + mem.get("per_element_sq", 0) * n * n + mem.get("per_sample", 0)
        bytes_per_float = BYTES_PER_FLOAT[precision]
        activation_bytes = batch_size * floats_per_sample * bytes_per_float * ACTIVATION_OVERHEAD

        params = (model["registry"] or {}).get("parameter_count")
        param_bytes = (params or 0) * (4 if precision == "fp32" else bytes_per_float)
        train_gb = ((params or 0) * TRAIN_BYTES_PER_PARAM + activation_bytes) / GB + CUDA_CONTEXT_GB
        infer_gb = (param_bytes + activation_bytes * INFERENCE_ACTIVATION_FRACTION) / GB + CUDA_CONTEXT_GB

        def per_sample_floats(count: float) -> float:
            return mem["per_element"] * count + mem.get("per_element_sq", 0) * count**2 + mem.get("per_sample", 0)

        warnings = []
        if mem.get("per_element_sq", 0):
            warnings.append("Quadratic term present: memory grows ~4x when num_elements doubles at large N.")
        if params is None:
            warnings.append("Parameter count unavailable (model could not be instantiated); weights excluded.")
        if model["status"] == "uncurated":
            warnings.append("No catalog entry - generic estimate, accuracy unknown.")
        if meta["element"] == "voxel":
            side = round(n ** (1 / 3))
            warnings.append(f"num_elements counts voxels: {num_elements} ~ {side}^3 grid.")
        if meta["element"] == "query point" and "pde_operator" in meta["tasks"]:
            warnings.append("PINN residuals with second derivatives need roughly 3x the activation memory.")

        return {
            "model": model["name"],
            "element": meta["element"],
            "num_elements": num_elements,
            "batch_size": batch_size,
            "precision": precision,
            "complexity": meta["complexity"],
            "parameter_count": params,
            "memory_gb": {
                "weights": round(param_bytes / GB, 4),
                "activations_training": round(activation_bytes / GB, 3),
                "training_peak": round(train_gb, 3),
                "inference_peak": round(infer_gb, 3),
            },
            "hardware_tier": {"training": hardware_tier(train_gb), "inference": hardware_tier(infer_gb)},
            "scaling_if_elements_double": round(per_sample_floats(2 * n) / per_sample_floats(n), 2),
            "assumptions": (
                f"Default config; Adam ({TRAIN_BYTES_PER_PARAM} B/param); activations = "
                f"{mem['per_element']}*N + {mem.get('per_element_sq', 0)}*N^2 + {mem.get('per_sample', 0)} floats "
                f"per sample x {ACTIVATION_OVERHEAD} overhead; +{CUDA_CONTEXT_GB} GB CUDA context. "
                "Heuristic: validate with torch.cuda.max_memory_allocated()."
            ),
            "warnings": warnings,
        }

    def recommend_architecture(
        self,
        problem_description: str,
        input_format: str,
        max_elements_per_sample: int,
        dynamic_topology: bool = False,
        target_metric: str = "accuracy",
        top_k: int = 5,
        required_output: str | None = None,
    ) -> dict[str, Any]:
        input_rep = normalize_representation(input_format)
        if target_metric not in ("accuracy", "speed"):
            raise ValueError("target_metric must be 'accuracy' or 'speed'.")
        if required_output is not None and required_output not in OUTPUTS:
            raise ValueError(f"required_output must be one of {sorted(OUTPUTS)}.")
        text = problem_description.lower()
        inferred_tasks = [task for task, words in _TASK_KEYWORDS.items() if any(w in text for w in words)]

        ranked = []
        for name, model in self.state["models"].items():
            meta = model["metadata"]
            score, rationale, pitfalls = 0.0, [], []

            if input_rep in meta["representations"]:
                score += 3
                rationale.append(f"Native {input_rep} input.")
            elif input_rep in meta.get("also_accepts", {}):
                score += 1
                rationale.append(f"Accepts {input_rep} after conversion: {meta['also_accepts'][input_rep]}")
            else:
                continue

            matched_tasks = [task for task in inferred_tasks if task in meta["tasks"]]
            if matched_tasks:
                score += 2
                rationale.append(f"Suited to {', '.join(matched_tasks)}.")
            elif inferred_tasks:
                score -= 1
                pitfalls.append(f"Not designed for {', '.join(inferred_tasks)}.")

            outputs = meta.get("outputs")
            supports_output = required_output is None or outputs is None or required_output in outputs
            if not supports_output:
                score -= 4
                pitfalls.insert(0, f"Cannot predict {OUTPUTS[required_output]}: its heads give "
                                   f"{', '.join(OUTPUTS.get(o, o) for o in outputs)}.")

            hits = [kw for kw in meta.get("keywords", []) if kw in text]
            if hits:
                score += min(1.5, 0.5 * len(hits))
                rationale.append(f"Matches problem keywords: {', '.join(hits)}.")

            if dynamic_topology:
                support = meta["dynamic_topology"]
                if support is True:
                    score += 1.5
                    rationale.append("Copes with topology changing between samples/steps.")
                elif support == "partial":
                    pitfalls.append("Changing topology requires re-voxelization at a fixed resolution.")
                else:
                    score -= 2
                    pitfalls.append("Assumes a fixed geometry/topology; refit per geometry.")

            preference = meta["speed"] if target_metric == "speed" else meta["expressivity"]
            score += 0.5 * preference

            estimate = self.estimate_resource_footprint(name, max_elements_per_sample)
            train_gb = estimate["memory_gb"]["training_peak"]
            if train_gb > 24:
                score -= 3
                pitfalls.append(
                    f"VRAM saturation: ~{train_gb:.1f} GB to train one sample of {max_elements_per_sample} "
                    f"{meta['element']}s - needs multi-GPU, sub-sampling, patching or checkpointing."
                )
            elif train_gb > 8:
                score -= 1
                pitfalls.append(f"~{train_gb:.1f} GB training peak per sample - needs a 16-24GB GPU.")
            if meta["memory_model"].get("per_element_sq") and max_elements_per_sample > 10_000:
                pitfalls.append(f"Quadratic {meta['complexity']['time']} cost at {max_elements_per_sample} elements.")
            if model["status"] == "uncurated":
                pitfalls.append("Uncurated model: metadata is a generic default.")

            ranked.append({
                "model": name,
                "display_name": meta["display_name"],
                "score": round(score, 2),
                "matched_tasks": matched_tasks,
                "complexity": meta["complexity"]["time"],
                "estimated_training_peak_gb": train_gb,
                "hardware_tier": estimate["hardware_tier"]["training"],
                "rationale": rationale,
                "pitfalls": pitfalls,
                "examples": model["examples"][:3],
                "native_input": input_rep in meta["representations"],
                "outputs": outputs,
                "supports_output": supports_output,
                "evidence": _evidence_summary(meta.get("evidence")),
            })

        # Models that give the task and output come first; score orders each group.
        ranked.sort(key=lambda row: (row["supports_output"] and (not inferred_tasks or bool(row["matched_tasks"])),
                                     row["score"]), reverse=True)
        return {
            "input_format": input_rep,
            "inferred_tasks": inferred_tasks,
            "target_metric": target_metric,
            "required_output": required_output,
            **self._verdict(text, inferred_tasks, input_rep, ranked, required_output),
            "recommendations": ranked[:top_k],
            "considered": len(ranked),
            "note": "Heuristic ranking from catalog.yaml metadata; benchmark the top candidates on your data.",
        }

    def _verdict(
        self, text: str, inferred_tasks: list[str], input_rep: str, ranked: list[dict], required_output: str | None = None
    ) -> dict[str, Any]:
        """``fit``; ``weak_fit`` (fitting models need a representation conversion); ``no_fit`` (no registered
        model gives the task and output on any usable input), with the networks and datasets to add."""
        covered = {task for model in self.state["models"].values() for task in model["metadata"]["tasks"]}
        for capability, info in MISSING_CAPABILITIES.items():
            if capability not in covered and any(word in text for word in info["keywords"]):
                return {
                    "verdict": "no_fit",
                    "missing_capability": capability,
                    "reason": f"No registered model does {capability.replace('_', ' ')}. {info['needs']}",
                    "suggested_networks": info["suggested_networks"],
                    "suggested_datasets": info.get("suggested_datasets", []),
                }
        if not ranked:
            return {"verdict": "no_fit", "missing_capability": f"{input_rep} input",
                    "reason": f"No registered model accepts {input_rep} input, natively or after conversion.",
                    "suggested_networks": [], "suggested_datasets": []}
        usable = [r for r in ranked if r["supports_output"] and (not inferred_tasks or r["matched_tasks"])]
        if not usable:
            need = " / ".join(t.replace("_", " ") for t in inferred_tasks) or "this problem"
            if required_output:
                need += f" with {OUTPUTS[required_output]}"
            gap: dict[str, list] = {}
            for task in inferred_tasks:
                gap = (GAP_SUGGESTIONS.get((task, required_output, input_rep))
                       or GAP_SUGGESTIONS.get((task, required_output)) or gap)
            elsewhere = sorted({
                f"{rep} ({name})"
                for name, model in self.state["models"].items()
                for rep in model["metadata"]["representations"]
                if rep != input_rep
                and (not inferred_tasks or set(inferred_tasks) & set(model["metadata"]["tasks"]))
                and (required_output is None or required_output in (model["metadata"].get("outputs") or []))
            })
            reason = f"No registered model does {need} on {input_rep} input."
            if elsewhere:
                reason += " Converting the data would allow: " + ", ".join(elsewhere[:6]) + "."
            return {"verdict": "no_fit", "missing_capability": f"{need} on {input_rep}", "reason": reason,
                    "suggested_networks": gap.get("networks", []), "suggested_datasets": gap.get("datasets", []),
                    "convertible_alternatives": elsewhere}
        if not any(r["native_input"] for r in usable):
            return {"verdict": "weak_fit",
                    "reason": f"Models fit only after converting {input_rep} input; see each model's rationale."}
        return {"verdict": "fit"}


def _evidence_summary(evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    """Paper, top reported results and implementation notes for a ranked row (full record in the dossier)."""
    if not evidence:
        return None
    return {
        "paper": evidence.get("paper"),
        "reported": (evidence.get("reported") or [])[:3],
        "benchmark_datasets": evidence.get("benchmark_datasets", []),
        "implementation_notes": evidence.get("implementation_notes", ""),
        "checked": evidence.get("checked", ""),
    }


def hardware_tier(gigabytes: float) -> str:
    for limit, label in _TIERS:
        if gigabytes < limit:
            return label
    return "Multi-GPU / >24GB required"
