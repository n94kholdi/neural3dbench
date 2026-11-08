"""In-process bridge from the dashboard to the neural3d-advisor.

Uses ``tools.mcp_advisor.Advisor`` directly (no MCP transport). If the advisor
cannot load (for example ``pyyaml`` is missing), it falls back to inspecting
``models.ModelRegistry`` in-process, with reduced metadata and ``source="registry"``.
Results are plain dicts shaped like the matching MCP tool output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Fallback only; mirrors tools.mcp_advisor.advisor._REPO_REPRESENTATIONS.
_REPO_REPRESENTATIONS = {
    "POINT_CLOUD": "point_cloud",
    "GRAPH": "mesh_graph",
    "MESH_GRAPH": "mesh_graph",
    "GRID": "voxel_grid",
    "VOXEL_GRID": "voxel_grid",
    "PARAMETRIC_SURFACE": "continuous_coords",
}
_TRAIN_BYTES_PER_PARAM = 16  # Adam: weights, grads, two moments (fp32)
_CUDA_CONTEXT_GB = 0.5
_GB = 1024**3


class AdvisorBridge:
    """Query the advisor from Python: recommendations, dossiers and memory estimates."""

    def __init__(self, advisor: Any | None = None) -> None:
        self.advisor = advisor
        self.error: str | None = None
        if advisor is None:
            try:
                from tools.mcp_advisor import Advisor

                self.advisor = Advisor()
                self.advisor.state  # force registry introspection now so failures surface here
            except Exception as exc:  # noqa: BLE001 - any advisor failure switches to the fallback
                self.advisor = None
                self.error = f"{type(exc).__name__}: {exc}"

    @property
    def source(self) -> str:
        return "advisor" if self.advisor is not None else "registry"

    # ------------------------------------------------------------------ API

    def get_recommendations(
        self,
        problem_type: str,
        representation: str,
        num_nodes: int,
        vram_gb: float | None = None,
        target_metric: str = "accuracy",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Ranked models for a problem; each row gains ``fits_vram`` against ``vram_gb``.

        problem_type: free text or a task name, e.g. "surrogate_simulation" or "CFD pressure on car surfaces".
        representation: point_cloud | mesh_graph | voxel_grid | continuous_coords (aliases accepted).
        """
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        if self.advisor is not None:
            # Task names are matched by keyword, so turn "surrogate_simulation" into "surrogate simulation".
            result = self.advisor.recommend_architecture(
                problem_type.replace("_", " "), representation, num_nodes, target_metric=target_metric, top_k=top_k
            )
        else:
            result = self._fallback_recommendations(representation, num_nodes, top_k)

        for row in result["recommendations"]:
            peak = row.get("estimated_training_peak_gb")
            row["fits_vram"] = None if vram_gb is None or peak is None else peak <= vram_gb
        if vram_gb is not None:
            # Stable sort: models that fit come first, score order kept inside each group.
            result["recommendations"].sort(key=lambda row: row["fits_vram"] is False)
        result.update(vram_gb=vram_gb, source=self.source)
        return result

    def get_dossier(self, model_name: str) -> dict[str, Any]:
        """Biases, merits, demerits, tensor contract and implementation details for one model."""
        if self.advisor is not None:
            return {**self.advisor.get_model_dossier(model_name), "source": self.source}
        return self._fallback_dossier(model_name)

    def estimate_memory(self, model_name: str, num_nodes: int, batch_size: int = 1, precision: str = "fp32") -> dict[str, Any]:
        """Heuristic GPU memory (GB) for training and inference."""
        if num_nodes <= 0 or batch_size <= 0:
            raise ValueError("num_nodes and batch_size must be positive.")
        if self.advisor is not None:
            return {**self.advisor.estimate_resource_footprint(model_name, num_nodes, batch_size, precision), "source": self.source}
        return self._fallback_memory(model_name, num_nodes, batch_size, precision)

    def preflight(
        self,
        model_name: str,
        problem_description: str,
        num_nodes: int,
        assumptions: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Guardrail report for one model: assumption audit, symmetry verdicts and training time.

        assumptions keys (all optional): boundary_conditions, topology_changes, geometry_varies,
        evaluate_at_unseen_resolution, required_symmetries, target_kind, target_gpu, dataset_size, epochs.
        Returns ``None`` on the registry fallback, which has no catalog metadata to check against.
        """
        if self.advisor is None:
            return None
        from tools.mcp_advisor import guardrails

        a = assumptions or {}
        audit = guardrails.audit_problem_assumptions(
            self.advisor, problem_description, model_name, a.get("boundary_conditions", "none"),
            a.get("topology_changes", False), a.get("geometry_varies", False),
            a.get("evaluate_at_unseen_resolution", False),
        )
        findings = next(iter(audit["audits"].values()), [])
        symmetries = a.get("required_symmetries") or []
        symmetry = guardrails.check_symmetry_and_invariance(
            self.advisor, model_name, symmetries, a.get("target_kind", "scalar")
        ) if symmetries else None
        budget = guardrails.estimate_training_budget(
            self.advisor, model_name, num_nodes, int(a.get("dataset_size", 1000)), int(a.get("epochs", 100)),
            a.get("target_gpu", "RTX4090"),
        )
        broken = [] if symmetry is None else [v for v in symmetry["verdicts"] if not v["guaranteed_by_architecture"]]
        return {
            "findings": findings,
            "errors": sum(f["severity"] == "error" for f in findings),
            "warnings": sum(f["severity"] == "warning" for f in findings) + len(broken),
            "symmetry": symmetry,
            "budget": budget,
        }

    def benchmark_results(self, tracking_uri: str = "./mlruns", **filters: Any) -> dict[str, Any]:
        """Measured MLflow results, same as the ``get_benchmark_results`` MCP tool."""
        try:
            from tools.mcp_advisor.results import benchmark_results
        except ImportError as exc:  # tools.mcp_advisor needs pyyaml
            return {"runs": [], "best_per_dataset": [], "note": f"Results unavailable: {exc}"}
        return benchmark_results(tracking_uri, **filters)

    # ------------------------------------------------------------- fallback

    @staticmethod
    def _registry() -> Any:
        from models import ModelRegistry

        return ModelRegistry

    def _representation(self, name: str) -> str | None:
        code = self._registry().get_config_class(name)().representation
        return _REPO_REPRESENTATIONS.get(str(code).upper()) if code else None

    def _parameter_count(self, name: str) -> int | None:
        try:
            return sum(p.numel() for p in self._registry().create(name).parameters())
        except Exception:  # noqa: BLE001 - some models need required config fields
            return None

    def _fallback_recommendations(self, representation: str, num_nodes: int, top_k: int) -> dict[str, Any]:
        wanted = _REPO_REPRESENTATIONS.get(representation.upper(), representation)
        rows = []
        for name in sorted(self._registry()._registry):
            if self._representation(name) != wanted:
                continue
            memory = self._fallback_memory(name, num_nodes, 1, "fp32")
            rows.append({
                "model": name,
                "display_name": name,
                "score": 0.0,
                "estimated_training_peak_gb": memory["memory_gb"]["training_peak"],
                "rationale": [f"Registered with native {wanted} input."],
                "pitfalls": ["Advisor unavailable: unranked, activations not estimated."],
            })
        return {
            "input_format": wanted,
            "inferred_tasks": [],
            "recommendations": rows[:top_k],
            "considered": len(rows),
            "note": f"Registry fallback ({self.error}).",
        }

    def _fallback_dossier(self, model_name: str) -> dict[str, Any]:
        registry = self._registry()
        model_class = registry.get(model_name)
        config_class = registry.get_config_class(model_name)
        return {
            "name": model_name,
            "status": "uncurated",
            "representations": [rep for rep in [self._representation(model_name)] if rep],
            "implementation": {
                "class": model_class.__name__,
                "module": model_class.__module__,
                "config_class": config_class.__name__,
                "parameter_count_default": self._parameter_count(model_name),
                "docstring": (model_class.__doc__ or "").strip(),
            },
            "source": self.source,
            "note": f"Registry fallback ({self.error}).",
        }

    def _fallback_memory(self, model_name: str, num_nodes: int, batch_size: int, precision: str) -> dict[str, Any]:
        self._registry().get(model_name)  # raises KeyError for unknown models
        params = self._parameter_count(model_name)
        train_gb = (params or 0) * _TRAIN_BYTES_PER_PARAM / _GB + _CUDA_CONTEXT_GB
        return {
            "model": model_name,
            "num_elements": num_nodes,
            "batch_size": batch_size,
            "precision": precision,
            "parameter_count": params,
            "memory_gb": {"training_peak": round(train_gb, 3), "inference_peak": None},
            "warnings": ["Advisor unavailable: weights and optimizer state only, activations excluded."],
            "source": self.source,
        }
