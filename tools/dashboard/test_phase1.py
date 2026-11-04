"""Phase 1 smoke test: Plotly figures and AdvisorBridge on a dummy sphere.

Run with ``python tools/dashboard/test_phase1.py`` or ``pytest tools/dashboard/test_phase1.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dashboard.advisor_bridge import AdvisorBridge  # noqa: E402
from tools.dashboard.visualizer import (  # noqa: E402
    MAX_RENDER_ELEMENTS,
    plot_field_comparison,
    plot_mesh,
    plot_point_cloud,
)


def make_sphere(n_lat: int = 20, n_lon: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """UV sphere: vertices [N, 3] and triangle faces [F, 3] (poles duplicated, which is fine for plotting)."""
    theta, phi = np.meshgrid(np.linspace(0, np.pi, n_lat), np.linspace(0, 2 * np.pi, n_lon), indexing="ij")
    vertices = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n_lat - 1), np.arange(n_lon - 1), indexing="ij")
    a, b = (i * n_lon + j).ravel(), (i * n_lon + j + 1).ravel()
    c, d = a + n_lon, b + n_lon
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    return vertices, faces


def test_visualizer() -> None:
    vertices, faces = make_sphere()
    gt = vertices[:, 2]
    pred = gt + 0.05 * np.random.default_rng(0).standard_normal(len(gt))

    assert isinstance(plot_point_cloud(vertices), go.Figure)
    assert isinstance(plot_point_cloud(vertices, features=np.c_[gt, pred]), go.Figure)
    assert isinstance(plot_mesh(vertices, faces, field=gt), go.Figure)
    assert isinstance(plot_mesh(vertices), go.Figure)  # convex hull when faces are missing

    figure = plot_field_comparison(vertices, faces, gt, pred)
    assert len(figure.data) == 3
    assert np.allclose(figure.data[2].intensity, np.abs(gt - pred))
    assert figure.data[0].cmin == figure.data[1].cmin


def test_downsampling() -> None:
    vertices, faces = make_sphere(250, 250)  # 62,500 vertices
    assert len(vertices) > MAX_RENDER_ELEMENTS
    assert len(plot_point_cloud(vertices).data[0].x) == MAX_RENDER_ELEMENTS

    field = vertices[:, 0]
    figure = plot_field_comparison(vertices, faces, field, field * 0.9)
    mesh = figure.data[0]
    assert len(mesh.x) <= MAX_RENDER_ELEMENTS
    assert max(max(mesh.i), max(mesh.j), max(mesh.k)) < len(mesh.x)


def _check_bridge(bridge: AdvisorBridge) -> None:
    result = bridge.get_recommendations("surrogate_simulation", "point_cloud", num_nodes=5_000, vram_gb=8)
    assert result["recommendations"], result
    assert all("fits_vram" in row for row in result["recommendations"])

    name = result["recommendations"][0]["model"]
    assert bridge.get_dossier(name)["name"] == name
    memory = bridge.estimate_memory(name, num_nodes=5_000, batch_size=2)
    assert memory["memory_gb"]["training_peak"] > 0


def test_advisor_bridge() -> None:
    bridge = AdvisorBridge()
    assert bridge.source == "advisor", bridge.error
    _check_bridge(bridge)


def test_registry_fallback() -> None:
    bridge = AdvisorBridge()
    bridge.advisor, bridge.error = None, "forced for test"
    assert bridge.source == "registry"
    _check_bridge(bridge)


if __name__ == "__main__":
    for test in (test_visualizer, test_downsampling, test_advisor_bridge, test_registry_fallback):
        test()
        print(f"ok  {test.__name__}")
