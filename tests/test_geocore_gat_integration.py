from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = WORKSPACE_ROOT / "neural3dbench" / "examples" / "geocore"
GEOCORE_ROOT = WORKSPACE_ROOT / "geometry-py" / "geocore"
for source_root in (EXAMPLES_ROOT, GEOCORE_ROOT):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import geocore as gc  # noqa: E402
from geocore_gat_integration import (  # noqa: E402
    adapt_graph,
    assert_bidirectional,
    run,
)


def test_saved_geocore_mesh_graph_runs_through_gat(tmp_path):
    run(tmp_path)
    raw = gc.load(str(tmp_path / "gat_mesh_graph.npz"))

    assert str(raw["representation"]) == "MESH_GRAPH"
    assert_bidirectional(raw["edge_index"])
    model_input = adapt_graph(raw)
    assert model_input.representation == "GRAPH"
    assert torch.equal(model_input.edge_index, torch.as_tensor(raw["edge_index"]))
    assert model_input.node_features.shape == (48, 3)
    assert model_input.edge_features.shape[0] == model_input.edge_index.shape[1]
    assert np.array_equal(model_input.coordinates.numpy(), raw["coordinates"])