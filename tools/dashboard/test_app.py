"""Phase 3 test: geometry parsers and a headless Streamlit run of tools/dashboard/app.py.

Run with ``pytest tools/dashboard/test_app.py``. The app test chdirs into a temp dir so
``./mlruns`` and ``./checkpoints`` do not land in the repository.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
pytest.importorskip("streamlit")
pytest.importorskip("mlflow")

from streamlit.testing.v1 import AppTest  # noqa: E402

from tools.dashboard.app import load_geometry, load_sample  # noqa: E402

APP = str(Path(__file__).with_name("app.py"))


@pytest.fixture(autouse=True)
def _fresh_streamlit_caches() -> None:
    """The cached Trainer outlives an AppTest; clear it so a run from another tmp dir does not leak in."""
    import streamlit as st

    st.cache_resource.clear()
    st.cache_data.clear()


def test_parse_obj_quad_and_slashes() -> None:
    text = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3 4/4/4\nf -4 -3 -2\n"
    geometry = load_geometry("quad.obj", text.encode())
    assert geometry["vertices"].shape == (4, 3)
    assert geometry["faces"].tolist() == [[0, 1, 2], [0, 2, 3], [0, 1, 2]]


def test_parse_ascii_ply() -> None:
    ply = (
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n"
        "0 0 0 255\n1 0 0 255\n0 1 0 255\n3 0 1 2\n"
    )
    geometry = load_geometry("tri.ply", ply.encode())
    assert geometry["vertices"][2].tolist() == [0, 1, 0]
    assert geometry["faces"].tolist() == [[0, 1, 2]]
    with pytest.raises(ValueError, match="ASCII"):
        load_geometry("bin.ply", b"ply\nformat binary_little_endian 1.0\nend_header\n")


def test_parse_pt_tensor_and_dict() -> None:
    def saved(obj: object) -> bytes:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        return buffer.getvalue()

    assert load_geometry("a.pt", saved(torch.rand(1, 10, 3)))["vertices"].shape == (10, 3)
    geometry = load_geometry("b.pt", saved({"coords": torch.rand(10, 3), "target": torch.rand(1, 10, 1)}))
    assert geometry["field"].shape == (10, 1) and geometry["faces"] is None
    with pytest.raises(ValueError):
        load_geometry("c.pt", saved({"nothing": 1}))
    with pytest.raises(ValueError):
        load_geometry("d.stl", b"")


def test_samples() -> None:
    assert load_sample("shape:cube", 128)["vertices"].shape == (128, 3)
    assert load_sample("dataset:poisson", 128)["field"].shape == (128, 1)
    assert load_sample("dataset:chair_parts", 128)["field"].shape == (128, 1)


def test_advisor_task_reaches_training_tab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.selectbox[0].set_value("segmentation_classification")
    app.text_input[0].set_value("semantic part segmentation of chairs")
    app.button(key="FormSubmitter:problem-Get recommendations").click().run()
    top = app.session_state["recommendations"]["recommendations"][0]["model"]
    app.button(key=f"use_{top}").click().run()
    assert app.session_state["train_model"] == top
    assert app.session_state["train_dataset"] == "chair_parts"
    assert not [w.value for w in app.warning if "advisor problem" in w.value]

    app.selectbox(key="train_dataset").set_value("poisson").run()
    assert any("part segmentation" in w.value for w in app.warning)


def test_app_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception, app.exception

    # Tab 1: recommendations render as cards with a "Use for training" button.
    app.button(key="FormSubmitter:problem-Get recommendations").click().run()
    assert app.session_state["recommendations"]["recommendations"]
    assert not app.exception, app.exception

    # Tab 3: a short run through the UI.
    app.number_input(key="train_epochs").set_value(2)
    app.number_input(key="train_samples").set_value(8)
    app.number_input(key="train_points").set_value(32)
    app.text_input(key="train_experiment").set_value("dashboard-test")
    next(b for b in app.button if b.label == "Start Training").click().run()
    assert not app.exception and not app.error, (app.exception, [e.value for e in app.error])

    deadline = time.time() + 120
    while any(i.value.startswith("Finish a training run") for i in app.info) and time.time() < deadline:
        time.sleep(0.5)
        app.run()
    assert not app.exception, app.exception
    assert any(m.label == "Final val_loss" for m in app.metric), ([e.value for e in app.error], [c.value for c in app.code])
    assert not app.error, [e.value for e in app.error]  # field comparison rendered from the checkpoint

    # Tab 4: relaunch copies the finished run's settings back into the form.
    app.number_input(key="train_epochs").set_value(7)
    next(b for b in app.button if b.label.startswith("Launch New Run")).click().run()
    assert app.session_state["train_epochs"] == 2



def test_uncovered_task_shows_not_supported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.selectbox[0].set_value("object_detection")
    app.button(key="FormSubmitter:problem-Get recommendations").click().run()
    assert not app.exception, app.exception
    assert any("Not supported yet" in e.value for e in app.error)
    assert app.session_state["problem_dataset"] is None
    assert all(b.disabled for b in app.button if b.label == "Use for training")



def test_data_tab_modes_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEURAL3DBENCH_DATA", str(tmp_path / "cache"))
    app = AppTest.from_file(APP, default_timeout=120).run()
    for mode, element in (("Download a dataset", "Download and import"), ("Upload a labelled dataset", None)):
        next(r for r in app.radio if r.label == "Data").set_value(mode).run()
        assert not app.exception, app.exception
        assert (element in [b.label for b in app.button]) == (element is not None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
