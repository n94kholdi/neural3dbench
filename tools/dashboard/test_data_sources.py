"""Real-data layer offline: OFF parsing, uploads, the .npz store and both importers on tiny fake archives.

Run with ``pytest tools/dashboard/test_data_sources.py``. Archives are served from ``file://`` URLs,
so no network is used.
"""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dashboard import data_sources, tasks  # noqa: E402
from tools.dashboard.app import parse_labelled_upload  # noqa: E402

CUBE_OFF = """OFF
8 6 0
-1 -1 -1
1 -1 -1
1 1 -1
-1 1 -1
-1 -1 1
1 -1 1
1 1 1
-1 1 1
4 0 1 2 3
4 4 5 6 7
4 0 1 5 4
4 2 3 7 6
4 1 2 6 5
4 0 3 7 4
"""


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(tasks, "DATA_DIR", tmp_path / "cache")
    yield tmp_path / "cache"
    for name in [n for n in tasks.DATASETS if n not in tasks.BUILTIN_DATASETS]:
        del tasks.DATASETS[name]


def _archive(path: Path, files: dict[str, str]) -> data_sources.Archive:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    data = path.read_bytes()
    return data_sources.Archive(path.name, (path.as_uri(),), len(data), hashlib.sha256(data).hexdigest())


def test_parse_off_and_sampling() -> None:
    vertices, triangles = data_sources.parse_off(CUBE_OFF)
    assert vertices.shape == (8, 3) and triangles.shape == (12, 3)
    malformed = CUBE_OFF.replace("OFF\n8 6 0", "OFF8 6 0")  # ModelNet's header quirk
    assert data_sources.parse_off(malformed)[1].shape == (12, 3)
    points = data_sources.sample_surface(vertices, triangles, 500, np.random.default_rng(0))
    assert np.allclose(np.abs(points).max(axis=1), 1.0)  # every point lies on a cube face


def test_download_checks_hash_and_falls_back(tmp_path: Path) -> None:
    good = _archive(tmp_path / "a.zip", {"x.txt": "hello"})
    bad_url = (tmp_path / "missing.zip").as_uri()
    archive = data_sources.Archive("a.zip", (bad_url, *good.urls), good.size, good.sha256)
    assert data_sources.download(archive).read_bytes() == (tmp_path / "a.zip").read_bytes()
    corrupt = data_sources.Archive("b.zip", good.urls, good.size, "0" * 64)
    with pytest.raises(RuntimeError, match="sha256"):
        data_sources.download(corrupt)
    assert not list((tasks.DATA_DIR / "raw").glob("*.part"))


def test_import_modelnet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {f"ModelNet10/{c}/{split}/{c}_{i}.off": CUBE_OFF for c in ("bed", "chair") for split in ("train", "test")
             for i in range(2)}
    files["__MACOSX/ModelNet10/chair/train/._chair_0.off"] = "junk"
    monkeypatch.setitem(data_sources.MODELNET, 10, _archive(tmp_path / "ModelNet10.zip", files))
    dataset = data_sources.import_modelnet(10, num_points=64)
    assert dataset.name == "modelnet10" and dataset.task == tasks.SHAPE_CLASSIFICATION
    assert dataset.class_names == ("bed", "chair")
    coords, labels = tasks.make_dataset("modelnet10", 8, 32)
    assert coords.shape == (8, 32, 3) and sorted(labels.tolist()) == [0] * 4 + [1] * 4


def test_import_shapenet_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)

    def shape(low: int, high: int) -> str:
        rows = np.column_stack([rng.random((40, 6)), rng.integers(low, high, 40)])
        return "\n".join(" ".join(f"{v:g}" for v in row) for row in rows)

    root = "shapenetcore_partanno_segmentation_benchmark_v0_normal"
    files = {f"{root}/synsetoffset2category.txt": "Chair\t03001627\nTable\t04379243\n"}
    files.update({f"{root}/03001627/c{i}.txt": shape(12, 16) for i in range(3)})
    files.update({f"{root}/04379243/t{i}.txt": shape(47, 50) for i in range(3)})
    monkeypatch.setattr(data_sources, "SHAPENET_PART", _archive(tmp_path / "snp.zip", files))
    dataset = data_sources.import_shapenet_part("Chair", num_points=32)
    assert dataset.name == "shapenet_part_chair" and dataset.task == tasks.PART_SEGMENTATION
    assert dataset.num_classes == 4
    coords, labels = tasks.make_dataset("shapenet_part_chair", 3, 16)
    assert coords.shape == (3, 16, 3) and labels.min() >= 0 and labels.max() <= 3  # remapped from 12..15


def test_upload_save_and_rename_classes() -> None:
    coords = np.random.default_rng(0).random((4, 50, 3))
    labels = np.random.default_rng(1).integers(0, 3, (4, 50))
    buffer = io.BytesIO()
    np.savez(buffer, points=coords, labels=labels.astype(float), class_names=np.array(["a", "b", "c"]))
    parsed = parse_labelled_upload([("mine.npz", buffer.getvalue())], 2048)
    dataset = tasks.save_npz_dataset("mine", parsed["coords"], labels=parsed["labels"], class_names=parsed["class_names"])
    assert dataset.task == tasks.PART_SEGMENTATION and dataset.class_names == ("a", "b", "c")

    renamed = tasks.update_classes("mine", ["seat", "back", "legs"], ["#ff0000", "#00ff00", "#0000ff"])
    assert renamed.class_names == ("seat", "back", "legs") and renamed.class_colors[0] == "#ff0000"
    with pytest.raises(ValueError):
        tasks.update_classes("mine", ["seat", "back", "legs"], ["red", "green", "blue"])
    with pytest.raises(ValueError):
        tasks.save_npz_dataset("chair_parts", coords, labels=labels)  # built-in name
    with pytest.raises(ValueError):
        tasks.save_npz_dataset("Bad Name", coords, labels=labels)

    del tasks.DATASETS["mine"]
    assert "mine" in tasks.register_cached_datasets()  # survives a restart


def test_ply_upload_needs_labels() -> None:
    def ply(n: int, with_label: bool) -> bytes:
        header = "ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n" % n
        header += "property int label\n" if with_label else ""
        rows = "\n".join(f"{i} {i % 3} 0" + (f" {i % 2}" if with_label else "") for i in range(n))
        return (header + "end_header\n" + rows + "\n").encode()

    parsed = parse_labelled_upload([("a.ply", ply(10, True)), ("b.ply", ply(30, True))], 16)
    assert parsed["coords"].shape == (2, 16, 3) and parsed["labels"].shape == (2, 16)
    with pytest.raises(ValueError, match="label"):
        parse_labelled_upload([("a.ply", ply(10, False)), ("b.ply", ply(10, False))], 16)


def test_file_dataset_trains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mlflow")
    from tools.dashboard.trainer import Trainer

    coords, labels = tasks.make_dataset("chair_parts", 12, 128)
    tasks.save_npz_dataset("my_chairs", coords.numpy(), labels=labels.numpy(), class_names=list(tasks.CHAIR_PARTS))
    monkeypatch.chdir(tmp_path)
    trainer = Trainer()
    trainer.start_training({"model_name": "pointnet", "dataset": "my_chairs", "epochs": 1, "num_samples": 12,
                            "num_points": 64, "batch_size": 4})
    status = trainer.wait(timeout=120)
    assert status["status"] == "completed", status["error"]
    assert torch.load(status["checkpoint_path"], weights_only=True)["config"]["output_dim"] == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
