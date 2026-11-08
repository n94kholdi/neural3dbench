"""Real datasets for the dashboard: verified downloads, importers and labelled uploads.

Every source is converted once into the ``.npz`` format of ``tasks.save_npz_dataset``
(``coords [B, N, 3]`` plus labels or a target) under ``tasks.DATA_DIR``, then trains like the
synthetic datasets. Archives are read in place with ``zipfile`` and never extracted, so a
hostile archive cannot write outside the cache.

=============== ===================== ================================================ ========
source          task                  origin                                           download
=============== ===================== ================================================ ========
ShapeNetPart    part segmentation     Hugging Face mirror of the Stanford benchmark    707 MB
ModelNet10      shape classification  Princeton ModelNet (OFF meshes, points sampled)  473 MB
ModelNet40      shape classification  Princeton ModelNet (OFF meshes, points sampled)  2.0 GB
=============== ===================== ================================================ ========
"""

from __future__ import annotations

import hashlib
import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from tools.dashboard import tasks

Progress = Callable[[float, str], None]  # (fraction in [0, 1], message)


@dataclass(frozen=True)
class Archive:
    filename: str
    urls: tuple[str, ...]  # tried in order
    size: int
    sha256: str | None = None  # None: the publisher gives no checksum, only the size is checked


SHAPENET_PART = Archive(
    "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip",
    (
        "https://huggingface.co/datasets/cminst/ShapeNet/resolve/main/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip",
        "https://huggingface.co/datasets/wangps/shapenet_segmentation/resolve/main/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip",
    ),
    706_988_960,
    "0e26411700bae2da38ee8ecc719ba4db2e6e0133486e258665952ad5dfced0fe",
)
MODELNET = {
    10: Archive("ModelNet10.zip", ("http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",), 473_402_300),
    40: Archive("ModelNet40.zip", ("http://modelnet.cs.princeton.edu/ModelNet40.zip",), 2_039_180_837),
}

# Global part-id ranges per ShapeNetPart category (50 parts in total), as in the PointNet++ reference code.
SHAPENET_PART_RANGES = {
    "Airplane": (0, 4), "Bag": (4, 6), "Cap": (6, 8), "Car": (8, 12), "Chair": (12, 16), "Earphone": (16, 19),
    "Guitar": (19, 22), "Knife": (22, 24), "Lamp": (24, 28), "Laptop": (28, 30), "Motorbike": (30, 36),
    "Mug": (36, 38), "Pistol": (38, 41), "Rocket": (41, 44), "Skateboard": (44, 47), "Table": (47, 50),
}


def _report(progress: Progress | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(min(max(fraction, 0.0), 1.0), message)


# ---------------------------------------------------------------- download


def download(archive: Archive, progress: Progress | None = None) -> Path:
    """Fetch ``archive`` into ``DATA_DIR/raw`` (reused when already complete); checks size and sha256."""
    dest = tasks.DATA_DIR / "raw" / archive.filename
    if dest.is_file() and dest.stat().st_size == archive.size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    errors = []
    for url in archive.urls:
        try:
            _fetch(url, partial, archive, progress)
        except (OSError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
            partial.unlink(missing_ok=True)
            continue
        partial.replace(dest)
        return dest
    raise RuntimeError("download failed from every mirror:\n" + "\n".join(errors))


def _fetch(url: str, partial: Path, archive: Archive, progress: Progress | None) -> None:
    digest, done = hashlib.sha256(), 0
    request = urllib.request.Request(url, headers={"User-Agent": "neural3dbench-dashboard"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            _report(progress, done / archive.size, f"Downloading {archive.filename}: {done / 1e6:,.0f} / {archive.size / 1e6:,.0f} MB")
    if done != archive.size:
        raise ValueError(f"got {done} bytes, expected {archive.size}.")
    if archive.sha256 and digest.hexdigest() != archive.sha256:
        raise ValueError("sha256 mismatch; the mirror file differs from the pinned benchmark archive.")


# ---------------------------------------------------------------- shared geometry helpers


def resample(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of ``count`` points: without replacement when possible."""
    return rng.choice(len(points), size=count, replace=len(points) < count)


def normalize(points: np.ndarray) -> np.ndarray:
    """Center on the centroid and scale into the unit sphere."""
    points = points - points.mean(0)
    return points / max(float(np.linalg.norm(points, axis=1).max()), 1e-12)


def parse_off(text: str) -> tuple[np.ndarray, np.ndarray]:
    """OFF mesh -> ``(vertices [V, 3], triangles [F, 3])``; polygons are fan-triangulated.

    Accepts ModelNet's malformed header where the counts follow ``OFF`` on the same line.
    """
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines or not lines[0].strip().startswith("OFF"):
        raise ValueError("not an OFF file.")
    head = lines[0].strip()[3:].split()
    counts, body = (head, lines[1:]) if head else (lines[1].split(), lines[2:])
    n_vertices, n_faces = int(counts[0]), int(counts[1])
    vertices = np.array(" ".join(body[:n_vertices]).split(), dtype=np.float64).reshape(n_vertices, 3)
    triangles = []
    for line in body[n_vertices : n_vertices + n_faces]:
        index = [int(v) for v in line.split()]
        polygon = index[1 : 1 + index[0]]
        triangles += [[polygon[0], polygon[i], polygon[i + 1]] for i in range(1, len(polygon) - 1)]
    return vertices, np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def sample_surface(vertices: np.ndarray, triangles: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Area-weighted uniform points on a triangle mesh."""
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if not area.sum() > 0:
        raise ValueError("mesh has no surface area.")
    face = rng.choice(len(triangles), size=count, p=area / area.sum())
    u, v = rng.random((2, count, 1))
    flip = (u + v) > 1
    u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
    return a[face] + u * (b[face] - a[face]) + v * (c[face] - a[face])


# ---------------------------------------------------------------- importers


def import_shapenet_part(
    category: str, num_points: int = 2048, max_shapes: int | None = 500, seed: int = 0, progress: Progress | None = None
) -> tasks.Dataset:
    """ShapeNetPart shapes of one ``category`` (or ``"All"``) as ``shapenet_part_<category>``.

    Labels are remapped to local part ids; the archive has no part names, so parts are named
    ``<Category> part k``; rename them in the dashboard.
    """
    if category != "All" and category not in SHAPENET_PART_RANGES:
        raise ValueError(f"category must be 'All' or one of {sorted(SHAPENET_PART_RANGES)}.")
    path = download(SHAPENET_PART, progress)
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        mapping_file = next(n for n in names if n.endswith("synsetoffset2category.txt"))
        synsets = dict(line.split() for line in archive.read(mapping_file).decode().splitlines() if line.strip())
        wanted = synsets if category == "All" else {category: synsets[category]}
        members = [n for n in names if n.endswith(".txt") and n.split("/")[-2] in wanted.values()]
        members = [members[i] for i in sorted(rng.permutation(len(members))[: max_shapes or len(members)])]
        coords, labels = [], []
        for i, member in enumerate(members):
            rows = np.loadtxt(io.BytesIO(archive.read(member)), dtype=np.float64)  # x y z nx ny nz part
            pick = resample(rows, num_points, rng)
            coords.append(normalize(rows[pick, :3]))
            labels.append(rows[pick, 6].astype(np.int64))
            _report(progress, (i + 1) / len(members), f"Converting ShapeNetPart: {i + 1} / {len(members)} shapes")
    labels = np.stack(labels)
    if category == "All":
        class_names = [f"{cat} part {k}" for cat, (lo, hi) in SHAPENET_PART_RANGES.items() for k in range(hi - lo)]
    else:
        low, high = SHAPENET_PART_RANGES[category]
        labels -= low
        class_names = [f"{category} part {k}" for k in range(high - low)]
    return tasks.save_npz_dataset(
        f"shapenet_part_{category.lower()}", np.stack(coords), labels=labels, class_names=class_names,
        description=f"ShapeNetPart {category}: {len(members)} shapes x {num_points} points, {len(class_names)} parts.",
    )


def import_modelnet(
    variant: int = 10, num_points: int = 1024, max_per_class: int | None = 100, seed: int = 0, progress: Progress | None = None
) -> tasks.Dataset:
    """ModelNet10/40 meshes (train and test splits) as point clouds sampled on the surface, ``modelnet<variant>``."""
    if variant not in MODELNET:
        raise ValueError("variant must be 10 or 40.")
    path = download(MODELNET[variant], progress)
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(path) as archive:
        members = sorted(n for n in archive.namelist() if n.endswith(".off") and "__MACOSX" not in n)
        class_names = sorted({n.split("/")[-3] for n in members})
        by_class: dict[str, list[str]] = {c: [] for c in class_names}
        for member in members:
            by_class[member.split("/")[-3]].append(member)
        chosen = [m for c in class_names for m in sorted(rng.permutation(by_class[c])[: max_per_class or None])]
        coords, labels, skipped = [], [], 0
        for i, member in enumerate(chosen):
            try:
                vertices, triangles = parse_off(archive.read(member).decode("ascii", errors="replace"))
                coords.append(normalize(sample_surface(vertices, triangles, num_points, rng)))
                labels.append(class_names.index(member.split("/")[-3]))
            except ValueError:
                skipped += 1  # a few ModelNet meshes are degenerate
            _report(progress, (i + 1) / len(chosen), f"Sampling ModelNet{variant}: {i + 1} / {len(chosen)} meshes")
    return tasks.save_npz_dataset(
        f"modelnet{variant}", np.stack(coords), labels=np.asarray(labels), class_names=class_names,
        description=f"ModelNet{variant}: {len(labels)} meshes x {num_points} surface points"
        + (f" ({skipped} degenerate skipped)" if skipped else "") + ".",
    )


def clean_mesh(vertices: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop degenerate and duplicate triangles and unreferenced vertices."""
    triangles = triangles[(triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
                          & (triangles[:, 0] != triangles[:, 2])]
    _, first = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
    triangles = triangles[np.sort(first)]
    used, triangles = np.unique(triangles, return_inverse=True)
    return vertices[used], triangles.reshape(-1, 3)


def simplify_mesh(vertices: np.ndarray, triangles: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Clean the mesh and, above ``max_faces``, decimate it by vertex clustering on the finest grid that fits.

    Vertex clustering merges all vertices in a grid cell into their mean. It is fast and needs no
    extra dependency but does not preserve manifoldness (fine for MeshNet, DiffusionNet, MeshWalker).
    """
    vertices, triangles = clean_mesh(vertices, triangles)
    resolution = 256.0
    while len(triangles) > max_faces:
        cell = np.floor((vertices - vertices.min(0)) / (np.ptp(vertices, axis=0).max() / resolution + 1e-12))
        _, cluster, size = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
        cluster = cluster.reshape(-1)
        merged = np.zeros((len(size), 3))
        np.add.at(merged, cluster, vertices)
        candidate = clean_mesh(merged / size[:, None], cluster[triangles])
        if len(candidate[1]) <= max_faces:
            vertices, triangles = candidate
            break
        resolution *= 0.8
    if len(triangles) == 0:
        raise ValueError("mesh has no valid triangles.")
    return vertices, triangles


def load_modelnet_meshes(
    variant: int = 10, split: str = "train", max_faces: int = 2048, max_per_class: int | None = None,
    seed: int = 0, progress: Progress | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, list[str]]:
    """ModelNet meshes of one official split as ``([(vertices [V, 3], faces [F, 3]), ...], labels [B], class_names)``.

    Meshes are centered on their bounding box, scaled into the unit sphere and decimated to at most
    ``max_faces`` triangles (``simplify_mesh``). The result is cached under ``DATA_DIR/meshes``.
    """
    if variant not in MODELNET or split not in ("train", "test"):
        raise ValueError("variant must be 10 or 40 and split 'train' or 'test'.")
    cache = tasks.DATA_DIR / "meshes" / f"modelnet{variant}_{split}_f{max_faces}_c{max_per_class or 'all'}_s{seed}.npz"
    if not cache.is_file():
        path = download(MODELNET[variant], progress)
        rng = np.random.default_rng(seed)
        with zipfile.ZipFile(path) as archive:
            members = sorted(n for n in archive.namelist()
                             if n.endswith(".off") and "__MACOSX" not in n and n.split("/")[-2] == split)
            class_names = sorted({n.split("/")[-3] for n in members})
            chosen = [m for c in class_names
                      for m in sorted(rng.permutation([n for n in members if n.split("/")[-3] == c])[: max_per_class or None])]
            vertices, faces, labels = [], [], []
            for i, member in enumerate(chosen):
                try:
                    verts, tris = parse_off(archive.read(member).decode("ascii", errors="replace"))
                    verts = verts - (verts.min(0) + verts.max(0)) / 2
                    verts = verts / max(float(np.linalg.norm(verts, axis=1).max()), 1e-12)
                    verts, tris = simplify_mesh(verts, tris, max_faces)
                except ValueError:
                    continue  # a few ModelNet meshes are degenerate
                vertices.append(verts.astype(np.float32))
                faces.append(tris.astype(np.int64))
                labels.append(class_names.index(member.split("/")[-3]))
                _report(progress, (i + 1) / len(chosen), f"Reading ModelNet{variant} {split}: {i + 1} / {len(chosen)} meshes")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, vertices=np.concatenate(vertices), faces=np.concatenate(faces),
            vertex_counts=np.array([len(v) for v in vertices]), face_counts=np.array([len(f) for f in faces]),
            labels=np.array(labels), class_names=np.array(class_names, dtype=str),
        )
    with np.load(cache, allow_pickle=False) as data:
        vertices = np.split(data["vertices"], np.cumsum(data["vertex_counts"])[:-1])
        faces = np.split(data["faces"], np.cumsum(data["face_counts"])[:-1])
        return list(zip(vertices, faces)), data["labels"], [str(n) for n in data["class_names"]]


# ---------------------------------------------------------------- uploads

_COORD_KEYS = ("coords", "coordinates", "points", "pos", "xyz", "vertices")
_LABEL_KEYS = ("labels", "label", "seg", "part_labels", "class", "y")
_TARGET_KEYS = ("target", "field", "values")


def read_npz_upload(data: bytes) -> dict[str, np.ndarray | list[str] | None]:
    """Parse an uploaded ``.npz``: coords ``[B, N, 3]`` plus labels ``[B, N]`` / ``[B]`` or a float target.

    Returns ``{"coords", "labels", "target", "class_names"}`` (missing entries are None / []).
    """
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        keys = set(npz.files)
        coord_key = next((k for k in _COORD_KEYS if k in keys), None)
        if coord_key is None:
            raise ValueError(f"need one of the arrays {_COORD_KEYS}; found {sorted(keys)}.")
        coords = npz[coord_key]
        label_key = next((k for k in _LABEL_KEYS if k in keys), None)
        target_key = next((k for k in _TARGET_KEYS if k in keys), None)
        labels = npz[label_key] if label_key else None
        target = npz[target_key] if target_key and labels is None else None
        class_names = [str(n) for n in npz["class_names"]] if "class_names" in keys else []
    if labels is None and target is None:
        raise ValueError(f"need labels ({_LABEL_KEYS}) or a target ({_TARGET_KEYS}).")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim == 3 and labels.shape[-1] == 1:
            labels = labels[..., 0]
        if not np.issubdtype(labels.dtype, np.integer):
            if not np.allclose(labels, np.round(labels)):
                raise ValueError("labels must be integer class ids.")
            labels = labels.astype(np.int64)
    return {"coords": coords, "labels": labels, "target": target, "class_names": class_names}


def stack_samples(samples: list[tuple[np.ndarray, np.ndarray]], num_points: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Resample per-file ``(coords [N_i, 3], labels [N_i])`` to a common ``num_points`` and stack."""
    rng = np.random.default_rng(seed)
    coords, labels = [], []
    for points, point_labels in samples:
        if len(points) != len(point_labels):
            raise ValueError("each sample needs one label per point.")
        pick = resample(points, num_points, rng)
        coords.append(normalize(np.asarray(points, dtype=np.float64)[pick]))
        labels.append(np.asarray(point_labels, dtype=np.int64)[pick])
    return np.stack(coords), np.stack(labels)


def suggest_name(filename: str) -> str:
    """A valid dataset name from an upload filename."""
    name = re.sub(r"[^a-z0-9_]+", "_", Path(filename).stem.lower()).strip("_") or "uploaded"
    return name if name[0].isalnum() else f"d_{name}"
