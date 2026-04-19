"""STL / mesh I/O and sampling to point clouds."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import open3d as o3d


def load_triangle_mesh(path: Path | str) -> o3d.geometry.TriangleMesh:
    path = Path(path)
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise RuntimeError(f"Empty mesh: {path}")
    mesh.compute_vertex_normals()
    return mesh


def mesh_to_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    number_of_points: int = 80_000,
    seed: int = 0,
) -> o3d.geometry.PointCloud:
    _ = seed  # reserved for reproducible sampling if API allows
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    return pcd


def mesh_centered_copy(mesh: o3d.geometry.TriangleMesh) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Return mesh copy with vertices centered; centroid in original coordinates."""
    m = copy.deepcopy(mesh)
    verts = np.asarray(m.vertices)
    c = verts.mean(axis=0)
    m.translate(-c)
    return m, c
