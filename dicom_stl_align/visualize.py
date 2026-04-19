"""Open3D viewers for overlays and before/after comparison."""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d


def _clone_pcd_colored(pcd: o3d.geometry.PointCloud, rgb) -> o3d.geometry.PointCloud:
    c = copy.deepcopy(pcd)
    c.paint_uniform_color(rgb)
    return c


def _clone_mesh_colored(mesh: o3d.geometry.TriangleMesh, T: np.ndarray, rgb) -> o3d.geometry.TriangleMesh:
    m = copy.deepcopy(mesh)
    m.transform(T)
    m.paint_uniform_color(rgb)
    m.compute_vertex_normals()
    return m


def show_overlay(
    target_pcd: o3d.geometry.PointCloud,
    source_mesh: o3d.geometry.TriangleMesh,
    T_source: np.ndarray,
    *,
    window_name: str = "Overlay",
) -> None:
    """Target as gray points; transformed STL mesh in orange."""
    geoms = [
        _clone_pcd_colored(target_pcd, [0.65, 0.65, 0.72]),
        _clone_mesh_colored(source_mesh, T_source, [0.95, 0.45, 0.08]),
    ]
    o3d.visualization.draw_geometries(geoms, window_name=window_name, width=1280, height=900)


def show_before_after(
    target_pcd: o3d.geometry.PointCloud,
    source_mesh: o3d.geometry.TriangleMesh,
    T_before: np.ndarray,
    T_after: np.ndarray,
) -> None:
    """Two modal windows: manual pose vs refined ICP pose."""
    show_overlay(target_pcd, source_mesh, T_before, window_name="Before ICP (manual initial)")
    show_overlay(target_pcd, source_mesh, T_after, window_name="After ICP (refined)")
