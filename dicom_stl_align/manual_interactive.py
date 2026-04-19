"""Keyboard-driven rigid pose for the STL mesh (world-frame increments)."""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import open3d as o3d

from dicom_stl_align.transforms import delta_rotation, delta_translation

_ACTION_DOWN = 1


def run_manual_pose_editor(
    target_pcd: o3d.geometry.PointCloud,
    source_mesh: o3d.geometry.TriangleMesh,
    T_init: np.ndarray,
    *,
    step_translate: float = 3.0,
    step_rotate_deg: float = 2.0,
    on_change: Callable[[np.ndarray], None] | None = None,
) -> np.ndarray:
    """
    Open a viewer; adjust pose with keys; close window to accept.

    Keys (keydown): WASD QE translate; IJK UO rotate (deg).
    P : print 4x4 matrix.
    """
    T = np.array(T_init, dtype=np.float64)
    base_mesh = copy.deepcopy(source_mesh)
    vis_mesh = copy.deepcopy(source_mesh)

    tgt_vis = copy.deepcopy(target_pcd)
    tgt_vis.paint_uniform_color([0.62, 0.62, 0.68])
    vis_mesh.paint_uniform_color([0.92, 0.42, 0.08])

    def apply_T_to_vis_mesh():
        verts = np.asarray(base_mesh.vertices)
        hom = np.hstack([verts, np.ones((verts.shape[0], 1))])
        vis_mesh.vertices = o3d.utility.Vector3dVector((T @ hom.T).T[:, :3])
        vis_mesh.triangles = base_mesh.triangles
        vis_mesh.compute_vertex_normals()
        vis_mesh.paint_uniform_color([0.92, 0.42, 0.08])

    apply_T_to_vis_mesh()

    vis = o3d.visualization.VisualizerWithKeyCallback()

    def refresh():
        apply_T_to_vis_mesh()
        vis.update_geometry(vis_mesh)
        vis.poll_events()
        vis.update_renderer()
        if on_change is not None:
            on_change(T.copy())

    def nudge_translate(dx: float, dy: float, dz: float):
        nonlocal T
        T = delta_translation(dx, dy, dz) @ T

    def nudge_rotate(rx: float, ry: float, rz: float):
        nonlocal T
        T = delta_rotation(rx, ry, rz) @ T

    def make_key_cb(dx=0.0, dy=0.0, dz=0.0, rx=0.0, ry=0.0, rz=0.0):
        def cb(vis_, action, mods):
            if action != _ACTION_DOWN:
                return False
            if dx or dy or dz:
                nudge_translate(dx * step_translate, dy * step_translate, dz * step_translate)
            if rx or ry or rz:
                nudge_rotate(rx * step_rotate_deg, ry * step_rotate_deg, rz * step_rotate_deg)
            refresh()
            return False

        return cb

    def print_pose(vis_, action, mods):
        if action != _ACTION_DOWN:
            return False
        print("Current T (4x4):\n", np.array2string(T, precision=4, suppress_small=True))
        return False

    for key, dx, dy, dz in [
        ("W", 0, 1, 0),
        ("S", 0, -1, 0),
        ("A", -1, 0, 0),
        ("D", 1, 0, 0),
        ("Q", 0, 0, 1),
        ("E", 0, 0, -1),
    ]:
        vis.register_key_action_callback(ord(key), make_key_cb(dx, dy, dz))

    for key, rx, ry, rz in [
        ("I", 1, 0, 0),
        ("K", -1, 0, 0),
        ("J", 0, 1, 0),
        ("L", 0, -1, 0),
        ("U", 0, 0, 1),
        ("O", 0, 0, -1),
    ]:
        vis.register_key_action_callback(ord(key), make_key_cb(0, 0, 0, rx, ry, rz))

    vis.register_key_action_callback(ord("P"), print_pose)

    vis.create_window("Manual pose: WASD/QE translate, IJK/UO rotate, P print, close=accept", 1280, 900)
    vis.add_geometry(tgt_vis)
    vis.add_geometry(vis_mesh)

    print(
        "\n[Manual alignment] Close the window when done.\n"
        "  WASD / Q,E : translate (same units as point cloud, ~mm)\n"
        "  I,K / J,L / U,O : rotate X / Y / Z (deg)\n"
        "  P : print 4x4 matrix\n"
    )

    vis.run()
    vis.destroy_window()
    return T
