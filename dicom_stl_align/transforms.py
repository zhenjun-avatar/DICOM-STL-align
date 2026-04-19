"""Rigid transforms (4x4) built from Euler angles and translation."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 from 3x3 rotation R and translation vector t (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def euler_xyz_deg_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Intrinsic XYZ Euler in degrees -> rotation matrix."""
    return Rotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()


def delta_translation(dx: float, dy: float, dz: float) -> np.ndarray:
    return make_T(np.eye(3), np.array([dx, dy, dz], dtype=np.float64))


def delta_rotation(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    return make_T(euler_xyz_deg_to_matrix(rx_deg, ry_deg, rz_deg), np.zeros(3))
