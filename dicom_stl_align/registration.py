"""ICP wrapper around Open3D registration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class ICPConfig:
    max_correspondence_distance: float = 5.0
    max_iteration: int = 80
    point_to_plane: bool = False


def suggest_max_correspondence_distance(
    target: o3d.geometry.PointCloud,
    source: o3d.geometry.PointCloud,
    fraction: float = 0.12,
) -> float:
    """Heuristic max pairing distance from combined axis-aligned bounds."""
    tb = target.get_axis_aligned_bounding_box()
    sb = source.get_axis_aligned_bounding_box()
    ex = np.maximum(tb.get_extent(), sb.get_extent())
    return float(np.linalg.norm(ex) * fraction)


def run_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    cfg: ICPConfig | None = None,
) -> tuple[np.ndarray, o3d.pipelines.registration.RegistrationResult]:
    """Align *source* toward *target*; returns (4x4 transform, raw result)."""
    cfg = cfg or ICPConfig()
    if cfg.point_to_plane:
        if not source.has_normals():
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
            )
        if not target.has_normals():
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
            )
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        cfg.max_correspondence_distance,
        init_T,
        est,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.max_iteration),
    )
    return np.asarray(result.transformation), result
