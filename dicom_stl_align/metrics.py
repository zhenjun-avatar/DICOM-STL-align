"""Simple geometric metrics for judging registration without ground truth."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def nearest_target_distance_stats(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> dict[str, float] | None:
    """
    For each source point, distance to nearest target point (same units as coordinates).

    Lower mean/median after ICP usually means the source moved closer to the target cloud
    (for overlapping anatomy / consistent sampling).
    """
    s = np.asarray(source.points)
    t = np.asarray(target.points)
    if s.size == 0 or t.size == 0:
        return None
    tree = cKDTree(t)
    try:
        d, _ = tree.query(s, k=1, workers=-1)
    except TypeError:
        d, _ = tree.query(s, k=1)
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "p90": float(np.percentile(d, 90)),
        "max": float(np.max(d)),
    }


def format_stats(label: str, stats: dict[str, float] | None) -> str:
    if stats is None:
        return f"{label}: (empty point set)"
    return (
        f"{label}: mean={stats['mean']:.3f}  median={stats['median']:.3f}  "
        f"p90={stats['p90']:.3f}  max={stats['max']:.3f}"
    )


def interpret_icp_inlier_line(
    fitness: float,
    inlier_rmse: float,
    max_correspondence_distance: float,
) -> list[str]:
    """Explain Open3D ICP lines already printed above (fitness / inlier_rmse / max_dist)."""
    lines = [
        "ICP fields: fitness = fraction of source points with a target neighbour within max_dist;",
        "inlier_rmse = RMSE over those inlier pairs only (not full surface error).",
    ]
    if fitness >= 0.99 and max_correspondence_distance > 20:
        lines.append(
            "When max_dist is large, fitness near 1 is common even for weak overlap; rely on mean distance above."
        )
    _ = inlier_rmse  # documented for future thresholds
    return lines
