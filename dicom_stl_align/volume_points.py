"""Convert SimpleITK scalar volume to Open3D point cloud (physical coordinates)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import open3d as o3d
import SimpleITK as sitk


def _mask_from_image(
    image: sitk.Image,
    mode: Literal["otsu", "percentile"],
    percentile: float,
) -> sitk.Image:
    if mode == "otsu":
        return sitk.OtsuThreshold(image, 0, 1, 200)
    arr = sitk.GetArrayFromImage(image)
    flat = arr.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError("Empty image array")
    thr = float(np.percentile(flat, percentile))
    return sitk.Cast(image > thr, sitk.sitkUInt8)


def image_to_point_cloud(
    image: sitk.Image,
    *,
    mask_mode: Literal["otsu", "percentile"] = "otsu",
    percentile: float = 85.0,
    max_points: int = 60_000,
    seed: int = 0,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Sample voxels inside foreground mask; return (pcd, centroid_of_sampled_points).

    Coordinates are SimpleITK physical (LPS mm for typical CT).
    """
    mask = _mask_from_image(image, mask_mode, percentile)
    mask_arr = sitk.GetArrayFromImage(mask)  # z, y, x
    idx = np.argwhere(mask_arr > 0)
    if idx.size == 0:
        raise RuntimeError("Foreground mask is empty; try mask_mode='percentile' or adjust percentile.")

    rng = np.random.default_rng(seed)
    if idx.shape[0] > max_points:
        sel = rng.choice(idx.shape[0], size=max_points, replace=False)
        idx = idx[sel]

    pts = np.zeros((idx.shape[0], 3), dtype=np.float64)
    for i, (iz, iy, ix) in enumerate(idx):
        pts[i] = image.TransformIndexToPhysicalPoint((int(ix), int(iy), int(iz)))

    centroid = pts.mean(axis=0)
    pts -= centroid

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
    )
    return pcd, centroid
