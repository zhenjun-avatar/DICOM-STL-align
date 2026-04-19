"""End-to-end alignment: DICOM -> target PCD, STL -> source PCD, manual pose, ICP, views."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from dicom_stl_align import dicom_io, mesh_io, metrics, registration, volume_points
from dicom_stl_align.manual_interactive import run_manual_pose_editor
from dicom_stl_align.transforms import euler_xyz_deg_to_matrix, make_T
from dicom_stl_align.visualize import show_before_after


@dataclass
class PipelineConfig:
    dicom_path: Path
    stl_path: Path
    mask_mode: str = "otsu"  # "otsu" | "percentile"
    mask_percentile: float = 85.0
    max_volume_points: int = 50_000
    mesh_sample_points: int = 60_000
    voxel_down_size: float | None = None  # e.g. 2.0 mm; None = skip
    auto_scale_mesh_to_target: bool = True  # match bbox diagonal for mixed units (e.g. mm vs m)
    icp_max_distance: float | None = None  # None = auto from bounding boxes
    icp_iterations: int = 80
    icp_point_to_plane: bool = False
    manual_step_translate: float = 3.0
    manual_step_rotate_deg: float = 2.0
    init_rx_deg: float = 0.0
    init_ry_deg: float = 0.0
    init_rz_deg: float = 0.0
    init_tx: float = 0.0
    init_ty: float = 0.0
    init_tz: float = 0.0
    skip_manual: bool = False
    skip_preview: bool = False


class AlignmentPipeline:
    """Loads data, builds clouds, runs manual + ICP; keeps state for extensions."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.image = None
        self.target_pcd: o3d.geometry.PointCloud | None = None
        self.target_centroid: np.ndarray | None = None
        self.mesh_centered: o3d.geometry.TriangleMesh | None = None
        self.mesh_centroid: np.ndarray | None = None
        self.source_pcd: o3d.geometry.PointCloud | None = None
        self.T_manual: np.ndarray | None = None
        self.T_icp: np.ndarray | None = None
        self._icp_max_dist: float | None = None
        self._icp_reg: object | None = None

    def load_dicom(self):
        self.image = dicom_io.load_sitk_image(self.cfg.dicom_path)
        hdr = dicom_io.summarize_dicom_header(self.cfg.dicom_path)
        if hdr:
            print(hdr)

    def build_target_point_cloud(self):
        assert self.image is not None
        try:
            self.target_pcd, self.target_centroid = volume_points.image_to_point_cloud(
                self.image,
                mask_mode=self.cfg.mask_mode,
                percentile=self.cfg.mask_percentile,
                max_points=self.cfg.max_volume_points,
            )
        except RuntimeError:
            if self.cfg.mask_mode != "percentile":
                print("Otsu mask empty; retrying with percentile mask.")
                self.target_pcd, self.target_centroid = volume_points.image_to_point_cloud(
                    self.image,
                    mask_mode="percentile",
                    percentile=self.cfg.mask_percentile,
                    max_points=self.cfg.max_volume_points,
                )
            else:
                raise
        self.target_pcd = self._maybe_downsample(self.target_pcd)
        self._print_volume_geometry_hints()

    def _print_volume_geometry_hints(self) -> None:
        assert self.image is not None
        sx, sy, sz = self.image.GetSize()
        sp = self.image.GetSpacing()
        if sz <= 1:
            print(
                f"[Data] DICOM size (x,y,z)=({sx},{sy},{sz}), spacing={sp}. "
                "Only one slice along z: target points lie ~on a single plane."
            )
        stl_name = Path(self.cfg.stl_path).name.lower()
        if "surface_from_volume" in stl_name:
            print(
                "[Data] STL is derived from the same DICOM volume (same patient-space geometry). "
                "Expect small residual after centroid steps; ICP should refine modestly."
            )
        else:
            print(
                "[Data] STL and CT are independent demo assets; there is no gold-standard pose. "
                "Use distance stats below to see if ICP moved the source closer to the target cloud."
            )

    def load_and_center_mesh(self):
        mesh = mesh_io.load_triangle_mesh(self.cfg.stl_path)
        self.mesh_centered, self.mesh_centroid = mesh_io.mesh_centered_copy(mesh)

    def _auto_scale_mesh_to_target(self) -> None:
        if not self.cfg.auto_scale_mesh_to_target:
            return
        assert self.mesh_centered is not None and self.target_pcd is not None
        te = self.target_pcd.get_axis_aligned_bounding_box().get_extent()
        me = self.mesh_centered.get_axis_aligned_bounding_box().get_extent()
        t_diag = float(np.linalg.norm(te))
        m_diag = float(np.linalg.norm(me))
        if m_diag < 1e-9:
            return
        s = t_diag / m_diag
        s = float(np.clip(s, 0.02, 50.0))
        self.mesh_centered.scale(s, center=(0.0, 0.0, 0.0))
        self.mesh_centered.compute_vertex_normals()

    def build_source_point_cloud(self):
        assert self.mesh_centered is not None
        self.source_pcd = mesh_io.mesh_to_point_cloud(
            self.mesh_centered,
            number_of_points=self.cfg.mesh_sample_points,
        )
        self.source_pcd = self._maybe_downsample(self.source_pcd)

    def _maybe_downsample(self, geom: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if self.cfg.voxel_down_size and self.cfg.voxel_down_size > 0:
            ds = geom.voxel_down_sample(self.cfg.voxel_down_size)
            if len(ds.points) == 0:
                raise RuntimeError("Voxel downsample removed all points; reduce voxel size.")
            ds.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=max(self.cfg.voxel_down_size * 3, 1.0), max_nn=30
                )
            )
            return ds
        return geom

    def initial_transform(self) -> np.ndarray:
        R = euler_xyz_deg_to_matrix(
            self.cfg.init_rx_deg,
            self.cfg.init_ry_deg,
            self.cfg.init_rz_deg,
        )
        return make_T(
            R,
            np.array([self.cfg.init_tx, self.cfg.init_ty, self.cfg.init_tz], dtype=np.float64),
        )

    def run_manual(self) -> np.ndarray:
        assert self.target_pcd is not None and self.mesh_centered is not None
        T0 = self.initial_transform()
        if self.cfg.skip_manual:
            self.T_manual = T0
            return T0
        self.T_manual = run_manual_pose_editor(
            self.target_pcd,
            self.mesh_centered,
            T0,
            step_translate=self.cfg.manual_step_translate,
            step_rotate_deg=self.cfg.manual_step_rotate_deg,
        )
        return self.T_manual

    def run_icp(self) -> np.ndarray:
        assert self.source_pcd is not None and self.target_pcd is not None and self.T_manual is not None
        src = copy.deepcopy(self.source_pcd)
        src.transform(self.T_manual)
        max_dist = self.cfg.icp_max_distance
        if max_dist is None or max_dist <= 0:
            max_dist = registration.suggest_max_correspondence_distance(
                self.target_pcd, src, fraction=0.12
            )
        T_icp, reg = registration.run_icp(
            src,
            self.target_pcd,
            np.eye(4),
            registration.ICPConfig(
                max_correspondence_distance=max_dist,
                max_iteration=self.cfg.icp_iterations,
                point_to_plane=self.cfg.icp_point_to_plane,
            ),
        )
        self._icp_max_dist = max_dist
        self._icp_reg = reg
        print(f"ICP fitness={reg.fitness:.4f} RMSE={reg.inlier_rmse:.4f} max_dist={max_dist:.3f}")
        self.T_icp = T_icp
        return T_icp

    def combined_transform(self) -> np.ndarray:
        assert self.T_manual is not None and self.T_icp is not None
        return self.T_icp @ self.T_manual

    def print_alignment_report(self) -> None:
        """Console-only summary: easier to judge than raw 4x4 or ICP fitness alone."""
        assert (
            self.source_pcd is not None
            and self.target_pcd is not None
            and self.T_manual is not None
            and self.T_icp is not None
        )
        src_manual = copy.deepcopy(self.source_pcd)
        src_manual.transform(self.T_manual)
        before = metrics.nearest_target_distance_stats(src_manual, self.target_pcd)

        src_final = copy.deepcopy(self.source_pcd)
        src_final.transform(self.combined_transform())
        after = metrics.nearest_target_distance_stats(src_final, self.target_pcd)

        print("\n========== Alignment quality (heuristic) ==========")
        print(metrics.format_stats("Source->target after manual only", before))
        print(metrics.format_stats("Source->target after manual + ICP", after))
        if before and after:
            rel = (after["mean"] - before["mean"]) / (before["mean"] + 1e-9) * 100.0
            if after["mean"] < before["mean"]:
                print(
                    f"Mean distance decreased by {-rel:.1f}% -> ICP likely improved overlap for this run."
                )
            else:
                print(
                    f"Mean distance changed by {rel:+.1f}% -> weak overlap, bad init, or unrelated shapes."
                )
        if self._icp_reg is not None and self._icp_max_dist is not None:
            for line in metrics.interpret_icp_inlier_line(
                float(self._icp_reg.fitness),
                float(self._icp_reg.inlier_rmse),
                float(self._icp_max_dist),
            ):
                print(line)
        _, _, sz = self.image.GetSize()
        if sz <= 1:
            print("Visual: orange mesh vs gray points; single-slice CT -> gray points ~planar.")
        else:
            print("Visual: orange mesh vs gray points (multi-slice CT -> 3D target cloud).")
        print("===================================================\n")

    def visualize_results(self):
        assert self.target_pcd is not None and self.mesh_centered is not None
        assert self.T_manual is not None and self.T_icp is not None
        T_after = self.combined_transform()
        if self.cfg.skip_preview:
            return
        show_before_after(self.target_pcd, self.mesh_centered, self.T_manual, T_after)

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (T_manual, T_icp @ T_manual)."""
        self.load_dicom()
        self.build_target_point_cloud()
        self.load_and_center_mesh()
        self._auto_scale_mesh_to_target()
        self.build_source_point_cloud()
        self.run_manual()
        self.run_icp()
        self.print_alignment_report()
        self.visualize_results()
        return self.T_manual, self.combined_transform()
