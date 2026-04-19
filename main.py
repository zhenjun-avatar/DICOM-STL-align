"""
CLI entry: load DICOM + STL, optional keyboard manual pose, ICP, before/after overlay.

  .\\.venv\\Scripts\\python main.py
  .\\.venv\\Scripts\\python main.py --skip-manual --skip-preview

Paired CT + STL (same volume, multi-slice):
  .\\.venv\\Scripts\\python scripts\\download_paired_ge_ct.py
  .\\.venv\\Scripts\\python main.py --preset paired-ge
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dicom_stl_align.pipeline import AlignmentPipeline, PipelineConfig


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> int:
    root = _default_repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preset",
        choices=("default", "paired-ge"),
        default="default",
        help="paired-ge: multi-slice GE CT + STL from same volume (run scripts/download_paired_ge_ct.py first).",
    )
    p.add_argument("--dicom", type=Path, default=None, help="override DICOM file or series folder")
    p.add_argument("--stl", type=Path, default=None, help="override STL path")
    p.add_argument("--mask", choices=("otsu", "percentile"), default="otsu")
    p.add_argument("--percentile", type=float, default=85.0)
    p.add_argument("--max-volume-points", type=int, default=50_000)
    p.add_argument("--mesh-points", type=int, default=60_000)
    p.add_argument("--voxel-mm", type=float, default=0.0, help="0 = no voxel downsample")
    p.add_argument(
        "--icp-distance",
        type=float,
        default=0.0,
        help="0 = auto from point cloud bounds; else max correspondence distance",
    )
    p.add_argument("--icp-iters", type=int, default=80)
    p.add_argument(
        "--icp-point-to-plane",
        action="store_true",
        help="Use point-to-plane (needs good overlap / normals). Default: point-to-point.",
    )
    p.add_argument("--step-t", type=float, default=3.0, help="manual translate step")
    p.add_argument("--step-r", type=float, default=2.0, help="manual rotate step (deg)")
    p.add_argument("--rx", type=float, default=0.0)
    p.add_argument("--ry", type=float, default=0.0)
    p.add_argument("--rz", type=float, default=0.0)
    p.add_argument("--tx", type=float, default=0.0)
    p.add_argument("--ty", type=float, default=0.0)
    p.add_argument("--tz", type=float, default=0.0)
    p.add_argument("--skip-manual", action="store_true")
    p.add_argument("--skip-preview", action="store_true")
    p.add_argument(
        "--no-auto-scale",
        action="store_true",
        help="Disable bbox-based uniform scale between STL and DICOM point cloud.",
    )
    args = p.parse_args()

    if args.preset == "paired-ge":
        dicom_path = root / "sample_data" / "paired_ge_ct" / "dicom"
        stl_path = root / "sample_data" / "paired_ge_ct" / "surface_from_volume.stl"
        auto_scale = False
    else:
        dicom_path = args.dicom or (root / "sample_data" / "dicom" / "CT1_J2KR.dcm")
        stl_path = args.stl or (root / "sample_data" / "mesh" / "stanford-bunny.stl")
        auto_scale = not args.no_auto_scale

    if args.dicom is not None:
        dicom_path = args.dicom
    if args.stl is not None:
        stl_path = args.stl

    cfg = PipelineConfig(
        dicom_path=dicom_path,
        stl_path=stl_path,
        mask_mode=args.mask,
        mask_percentile=args.percentile,
        max_volume_points=args.max_volume_points,
        mesh_sample_points=args.mesh_points,
        voxel_down_size=args.voxel_mm if args.voxel_mm > 0 else None,
        icp_max_distance=args.icp_distance if args.icp_distance > 0 else None,
        icp_iterations=args.icp_iters,
        manual_step_translate=args.step_t,
        manual_step_rotate_deg=args.step_r,
        init_rx_deg=args.rx,
        init_ry_deg=args.ry,
        init_rz_deg=args.rz,
        init_tx=args.tx,
        init_ty=args.ty,
        init_tz=args.tz,
        skip_manual=args.skip_manual,
        skip_preview=args.skip_preview,
        icp_point_to_plane=args.icp_point_to_plane,
        auto_scale_mesh_to_target=auto_scale,
    )
    pipe = AlignmentPipeline(cfg)
    T_man, T_all = pipe.run()
    print("\nFinal transforms (mesh centered frame):")
    print("T_manual:\n", T_man)
    print("T_icp @ T_manual:\n", T_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
