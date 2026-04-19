"""Smoke tests: load sample DICOM/STL, ICP, transforms finite (no GUI)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class TestAlignmentSmoke(unittest.TestCase):
    def test_pipeline_headless(self):
        from dicom_stl_align.pipeline import AlignmentPipeline, PipelineConfig

        dcm = ROOT / "sample_data" / "dicom" / "CT1_J2KR.dcm"
        stl = ROOT / "sample_data" / "mesh" / "stanford-bunny.stl"
        self.assertTrue(dcm.is_file(), f"missing {dcm}")
        self.assertTrue(stl.is_file(), f"missing {stl}")

        cfg = PipelineConfig(
            dicom_path=dcm,
            stl_path=stl,
            skip_manual=True,
            skip_preview=True,
            max_volume_points=8000,
            mesh_sample_points=4000,
        )
        pipe = AlignmentPipeline(cfg)
        T_man, T_all = pipe.run()

        self.assertEqual(T_man.shape, (4, 4))
        self.assertEqual(T_all.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(T_man)))
        self.assertTrue(np.all(np.isfinite(T_all)))
        np.testing.assert_allclose(T_man[-1], [0, 0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(T_all[-1], [0, 0, 0, 1], atol=1e-6)

        R = T_all[:3, :3]
        det = float(np.linalg.det(R))
        self.assertGreater(abs(det), 0.5)
        self.assertLess(abs(det), 1.5)

        self.assertGreater(len(pipe.target_pcd.points), 0)
        self.assertGreater(len(pipe.source_pcd.points), 0)

    def test_dicom_io_single_file(self):
        from dicom_stl_align.dicom_io import load_sitk_image

        dcm = ROOT / "sample_data" / "dicom" / "CT1_J2KR.dcm"
        img = load_sitk_image(dcm)
        self.assertEqual(img.GetDimension(), 3)

    def test_volume_points_non_empty(self):
        from dicom_stl_align.dicom_io import load_sitk_image
        from dicom_stl_align.volume_points import image_to_point_cloud

        dcm = ROOT / "sample_data" / "dicom" / "CT1_J2KR.dcm"
        img = load_sitk_image(dcm)
        pcd, c = image_to_point_cloud(img, max_points=5000, mask_mode="otsu")
        self.assertGreater(len(pcd.points), 0)
        self.assertEqual(c.shape, (3,))

    @unittest.skipUnless(
        (ROOT / "sample_data" / "paired_ge_ct" / "dicom" / "01.dcm").is_file()
        and (ROOT / "sample_data" / "paired_ge_ct" / "surface_from_volume.stl").is_file(),
        "run scripts/download_paired_ge_ct.py to fetch paired GE CT + STL",
    )
    def test_paired_ge_pipeline_headless(self):
        from dicom_stl_align.pipeline import AlignmentPipeline, PipelineConfig

        dicom_dir = ROOT / "sample_data" / "paired_ge_ct" / "dicom"
        stl = ROOT / "sample_data" / "paired_ge_ct" / "surface_from_volume.stl"
        cfg = PipelineConfig(
            dicom_path=dicom_dir,
            stl_path=stl,
            skip_manual=True,
            skip_preview=True,
            max_volume_points=6000,
            mesh_sample_points=4000,
            auto_scale_mesh_to_target=False,
        )
        pipe = AlignmentPipeline(cfg)
        T_man, T_all = pipe.run()
        self.assertTrue(np.all(np.isfinite(T_all)))
        self.assertGreater(len(pipe.target_pcd.points), 100)
        self.assertGreater(len(pipe.source_pcd.points), 100)


if __name__ == "__main__":
    unittest.main()
