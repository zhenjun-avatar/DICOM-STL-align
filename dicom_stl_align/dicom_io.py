"""Load DICOM series or single file as SimpleITK image (physical space in mm)."""

from __future__ import annotations

from pathlib import Path

import pydicom
import SimpleITK as sitk


def load_sitk_image(dicom_path: Path | str) -> sitk.Image:
    """Load from a single .dcm file or a directory containing one series."""
    path = Path(dicom_path)
    if path.is_file():
        return sitk.ReadImage(str(path), sitk.sitkFloat32)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesFileNames(str(path))
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found under {path}")
    reader.SetFileNames(series_ids)
    return reader.Execute()


def summarize_dicom_header(dicom_path: Path | str, max_lines: int = 8) -> str:
    """Lightweight pydicom peek (first file in folder)."""
    path = Path(dicom_path)
    if path.is_file():
        first = path
    else:
        dcms = sorted(path.glob("*.dcm"))
        first = dcms[0] if dcms else None
    if first is None:
        return ""
    ds = pydicom.dcmread(str(first), stop_before_pixels=True, force=True)
    lines = []
    for name in ("Modality", "SeriesDescription", "Rows", "Columns", "PatientID"):
        if hasattr(ds, name):
            lines.append(f"{name}: {getattr(ds, name)}")
    return "\n".join(lines[:max_lines])
