"""
Download multi-slice CT DICOM (dcm_qa_ct GE) and build a paired STL from the same volume.

DICOM: https://github.com/neurolabusc/dcm_qa_ct (In/GE, 28 slices, BSD-style repo LICENSE).
STL:   iso-surface at Otsu threshold on the loaded volume (same coordinate system as DICOM).

Output:
  sample_data/paired_ge_ct/dicom/*.dcm
  sample_data/paired_ge_ct/surface_from_volume.stl

Usage:
  .\\.venv\\Scripts\\python scripts\\download_paired_ge_ct.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DICOM = ROOT / "sample_data" / "paired_ge_ct" / "dicom"
OUT_STL = ROOT / "sample_data" / "paired_ge_ct" / "surface_from_volume.stl"
README = ROOT / "sample_data" / "paired_ge_ct" / "README.txt"

BASE = (
    "https://raw.githubusercontent.com/neurolabusc/dcm_qa_ct/master/In/GE/"
)
SLICES = tuple(f"{i:02d}.dcm" for i in range(1, 29))


def _download(url: str, dest: Path, timeout_s: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "dicom-paired-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        dest.write_bytes(resp.read())


def _volume_to_stl(img, stl_path: Path) -> None:
    import numpy as np
    import open3d as o3d
    import SimpleITK as sitk
    from skimage.filters import threshold_otsu
    from skimage.measure import marching_cubes

    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    if arr.size == 0:
        raise RuntimeError("Empty volume")

    level = float(threshold_otsu(arr))
    spacing = img.GetSpacing()  # sx, sy, sz for ITK dims x,y,z
    # numpy array is indexed [z, y, x] -> marching_cubes spacing order matches axes 0,1,2
    spacing_zyx = (spacing[2], spacing[1], spacing[0])

    verts_zyx, faces, normals, _ = marching_cubes(arr, level=level, spacing=spacing_zyx)

    sx, sy, sz = spacing
    pts = []
    for v in verts_zyx:
        iz, iy, ix = float(v[0]) / sz, float(v[1]) / sy, float(v[2]) / sx
        pts.append(img.TransformContinuousIndexToPhysicalPoint((ix, iy, iz)))
    phy = np.asarray(pts, dtype=np.float64)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(phy)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    ok = o3d.io.write_triangle_mesh(str(stl_path), mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to write STL: {stl_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-dicom",
        action="store_true",
        help="Reuse existing DICOM under sample_data/paired_ge_ct/dicom if present.",
    )
    args = ap.parse_args()

    if not args.skip_dicom:
        print("Downloading GE CT series (28 slices)...")
        for name in SLICES:
            url = BASE + name
            dest = OUT_DICOM / name
            print(" ", name)
            try:
                _download(url, dest)
            except (urllib.error.URLError, OSError) as e:
                print(f"ERROR: {url}: {e}", file=sys.stderr)
                return 1

    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(OUT_DICOM))
    if not names:
        print(f"ERROR: no DICOM series in {OUT_DICOM}", file=sys.stderr)
        return 1
    reader.SetFileNames(names)
    img = reader.Execute()

    print("Building STL (Otsu iso-surface, same volume)...")
    try:
        _volume_to_stl(img, OUT_STL)
    except Exception as e:
        print(f"ERROR: STL build: {e}", file=sys.stderr)
        return 1

    README.parent.mkdir(parents=True, exist_ok=True)
    README.write_text(
        "\n".join(
            [
                "Paired sample: GE CT from dcm_qa_ct (neurolabusc/dcm_qa_ct, In/GE).",
                "STL: surface_from_volume.stl generated from the same SimpleITK volume (Otsu + marching cubes).",
                "Same patient-frame geometry: use main.py --preset paired-ge (disables auto-scale).",
                "",
            ]
        ),
        encoding="ascii",
    )
    print("Wrote", OUT_STL.relative_to(ROOT))
    print("Wrote", README.relative_to(ROOT))
    print("Done. Run: .\\.venv\\Scripts\\python main.py --preset paired-ge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
