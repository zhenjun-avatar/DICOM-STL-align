"""
Download small public DICOM fixtures + mesh samples for local experiments.

DICOM: DICOM WG04 JPEG2000 conformance objects (robyoung/dicom-test-files).
Mesh: common-3d-test-models Stanford bunny (OBJ) -> STL via Open3D.

Usage (from repo root):
  .\\.venv\\Scripts\\python scripts\\download_sample_data.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DICOM = ROOT / "sample_data" / "dicom"
OUT_MESH = ROOT / "sample_data" / "mesh"
MANIFEST = ROOT / "sample_data" / "MANIFEST.txt"

# Raw GitHub URLs (stable paths on default branch).
DICOM_FILES = [
    (
        "CT1_J2KR.dcm",
        "https://raw.githubusercontent.com/robyoung/dicom-test-files/master/data/WG04/J2KR/CT1_J2KR",
    ),
    (
        "CT2_J2KR.dcm",
        "https://raw.githubusercontent.com/robyoung/dicom-test-files/master/data/WG04/J2KR/CT2_J2KR",
    ),
    (
        "MR1_J2KR.dcm",
        "https://raw.githubusercontent.com/robyoung/dicom-test-files/master/data/WG04/J2KR/MR1_J2KR",
    ),
]

BUNNY_OBJ_URL = (
    "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/"
    "master/data/stanford-bunny.obj"
)


def _download(url: str, dest: Path, timeout_s: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "dicom-prototype-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    dest.write_bytes(data)


def _obj_to_stl(obj_path: Path, stl_path: Path) -> None:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(obj_path))
    if mesh.is_empty():
        raise RuntimeError(f"Open3D failed to read mesh: {obj_path}")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    ok = o3d.io.write_triangle_mesh(str(stl_path), mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Open3D failed to write STL: {stl_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--skip-stl",
        action="store_true",
        help="Only download DICOM; skip bunny OBJ/STL.",
    )
    args = p.parse_args()

    lines: list[str] = []

    OUT_DICOM.mkdir(parents=True, exist_ok=True)
    OUT_MESH.mkdir(parents=True, exist_ok=True)

    print("DICOM ->", OUT_DICOM)
    for name, url in DICOM_FILES:
        dest = OUT_DICOM / name
        print(f"  fetch {name}")
        try:
            _download(url, dest)
        except (urllib.error.URLError, OSError) as e:
            print(f"ERROR: {name}: {e}", file=sys.stderr)
            return 1
        lines.append(f"{dest.relative_to(ROOT)}\t{url}")

    if not args.skip_stl:
        bunny_obj = OUT_MESH / "stanford-bunny.obj"
        bunny_stl = OUT_MESH / "stanford-bunny.stl"
        print("Mesh ->", OUT_MESH)
        print("  fetch stanford-bunny.obj")
        try:
            _download(BUNNY_OBJ_URL, bunny_obj)
        except (urllib.error.URLError, OSError) as e:
            print(f"ERROR: bunny: {e}", file=sys.stderr)
            return 1
        lines.append(f"{bunny_obj.relative_to(ROOT)}\t{BUNNY_OBJ_URL}")
        print("  convert OBJ -> STL (Open3D)")
        try:
            _obj_to_stl(bunny_obj, bunny_stl)
        except Exception as e:
            print(f"ERROR: STL conversion: {e}", file=sys.stderr)
            return 1
        lines.append(f"{bunny_stl.relative_to(ROOT)}\t(local conversion from OBJ)")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        "\n".join(
            [
                "tab-separated: relative_path<TAB>source_url_or_note",
                "",
                *lines,
                "",
            ]
        ),
        encoding="ascii",
        newline="\n",
    )
    print("Wrote", MANIFEST.relative_to(ROOT))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
