# DICOM–STL alignment

PoC: load DICOM, load STL, Open3D visualization, keyboard rigid pre-alignment, ICP, before/after overlay. **Not for clinical or production use**—algorithm and workflow validation only.

## Environment

- Python 3.10+ (3.12 recommended)
- Install: `pip install -r requirements.txt`

## Quick start

```bash
# Default: single-slice WG04 CT + Stanford Bunny (demo only, no anatomical pairing)
python main.py

# Paired: multi-slice GE CT + STL surface from the same volume (bundled under sample_data/paired_ge_ct)
python main.py --preset paired-ge

# Headless smoke run
python main.py --preset paired-ge --skip-manual --skip-preview
```

Useful flags: `--dicom` (file or folder), `--stl`, `--skip-manual`, `--skip-preview`, `--no-auto-scale`, `--mesh-points`, `--max-volume-points`.

## Data scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_paired_ge_ct.py` | Download `dcm_qa_ct` GE series (28 slices), build co-registered `surface_from_volume.stl` locally |

See `sample_data/paired_ge_ct/README.txt` for pairing notes.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License & data

- MIT LICENSE
- `dcm_qa_ct` and other sample assets follow their upstream licenses.
