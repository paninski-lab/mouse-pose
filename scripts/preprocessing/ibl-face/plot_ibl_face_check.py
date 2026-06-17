#!/usr/bin/env python3
"""
Write annotated check images for the ibl-face dataset.

Reads CollectedData.csv / CollectedData_test.csv from _raw/ibl-face/ and
renders one image per labeled frame to _raw/ibl-face/labeled-data-check/{split}/.

Usage:
    conda run -n iblvideo2 python scripts/preprocessing/ibl-face/plot_ibl_face_check.py
"""

from pathlib import Path

from mouse_pose.paths import load_paths
from mouse_pose.plots.plot_keypoints import plot_labeled_frames

_paths = load_paths()
IBL_FACE_DIR = Path(_paths["raw_dir"]) / "ibl-face"
CHECK_DIR    = IBL_FACE_DIR / "labeled-data-check"

SPLITS = [
    ("train", "CollectedData.csv"),
    ("test",  "CollectedData_test.csv"),
]

for split, csv_name in SPLITS:
    csv_path = IBL_FACE_DIR / csv_name
    if not csv_path.exists():
        print(f"  {split}: {csv_path} not found — skipping")
        continue
    out_dir = CHECK_DIR / split
    print(f"  {split}: all frames -> {out_dir}")
    plot_labeled_frames(
        csv_path=csv_path,
        data_dir=IBL_FACE_DIR,
        out_dir=out_dir,
        n_frames=999_999,
        flat=True,
    )

print("Done.")
