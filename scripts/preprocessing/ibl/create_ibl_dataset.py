#!/usr/bin/env python3
"""
Merge ibl-paw paw labels with pseudo-labels from a pretrained eye+nose model
to create the _raw/ibl dataset.

Steps:
  1. Load the LP model and run predictions on every labeled frame in _raw/ibl-paw/.
  2. Filter pseudo-labels by likelihood >= THRESH.
     - Frames where any non-tube keypoint (pupil_top_r, nose_tip) is below threshold
       are printed for manual review.
     - Tube keypoints (tube_top, tube_bottom) below threshold are silently set to NaN.
  3. Merge paw labels (paw_l, paw_r) with pseudo-labels into new DLC-format CSVs.
  4. Copy images from _raw/ibl-paw/labeled-data/ to _raw/ibl/labeled-data/.
  5. Write one annotated check image per frame to _raw/ibl/labeled-data-check/.

Output:
  <raw_dir>/ibl/
    CollectedData.csv         train split (paw_l, paw_r, pupil_top_r, nose_tip, tube_top, tube_bottom)
    CollectedData_test.csv    test split
    labeled-data/<session>/<frame>.png
    labeled-data-check/<split>/<session>__<frame>.png   (flat, one per frame)

After running this script, ensure configs/datasets/ibl.yaml is configured and run:
    python scripts/convert_dataset.py --dataset ibl
    python scripts/build_dataset.py --tag <tag> --datasets ... ibl ...

Usage:
    python scripts/preprocessing/ibl/create_ibl_dataset.py [--dry_run]
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3]))
from mouse_pose.paths import load_paths
from mouse_pose.plots.plot_keypoints import plot_labeled_frames

_paths = load_paths()
RAW_DIR = Path(_paths["raw_dir"])

IBL_PAW_DIR = RAW_DIR / "ibl-paw"
IBL_OUT_DIR = RAW_DIR / "ibl"

# Path to the pretrained LP model used for pseudo-labeling.
# Update this if the model moves or you want to use a different checkpoint.
MODEL_DIR = Path("/media/mattw/behavior/results/pose-estimation/ibl-roi-detect/2023-06-21/19-47-17")

THRESH     = 0.9
PSEUDO_KPS = ["pupil_top_r", "nose_tip", "tube_top", "tube_bottom"]
TUBE_KPS   = frozenset(["tube_top", "tube_bottom"])
CHECK_KPS  = [kp for kp in PSEUDO_KPS if kp not in TUBE_KPS]  # pupil_top_r, nose_tip


# ── helpers ───────────────────────────────────────────────────────────────────

def load_paw_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)


def predict_split(model, csv_path: Path) -> pd.DataFrame:
    """Run the LP model on all frames in csv_path; returns predictions DataFrame."""
    result = model.predict_on_label_csv(
        csv_file=csv_path,
        data_dir=IBL_PAW_DIR,
        compute_metrics=False,
    )
    return result.predictions


def report_low_confidence(preds_df: pd.DataFrame, scorer: str, split: str) -> None:
    """Print frames where any CHECK_KP has likelihood < THRESH."""
    flagged = set()
    for kp in CHECK_KPS:
        try:
            liks = preds_df[(scorer, kp, "likelihood")].astype(float)
        except KeyError:
            print(f"  WARNING: keypoint '{kp}' not found in predictions")
            continue
        below = preds_df.index[liks < THRESH].tolist()
        flagged.update(below)

    if flagged:
        print(f"\n  ── {split}: {len(flagged)} frames with non-tube keypoints below "
              f"likelihood={THRESH} (manual review recommended) ──")
        for path in sorted(flagged):
            print(f"    {path}")
    else:
        print(f"  {split}: all non-tube keypoints >= {THRESH} threshold")


def merge_labels(paw_df: pd.DataFrame, preds_df: pd.DataFrame, scorer: str) -> pd.DataFrame:
    """Merge paw (x,y) and pseudo-label (x,y) columns into one DLC DataFrame.

    Pseudo-label x/y are NaN where likelihood < THRESH (all keypoints, including tube).
    Output columns ordered: paw_l, paw_r, pupil_top_r, nose_tip, tube_top, tube_bottom.
    """
    # Drop duplicate rows (the source ibl-paw train CSV has one duplicate frame)
    preds_df = preds_df[~preds_df.index.duplicated(keep="first")]
    paw_df   = paw_df[~paw_df.index.duplicated(keep="first")]

    old_scorer = paw_df.columns.get_level_values(0)[0]
    paw_df = paw_df.rename(columns={old_scorer: scorer}, level=0)

    shared_idx = preds_df.index.intersection(paw_df.index)
    n_missing = len(paw_df) - len(shared_idx)
    if n_missing:
        print(f"    WARNING: {n_missing} paw frames not found in predictions — pseudo-labels will be NaN")

    pseudo_cols = pd.MultiIndex.from_tuples(
        [(scorer, kp, coord) for kp in PSEUDO_KPS for coord in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    pseudo_block = pd.DataFrame(np.nan, index=paw_df.index, columns=pseudo_cols)

    for kp in PSEUDO_KPS:
        try:
            liks = preds_df.loc[shared_idx, (scorer, kp, "likelihood")].astype(float)
        except KeyError:
            print(f"    WARNING: '{kp}' missing from predictions — skipping")
            continue
        keep = shared_idx[liks.values >= THRESH]
        for coord in ("x", "y"):
            try:
                pseudo_block.loc[keep, (scorer, kp, coord)] = (
                    preds_df.loc[keep, (scorer, kp, coord)].values
                )
            except KeyError:
                pass

    return pd.concat([paw_df, pseudo_block], axis=1)


def copy_images() -> None:
    src = IBL_PAW_DIR / "labeled-data"
    dst = IBL_OUT_DIR / "labeled-data"
    print(f"  Copying images: {src} → {dst}")
    n_copied = n_skipped = 0
    for src_file in src.rglob("*.png"):
        dst_file = dst / src_file.relative_to(src)
        if dst_file.exists():
            n_skipped += 1
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        n_copied += 1
    print(f"    {n_copied} copied, {n_skipped} already present")


# ── main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    IBL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {MODEL_DIR}...")
    from lightning_pose.api import Model
    model = Model.from_dir(MODEL_DIR)

    splits = [
        ("train", "CollectedData.csv"),
        ("test",  "CollectedData_test.csv"),
    ]

    for split, csv_name in splits:
        print(f"\n── {split} ({csv_name}) ──────────────────────────────────────────")
        src_csv = IBL_PAW_DIR / csv_name
        paw_df  = load_paw_csv(src_csv)
        print(f"  {len(paw_df)} frames")

        print(f"  Running predictions...")
        preds_df = predict_split(model, src_csv)
        scorer   = preds_df.columns.get_level_values(0)[0]
        print(f"  Predictions scorer: '{scorer}'")

        report_low_confidence(preds_df, scorer, split)

        if not dry_run:
            merged  = merge_labels(paw_df, preds_df, scorer)
            out_csv = IBL_OUT_DIR / csv_name
            merged.to_csv(out_csv)
            n_paw    = paw_df.shape[1] // 2
            n_pseudo = len(PSEUDO_KPS)
            print(f"  → {out_csv.name}  ({len(merged)} rows, {n_paw} paw + {n_pseudo} pseudo kps)")

    if not dry_run:
        print("\n── copying images ───────────────────────────────────────────")
        copy_images()

        print("\n── visual check ─────────────────────────────────────────────")
        check_dir = IBL_OUT_DIR / "labeled-data-check"
        for split, csv_name in splits:
            out_csv = IBL_OUT_DIR / csv_name
            out_sub = check_dir / split
            print(f"  {split}: all frames → {out_sub}")
            plot_labeled_frames(
                csv_path=out_csv,
                data_dir=IBL_OUT_DIR,
                out_dir=out_sub,
                n_frames=999_999,
                flat=True,
            )

    print("\nDone.")
    if not dry_run:
        print(f"\nNext steps:")
        print(f"  1. Review annotated images in {IBL_OUT_DIR / 'labeled-data-check'}/")
        print(f"  2. Verify configs/datasets/ibl.yaml")
        print(f"  3. python scripts/convert_dataset.py --dataset ibl")
        print(f"  4. Rebuild merged datasets with scripts/build_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry_run", action="store_true",
                        help="Run predictions and print low-confidence frames, but don't write output files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
