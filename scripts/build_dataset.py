#!/usr/bin/env python3
"""
Build the combined training dataset by subsampling from pre-converted per-dataset CSVs.

Reads CollectedData_<dataset>_train.csv and CollectedData_<dataset>_test.csv
from data_dir (produced by convert_dataset.py), subsamples train frames, and
merges across datasets. Keypoints absent from a given dataset get visible=0.

Produces (in data_dir):
  CollectedData_<tag>_train.csv  merged train labels
  CollectedData_<tag>_test.csv   merged test labels (all frames, no subsampling)

Usage:
  python scripts/build_dataset.py --tag all
  python scripts/build_dataset.py --tag face+ibl --datasets facemap ibl --n_frames 200
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from mouse_pose.paths import load_paths

_paths   = load_paths()
DATA_DIR = Path(_paths["data_dir"])

ALL_DATASETS = ["facemap", "ibl", "cheese-2d"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)


def build_merged(per_dataset_dfs: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Concatenate per-dataset DataFrames. All CSVs share identical columns so a plain concat suffices."""
    return pd.concat([df for _, df in per_dataset_dfs])


def _dataset_rng(seed: int, name: str) -> np.random.Generator:
    """Independent RNG per dataset — same frames regardless of which other datasets are included."""
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2 ** 32)
    return np.random.default_rng([seed, h])


def main(datasets: list[str], n_frames: int, seed: int, tag: str) -> None:
    train_dfs: list[tuple[str, pd.DataFrame]] = []
    test_dfs:  list[tuple[str, pd.DataFrame]] = []

    for name in datasets:
        train_csv = DATA_DIR / f"CollectedData_{name}_train.csv"
        test_csv  = DATA_DIR / f"CollectedData_{name}_test.csv"

        if not train_csv.exists():
            print(f"WARNING: {train_csv.name} not found — skipping {name}")
            continue

        print(f"\n── {name} ──────────────────────────────────────")
        train_df = read_csv(train_csv)
        print(f"  Train: {len(train_df)} frames available")

        n = min(n_frames, len(train_df))
        if n < n_frames:
            print(f"  WARNING: only {n} frames available (requested {n_frames})")
        rng    = _dataset_rng(seed, name)
        idx    = rng.choice(len(train_df), size=n, replace=False)
        sample = train_df.iloc[sorted(idx)]
        print(f"  Sampled {len(sample)} frames")
        train_dfs.append((name, sample))

        if test_csv.exists():
            test_df = read_csv(test_csv)
            print(f"  Test:  {len(test_df)} frames")
            test_dfs.append((name, test_df))
        else:
            print(f"  WARNING: {test_csv.name} not found — skipping test split for {name}")

    if not train_dfs:
        print("ERROR: no datasets loaded.")
        return

    print("\n── merged train ────────────────────────────────────")
    merged_train = build_merged(train_dfs)
    merged_train_path = DATA_DIR / f"CollectedData_{tag}_train.csv"
    merged_train.to_csv(merged_train_path)
    print(f"  {len(merged_train)} rows → {merged_train_path.name}")

    if test_dfs:
        print("\n── merged test ─────────────────────────────────────")
        merged_test = build_merged(test_dfs)
        merged_test_path = DATA_DIR / f"CollectedData_{tag}_test.csv"
        merged_test.to_csv(merged_test_path)
        print(f"  {len(merged_test)} rows → {merged_test_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build combined dataset from pre-converted per-dataset CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tag", required=True,
        help='label for output CSVs, e.g. "all" → CollectedData_all_{train,test}.csv',
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help=f"Datasets to include (default: {ALL_DATASETS})",
    )
    parser.add_argument("--n_frames", type=int, default=600, help="frames to sample per dataset (default: 600)")
    parser.add_argument("--seed",     type=int, default=42,  help="random seed (default: 42)")
    args = parser.parse_args()
    main(args.datasets, args.n_frames, args.seed, args.tag)
