#!/usr/bin/env python3
"""
Convert a raw labeled dataset into standardized format for combined training.

Reads configs/datasets/<dataset>.yaml and applies:
  - Session and keypoint exclusions
  - Bilateral keypoint lateralization (adds _left / _right variants)
  - Keypoint renaming to canonical names from configs/keypoints.yaml
  - Visibility column (2=labeled, 1=unlabeled/wrong-side, 0=not in dataset)

Produces (in data_dir from paths.yaml):
  CollectedData_<dataset>_train.csv
  CollectedData_<dataset>_test.csv
  labeled-data/<dataset>/<session>/<frame>.png  (all images, both splits)

Visibility convention:
  2 = keypoint is labeled in this frame
  1 = keypoint belongs to this dataset but is unlabeled or on the opposite side
  0 = keypoint is not part of this dataset (assigned at merge time)

Usage:
  python scripts/convert_dataset.py --dataset facemap
  python scripts/convert_dataset.py --dataset ibl --raw_dir /other/raw
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── repo root & paths ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[1]))
from mouse_pose.paths import load_paths, repo_root

_paths      = load_paths()
RAW_DIR     = Path(_paths["raw_dir"])
DATA_DIR    = Path(_paths["data_dir"])
CONFIGS_DIR = repo_root() / "configs"

SCORER     = "All"
TRAIN_CSV  = "CollectedData.csv"
TEST_CSV   = "CollectedData_test.csv"


# ── per-dataset post-processing ───────────────────────────────────────────────
#
# Each entry in POST_PROCESS maps a dataset name to a function:
#   fn(df: pd.DataFrame, config: dict) -> pd.DataFrame
#
# Called after process_split() completes (canonical names, remapped index).
# To add a new dataset: define a function below and register it in POST_PROCESS.

# cheese-2d: keypoints that are expected to be visible in null (head-on) sessions.
# Missing labels for these are annotation gaps, not occlusion → vis=0 so the
# model is not trained to suppress these heatmaps.
_CHEESE_NULL_KPS = frozenset([
    "nose_tip", "nose_top",
    "eye_front_left",  "eye_top_left",  "eye_back_left",  "eye_bottom_left",
    "ear_base_left",   "ear_top_left",  "ear_tip_left",   "ear_bottom_left",
    "pad_top_left",    "pad_side_left", "pad_center",
    "eye_front_right", "eye_top_right", "eye_back_right", "eye_bottom_right",
    "ear_base_right",  "ear_top_right", "ear_tip_right",  "ear_bottom_right",
    "pad_top_right",   "pad_side_right",
])


def _post_process_cheese2d(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg_sessions = config.get("sessions") or {}
    sides = pd.Series(
        [cfg_sessions.get(Path(p).parts[-2]) for p in df.index], index=df.index
    )
    is_null  = (sides == "null").to_numpy()
    is_left  = (sides == "left").to_numpy()
    is_right = (sides == "right").to_numpy()

    for kp in df.columns.get_level_values(1).unique():
        vis_col = (SCORER, kp, "visible")
        is_vis1 = (df[vis_col] == 1.0).to_numpy()

        if kp in _CHEESE_NULL_KPS:
            df[vis_col] = np.where(is_null & is_vis1, 0.0, df[vis_col].to_numpy())
        if kp.endswith("_left"):
            df[vis_col] = np.where(is_left & is_vis1, 0.0, df[vis_col].to_numpy())
        if kp.endswith("_right"):
            df[vis_col] = np.where(is_right & is_vis1, 0.0, df[vis_col].to_numpy())

    return df


POST_PROCESS: dict[str, object] = {
    "cheese-2d": _post_process_cheese2d,
}


# ── config loading ─────────────────────────────────────────────────────────────

def load_canonical_keypoints(configs_dir: Path) -> list[str]:
    with open(configs_dir / "keypoints.yaml") as f:
        data = yaml.safe_load(f)
    return data["keypoints"]


def _normalize_config(cfg: dict) -> dict:
    exc = cfg.setdefault("exclude", {})
    exc["sessions"]  = exc.get("sessions")  or []
    exc["keypoints"] = exc.get("keypoints") or []
    cfg.setdefault("keypoints", {})
    return cfg


def load_dataset_config(configs_dir: Path, dataset_name: str) -> dict:
    path = configs_dir / "datasets" / f"{dataset_name}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _normalize_config(cfg)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)


# ── validation ────────────────────────────────────────────────────────────────

def _kps_from_df(df: pd.DataFrame) -> set[str]:
    return set(df.columns.get_level_values(1).unique())


def _sessions_from_df(df: pd.DataFrame) -> set[str]:
    return {Path(p).parts[-2] for p in df.index}


def validate(
    config: dict,
    canonical_kps: list[str],
    train_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
) -> list[str]:
    errors: list[str] = []

    all_csv_kps: set[str] = set()
    all_csv_sessions: set[str] = set()
    for df in (train_df, test_df):
        if df is not None:
            all_csv_kps |= _kps_from_df(df)
            all_csv_sessions |= _sessions_from_df(df)

    exc_kps      = config["exclude"]["keypoints"]
    exc_sessions = config["exclude"]["sessions"]
    mapping      = config["keypoints"]
    cfg_sessions = config.get("sessions") or {}

    for kp in exc_kps:
        if kp not in all_csv_kps:
            errors.append(f"exclude.keypoints: '{kp}' not found in any CSV")

    for s in exc_sessions:
        if s not in all_csv_sessions:
            errors.append(f"exclude.sessions: '{s}' not found in any CSV")

    for src in mapping:
        if src not in all_csv_kps:
            errors.append(f"keypoints: source '{src}' not found in any CSV")

    for src, tgt in mapping.items():
        if "{side}" in tgt:
            for side in ("left", "right"):
                resolved = tgt.format(side=side)
                if resolved not in canonical_kps:
                    errors.append(
                        f"keypoints: '{src}' → '{resolved}' not in keypoints.yaml"
                    )
        else:
            if tgt not in canonical_kps:
                errors.append(
                    f"keypoints: '{src}' → '{tgt}' not in keypoints.yaml"
                )

    for s in cfg_sessions:
        if s not in all_csv_sessions:
            errors.append(f"sessions: '{s}' not found in any CSV")

    accounted = set(cfg_sessions.keys()) | set(exc_sessions)
    for s in sorted(all_csv_sessions - accounted):
        errors.append(f"CSV session '{s}' missing from sessions and exclude.sessions")

    return errors


# ── processing ────────────────────────────────────────────────────────────────

def process_split(
    df: pd.DataFrame,
    config: dict,
    dataset_name: str,
    canonical_kps: list[str],
) -> pd.DataFrame:
    """
    Apply exclusions, lateralization, renaming, and visibility to one CSV split.
    Returns a DataFrame with canonical keypoint names and remapped index paths.
    All 37 canonical columns are present; absent keypoints get visible=0.
    """
    orig_scorer  = df.columns.get_level_values(0)[0]
    exc_sessions = set(config["exclude"]["sessions"])
    exc_kps      = set(config["exclude"]["keypoints"])
    mapping      = {k: v for k, v in config["keypoints"].items() if k not in exc_kps}
    cfg_sessions = config.get("sessions") or {}

    session_of = pd.Series(
        [Path(p).parts[-2] for p in df.index], index=df.index, dtype=str
    )
    keep = ~session_of.isin(exc_sessions)
    df           = df.loc[keep]
    session_of   = session_of.loc[keep]

    is_left  = (session_of.map(cfg_sessions) == "left").to_numpy()
    is_right = (session_of.map(cfg_sessions) == "right").to_numpy()

    tuples: list[tuple] = []
    arrays: dict        = {}

    def _add(canonical: str, x, y, vis) -> None:
        arrays[(SCORER, canonical, "x")]       = x
        arrays[(SCORER, canonical, "y")]       = y
        arrays[(SCORER, canonical, "visible")] = vis
        tuples.extend([(SCORER, canonical, c) for c in ("x", "y", "visible")])

    for kp, canonical_tmpl in mapping.items():
        x_orig  = df[(orig_scorer, kp, "x")].to_numpy(dtype=float)
        y_orig  = df[(orig_scorer, kp, "y")].to_numpy(dtype=float)
        labeled = ~np.isnan(x_orig)

        if "{side}" in canonical_tmpl:
            for side, is_side in (("left", is_left), ("right", is_right)):
                canonical = canonical_tmpl.format(side=side)
                x_vals   = np.where(is_side, x_orig, np.nan)
                y_vals   = np.where(is_side, y_orig, np.nan)
                vis_vals = np.where(is_side & labeled, 2.0, 1.0)
                _add(canonical, x_vals, y_vals, vis_vals)
        else:
            vis = np.where(labeled, 2.0, 1.0)
            _add(canonical_tmpl, x_orig, y_orig, vis)

    result = pd.DataFrame(arrays, index=df.index)
    result.columns = pd.MultiIndex.from_tuples(
        tuples, names=["scorer", "bodyparts", "coords"]
    )

    present = {t[1] for t in tuples}
    all_cols = pd.MultiIndex.from_tuples(
        [(SCORER, kp, coord) for kp in canonical_kps for coord in ("x", "y", "visible")],
        names=["scorer", "bodyparts", "coords"],
    )
    result = result.reindex(columns=all_cols)
    for kp in canonical_kps:
        if kp not in present:
            result[(SCORER, kp, "visible")] = 0.0

    result.index = [
        str(Path("labeled-data") / dataset_name / Path(p).parts[-2] / Path(p).parts[-1])
        for p in result.index
    ]

    post_fn = POST_PROCESS.get(dataset_name)
    if post_fn is not None:
        result = post_fn(result, config)

    return result


def copy_images(index: pd.Index, raw_dataset_dir: Path, out_dir: Path) -> int:
    """Copy images from raw_dataset_dir into out_dir. Returns number of files copied."""
    copied = 0
    for new_path in index:
        parts   = Path(new_path).parts  # ('labeled-data', dataset, session, frame)
        session, frame = parts[-2], parts[-1]
        src = raw_dataset_dir / "labeled-data" / session / frame
        dst = out_dir / new_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a raw labeled dataset to standardized format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name (subdirectory under raw_dir, e.g. 'facemap')",
    )
    parser.add_argument("--raw_dir",    type=Path, default=RAW_DIR,  help=f"Root of raw datasets (default: from paths.yaml)")
    parser.add_argument("--data_dir",   type=Path, default=DATA_DIR, help=f"Output directory (default: from paths.yaml)")
    parser.add_argument("--train_csv",  default=TRAIN_CSV,           help=f"Train CSV filename (default: {TRAIN_CSV})")
    parser.add_argument("--test_csv",   default=TEST_CSV,            help=f"Test CSV filename (default: {TEST_CSV})")
    args = parser.parse_args()

    raw_dataset_dir = args.raw_dir / args.dataset

    print("Loading configs...")
    canonical_kps = load_canonical_keypoints(CONFIGS_DIR)
    config        = load_dataset_config(CONFIGS_DIR, args.dataset)

    train_df = load_csv(raw_dataset_dir / args.train_csv)
    test_df  = load_csv(raw_dataset_dir / args.test_csv)

    if train_df is None and test_df is None:
        print(f"ERROR: no CSVs found in {raw_dataset_dir}", file=sys.stderr)
        sys.exit(1)
    if train_df is None:
        print(f"WARNING: {args.train_csv} not found — skipping train split")
    if test_df is None:
        print(f"WARNING: {args.test_csv} not found — skipping test split")

    print("Validating config...")
    errors = validate(config, canonical_kps, train_df, test_df)
    if errors:
        print(f"\n{len(errors)} validation error(s):")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    print("  All checks passed.")

    args.data_dir.mkdir(parents=True, exist_ok=True)

    for split_name, df in (("train", train_df), ("test", test_df)):
        if df is None:
            continue
        print(f"\n── {split_name} ────────────────────────────────────────────")
        print(f"  {len(df)} frames in source CSV")

        processed = process_split(df, config, args.dataset, canonical_kps)
        n_excluded = len(df) - len(processed)
        print(f"  {n_excluded} frames excluded ({len(processed)} remaining)")

        out_csv = args.data_dir / f"CollectedData_{args.dataset}_{split_name}.csv"
        processed.to_csv(out_csv)
        print(f"  Saved {out_csv.name}")

        n_copied = copy_images(processed.index, raw_dataset_dir, args.data_dir)
        n_exist  = len(processed) - n_copied
        print(f"  Images: {n_copied} copied, {n_exist} already present")

    print("\nDone.")


if __name__ == "__main__":
    main()
