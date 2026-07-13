#!/usr/bin/env python3
"""
Create the _raw/ibl dataset using the iblvideo LP pipeline, one session at a time.

Processing one session per video ensures the ROI detection network can compute a valid
crop window (averaged over frames from a single animal in a single rig).

Steps per session:
  1. Build a stitched video: [ctx-2, ctx-1, LABELED, ctx+1, ctx+2] x n_labeled_frames,
     upscaled 4x (320x256 -> 1280x1024) to match iblvideo LEFT_VIDEO params.
  2. Run lightning_pose() on the session video.
  3. Extract predictions at labeled-frame positions (rows 2, 7, 12, ... in parquet).
  4. Scale coordinates back to 320x256 space (divide by 4).
  5. Apply likelihood threshold (>= 0.9 -> keep; < 0.9 -> NaN).
  6. Compute pupil_center_r as median of 4 pupil keypoints (NaN if any is below threshold).

Results from all sessions are merged with original paw labels and written as
DLC-format CollectedData.csv / CollectedData_test.csv in _raw/ibl/.

Session videos: _raw/ibl/session_videos/{split}/{session}/videos/_iblrig_leftCamera.raw.mp4
Parquets:       _raw/ibl/session_videos/{split}/{session}/alf/_ibl_leftCamera.lightningPose.pqt

Run in iblvideo2 conda env:
    conda run -n iblvideo2 python scripts/preprocessing/ibl-face/create_ibl_face_dataset.py

    --dry_run        Print plan; no videos or inference
    --skip_video     Skip video creation if mp4 already exists
    --skip_pipeline  Skip inference if parquet already exists
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mouse_pose.paths import load_paths

_paths = load_paths()
RAW_DIR = Path(_paths["raw_dir"])

IBL_PAW_DIR  = RAW_DIR / "ibl-paw"
IBL_DIR = RAW_DIR / "ibl"
SESSION_VIDS = IBL_DIR / "session_videos"

THRESH     = 0.9
FPS        = 60
UPSCALE    = 4           # 320x256 -> 1280x1024
VIDEO_W    = 320 * UPSCALE
VIDEO_H    = 256 * UPSCALE
CHUNK_SIZE = 5           # [ctx, ctx, LABELED, ctx, ctx]
CTX        = 2

EYE_KPS    = ["pupil_top_r", "pupil_right_r", "pupil_bottom_r", "pupil_left_r"]
NOSE_KPS   = ["nose_tip"]
TONGUE_KPS = ["tongue_end_r", "tongue_end_l"]
FACE_KPS   = ["pupil_center_r"] + NOSE_KPS + TONGUE_KPS
PAW_KPS    = ["paw_l", "paw_r"]
SCORER     = "All"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_paw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    return df[~df.index.duplicated(keep="first")]


def context_chunk(session_dir: Path, frame_name: str) -> list[Path]:
    """Return [ctx-2, ctx-1, LABELED, ctx+1, ctx+2], clamping at session edges."""
    frames = sorted(session_dir.glob("img*.png"))
    names  = [f.name for f in frames]
    try:
        pos = names.index(frame_name)
    except ValueError:
        target = session_dir / frame_name
        return [target] * CHUNK_SIZE
    return [frames[max(0, min(len(frames) - 1, pos + d))] for d in range(-CTX, CTX + 1)]


def build_session_video(session: str, session_df: pd.DataFrame, out_path: Path) -> None:
    """Write a stitched mp4 for one session's labeled frames."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (VIDEO_W, VIDEO_H))
    session_dir = IBL_PAW_DIR / "labeled-data" / session

    for rel_path in session_df.index:
        frame_name = Path(rel_path).name
        chunk = context_chunk(session_dir, frame_name)
        for fpath in chunk:
            img = cv2.imread(str(fpath))
            if img is None:
                img = np.zeros((256, 320, 3), dtype=np.uint8)
            img = cv2.resize(img, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_LINEAR)
            writer.write(img)

    writer.release()


def _find_col(df: pd.DataFrame, kp: str, coord: str) -> str:
    for suffix in (f"_{coord}", f"_{coord}_ens_median"):
        col = f"{kp}{suffix}"
        if col in df.columns:
            return col
    raise KeyError(f"No column for '{kp}' '{coord}'. First cols: {list(df.columns[:8])}")


def extract_session_preds(pqt_path: Path, n_labeled: int) -> pd.DataFrame:
    """
    Read parquet and extract predictions at labeled-frame positions (rows CTX::CHUNK_SIZE).
    Returns DataFrame with columns {kp}_x, {kp}_y, {kp}_lik; coords in 320x256 space.
    """
    pqt  = pd.read_parquet(pqt_path)
    rows = pqt.iloc[CTX::CHUNK_SIZE].reset_index(drop=True)

    if len(rows) != n_labeled:
        print(f"      WARNING: extracted {len(rows)} rows, expected {n_labeled}")

    out = pd.DataFrame(index=range(len(rows)))
    for kp in EYE_KPS + NOSE_KPS + TONGUE_KPS:
        try:
            out[f"{kp}_x"]   = rows[_find_col(pqt, kp, "x")].values / UPSCALE
            out[f"{kp}_y"]   = rows[_find_col(pqt, kp, "y")].values / UPSCALE
            out[f"{kp}_lik"] = rows[_find_col(pqt, kp, "likelihood")].values
        except KeyError as e:
            print(f"      WARNING: {e} — setting {kp} to NaN")
            out[f"{kp}_x"]   = np.nan
            out[f"{kp}_y"]   = np.nan
            out[f"{kp}_lik"] = 0.0
    return out


def apply_threshold_and_report(
    face_preds: pd.DataFrame,
    session_idx: pd.Index,
    session: str,
) -> pd.DataFrame:
    """NaN out nose/tongue x/y below THRESH (eye likelihoods are unreliable, always kept)."""
    result  = face_preds.copy()
    flagged = set()

    for kp in NOSE_KPS + TONGUE_KPS:
        lik_col = f"{kp}_lik"
        bad = result[lik_col] < THRESH
        result.loc[bad, f"{kp}_x"] = np.nan
        result.loc[bad, f"{kp}_y"] = np.nan
        for i in result.index[bad]:
            if i < len(session_idx):
                flagged.add(session_idx[i])

    if flagged:
        print(f"      {len(flagged)} frames below likelihood={THRESH} (nose/tongue):")
        for path in sorted(flagged):
            print(f"        {path}")

    return result


def compute_pupil_center(face_preds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Median of 4 pupil x/y (always trusted; EKS likelihoods for eye are unreliable)."""
    xs = np.stack([face_preds[f"{kp}_x"].values for kp in EYE_KPS], axis=1)
    ys = np.stack([face_preds[f"{kp}_y"].values for kp in EYE_KPS], axis=1)
    cx = np.nanmedian(xs, axis=1)
    cy = np.nanmedian(ys, axis=1)
    return cx, cy


def copy_images() -> None:
    src = IBL_PAW_DIR / "labeled-data"
    dst = IBL_DIR / "labeled-data"
    n_copied = n_skip = 0
    for src_f in src.rglob("*.png"):
        dst_f = dst / src_f.relative_to(src)
        if dst_f.exists():
            n_skip += 1
            continue
        dst_f.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_f, dst_f)
        n_copied += 1
    print(f"    {n_copied} copied, {n_skip} already present")


# ── main ──────────────────────────────────────────────────────────────────────

def process_split(
    split: str,
    csv_name: str,
    ckpts_path,
    lightning_pose_fn,
    dry_run: bool,
    skip_video: bool,
    skip_pipeline: bool,
) -> pd.DataFrame:
    """Process one CSV split; return merged DLC DataFrame."""
    src_csv = IBL_PAW_DIR / csv_name
    paw_df  = load_paw_csv(src_csv)

    old_scorer = paw_df.columns.get_level_values(0)[0]
    paw_df = paw_df.rename(columns={old_scorer: SCORER}, level=0)

    n_total = len(paw_df)
    print(f"  {n_total} labeled frames")

    # Group by session, preserving CSV order
    session_of = pd.Series(
        [Path(p).parts[-2] for p in paw_df.index], index=paw_df.index
    )
    sessions = list(dict.fromkeys(session_of))  # deduplicated, order-preserving
    print(f"  {len(sessions)} sessions")

    if dry_run:
        for session in sessions[:3]:
            n = int((session_of == session).sum())
            print(f"    {session}: {n} frames -> {n * CHUNK_SIZE}-frame video")
        if len(sessions) > 3:
            print(f"    ... ({len(sessions) - 3} more sessions)")
        return paw_df  # placeholder

    # Initialize output DataFrame (paw labels pre-filled, face cols NaN)
    all_kps = PAW_KPS + FACE_KPS
    face_cols = pd.MultiIndex.from_tuples(
        [(SCORER, kp, c) for kp in all_kps for c in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    result = pd.DataFrame(np.nan, index=paw_df.index, columns=face_cols)
    for kp in PAW_KPS:
        for c in ("x", "y"):
            result[(SCORER, kp, c)] = paw_df[(SCORER, kp, c)].values

    n_sessions_ok = 0
    for i_sess, session in enumerate(sessions):
        mask        = session_of == session
        session_idx = paw_df.index[mask]
        session_paw = paw_df.loc[session_idx]
        n_labeled   = len(session_paw)
        print(f"  [{i_sess + 1}/{len(sessions)}] {session}  ({n_labeled} frames)")

        vid_dir    = SESSION_VIDS / split / session / "videos"
        video_path = vid_dir / "_iblrig_leftCamera.raw.mp4"
        pqt_path   = SESSION_VIDS / split / session / "alf" / "_ibl_leftCamera.lightningPose.pqt"

        try:
            if not skip_video or not video_path.exists():
                build_session_video(session, session_paw, video_path)
                print(f"    video: {n_labeled * CHUNK_SIZE} frames")
            else:
                print(f"    video: skipped (exists)")

            if not skip_pipeline or not pqt_path.exists():
                print(f"    pipeline: running on {video_path}")
                pqt_path = lightning_pose_fn(
                    mp4_file=str(video_path),
                    ckpts_path=ckpts_path,
                    remove_files=False,
                )
                print(f"    pipeline: done -> {Path(pqt_path).name}")
            else:
                print(f"    pipeline: skipped (parquet exists)")

            face_preds = extract_session_preds(pqt_path, n_labeled)
            face_preds = apply_threshold_and_report(face_preds, session_idx, session)
            cx, cy     = compute_pupil_center(face_preds)

            result.loc[session_idx, (SCORER, "pupil_center_r", "x")] = cx
            result.loc[session_idx, (SCORER, "pupil_center_r", "y")] = cy
            result.loc[session_idx, (SCORER, "nose_tip",       "x")] = face_preds["nose_tip_x"].values
            result.loc[session_idx, (SCORER, "nose_tip",       "y")] = face_preds["nose_tip_y"].values
            result.loc[session_idx, (SCORER, "tongue_end_r",   "x")] = face_preds["tongue_end_r_x"].values
            result.loc[session_idx, (SCORER, "tongue_end_r",   "y")] = face_preds["tongue_end_r_y"].values
            result.loc[session_idx, (SCORER, "tongue_end_l",   "x")] = face_preds["tongue_end_l_x"].values
            result.loc[session_idx, (SCORER, "tongue_end_l",   "y")] = face_preds["tongue_end_l_y"].values
            n_sessions_ok += 1

        except Exception as e:
            print(f"    ERROR: {e} — face keypoints will be NaN for this session")

    print(f"  {n_sessions_ok}/{len(sessions)} sessions processed successfully")
    return result


def main(dry_run=False, skip_video=False, skip_pipeline=False):
    IBL_DIR.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        from iblvideo import download_lp_models
        from iblvideo.pose_lp import lightning_pose
        ckpts_path       = download_lp_models()
        lightning_pose_fn = lightning_pose
    else:
        ckpts_path = lightning_pose_fn = None

    splits = [
        ("train", "CollectedData.csv"),
        ("test",  "CollectedData_test.csv"),
    ]

    for split, csv_name in splits:
        print(f"\n-- {split} ({csv_name}) " + "-" * 50)
        result = process_split(
            split, csv_name, ckpts_path, lightning_pose_fn,
            dry_run, skip_video, skip_pipeline,
        )
        if dry_run:
            continue

        out_csv = IBL_DIR / csv_name
        result.to_csv(out_csv)
        n_pupil  = int((~result[(SCORER, "pupil_center_r", "x")].isna()).sum())
        n_nose   = int((~result[(SCORER, "nose_tip",       "x")].isna()).sum())
        n_tongue = int((~result[(SCORER, "tongue_end_r",   "x")].isna()).sum())
        print(f"  -> {out_csv.name}: {len(result)} rows  "
              f"pupil={n_pupil}  nose={n_nose}  tongue={n_tongue}")

    if not dry_run:
        print("\n-- copying images " + "-" * 45)
        copy_images()

    print("\nDone.")
    if not dry_run:
        print(f"\nNext steps:")
        print(f"  1. conda run -n iblvideo2 python scripts/preprocessing/ibl-face/plot_ibl_face_check.py")
        print(f"  2. Review check images in {IBL_DIR / 'labeled-data-check'}/")
        print(f"  3. conda run -n pose python scripts/convert_dataset.py --dataset ibl")
        print(f"  4. Rebuild merged datasets with scripts/build_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry_run",       action="store_true")
    parser.add_argument("--skip_video",    action="store_true",
                        help="Skip video creation if mp4 already exists for a session")
    parser.add_argument("--skip_pipeline", action="store_true",
                        help="Skip inference if parquet already exists for a session")
    args = parser.parse_args()
    main(dry_run=args.dry_run, skip_video=args.skip_video, skip_pipeline=args.skip_pipeline)
