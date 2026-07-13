#!/usr/bin/env python3
"""
Plot labeled keypoints overlaid on frames for visual inspection.

Reads a DLC-format CSV (3-row header: scorer / bodyparts / coords), randomly
samples frames, and saves annotated PNGs to an output directory.

Usage:
    # N random frames, mirroring session/frame directory structure
    python plot_keypoints.py \\
        --csv      /path/to/CollectedData_all.csv \\
        --data_dir /path/to/dataset_root \\
        --out_dir  /path/to/output \\
        [--n_frames 100] [--seed 42]

    # One frame per session, all in a single flat directory
    python plot_keypoints.py \\
        --csv      /path/to/CollectedData_all.csv \\
        --data_dir /path/to/dataset_root \\
        --out_dir  /path/to/output \\
        --one_per_session --flat

The images referenced in the CSV index are expected at <data_dir>/<index_path>.
In flat mode, output filenames are <session>__<frame>.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Distinct colors for up to ~20 keypoints; cycles if more are present.
_COLORS = plt.cm.tab20.colors


def load_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)


def plot_frame(
    img: np.ndarray,
    keypoints: dict[str, tuple[float, float]],
    title: str = "",
) -> plt.Figure:
    """
    Draw keypoint scatter + name labels on a single frame.

    Args:
        img:       HxW or HxWxC numpy array.
        keypoints: {name: (x, y)} — NaN values are skipped.
        title:     Optional figure title (e.g. the frame path).

    Returns:
        Matplotlib Figure. Caller is responsible for closing it.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.set_title(title, fontsize=7, wrap=True)
    ax.axis("off")

    for i, (name, (x, y)) in enumerate(keypoints.items()):
        if np.isnan(x) or np.isnan(y):
            continue
        color = _COLORS[i % len(_COLORS)]
        ax.scatter(x, y, s=40, color=color, edgecolors="white", linewidths=0.5, zorder=3)
        ax.text(x + 4, y - 4, name, fontsize=6, color=color,
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.45, lw=0),
                zorder=4)

    fig.tight_layout(pad=0.5)
    return fig


def _sample(df: pd.DataFrame, n_frames: int, one_per_session: bool, seed: int) -> pd.DataFrame:
    """Return the subset of df to plot."""
    rng = np.random.default_rng(seed)
    if one_per_session:
        sessions = np.array([Path(p).parts[-2] for p in df.index])
        chosen = []
        for session in dict.fromkeys(sessions):  # preserves order, dedups
            idxs = np.where(sessions == session)[0]
            chosen.append(int(rng.choice(idxs)))
        return df.iloc[chosen]
    else:
        n = min(n_frames, len(df))
        return df.iloc[rng.choice(len(df), size=n, replace=False)]


def plot_labeled_frames(
    csv_path: Path,
    data_dir: Path,
    out_dir: Path,
    n_frames: int = 100,
    seed: int = 42,
    one_per_session: bool = False,
    flat: bool = False,
) -> None:
    """
    Sample frames from a DLC CSV and save annotated images.

    Args:
        csv_path:       Path to CollectedData CSV (DLC 3-row header format).
        data_dir:       Root directory; frame paths in the CSV index are relative to this.
        out_dir:        Output root for annotated images.
        n_frames:       Number of frames to sample (ignored when one_per_session=True).
        seed:           RNG seed for reproducible sampling.
        one_per_session: Sample exactly one frame per session instead of n_frames total.
        flat:           Save all images directly in out_dir (no subdirectories).
                        Filenames become <session>__<frame>.png.
    """
    df = load_csv(csv_path)
    scorer = df.columns.get_level_values(0)[0]
    bodyparts = list(df.columns.get_level_values(1).unique())

    sample = _sample(df, n_frames, one_per_session, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(sample)} annotated frames to {out_dir}")

    for frame_path, row in sample.iterrows():
        img_path = data_dir / frame_path
        if not img_path.exists():
            print(f"  WARNING: image not found, skipping: {img_path}")
            continue

        img = np.array(Image.open(img_path))

        keypoints = {}
        for bp in bodyparts:
            x = row.get((scorer, bp, "x"), np.nan)
            y = row.get((scorer, bp, "y"), np.nan)
            x = float(x) if pd.notna(x) else np.nan
            y = float(y) if pd.notna(y) else np.nan
            keypoints[bp] = (x, y)

        fig = plot_frame(img, keypoints, title=frame_path)

        parts = Path(frame_path).parts  # ('labeled-data', session, frame)
        session = parts[-2] if len(parts) >= 2 else "unknown"
        frame   = parts[-1]

        if flat:
            save_path = out_dir / f"{session}__{frame}"
        else:
            save_path = out_dir / session / frame
            save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DLC keypoints overlaid on labeled frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv", required=True, type=Path, help="Path to CollectedData CSV")
    parser.add_argument(
        "--data_dir", required=True, type=Path, help="Dataset root (images are relative to this)",
    )
    parser.add_argument(
        "--out_dir", required=True, type=Path, help="Output directory for annotated images",
    )
    parser.add_argument(
        "--n_frames", type=int, default=100,
        help="Frames to sample (default: 100); ignored with --one_per_session",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--one_per_session", action="store_true", help="Sample one frame per session",
    )
    parser.add_argument(
        "--flat", action="store_true", help="Save all images flat in out_dir (no subdirectories)",
    )
    args = parser.parse_args()

    plot_labeled_frames(
        csv_path=args.csv,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        n_frames=args.n_frames,
        seed=args.seed,
        one_per_session=args.one_per_session,
        flat=args.flat,
    )


if __name__ == "__main__":
    main()
