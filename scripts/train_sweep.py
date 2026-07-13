#!/usr/bin/env python3
"""
Cartesian training sweep over head-fixed datasets using the litpose CLI,
followed by per-model evaluation on held-out test CSVs.

Loops over all combinations of csv_files × backbones × train_frames × seeds
and calls `litpose train` for each, then runs evaluation.

Output lands at:
  <results_dir>/<tag>/<losses_tag>/tf<N>/<backbone>/seed<N>/
  └── eval/<dataset_name>/pixel_error.csv   ← per-model evaluation results

Merged CSVs (produced by build_dataset.py --tag <tag>) are evaluated against
all per-dataset test sets. Pixel error is NaN for keypoints absent from a dataset.

Usage examples:
  # dry run to preview all commands
  python scripts/train_sweep.py --dry_run \\
      --csv_files "CollectedData_facemap_train.csv;CollectedData_all_train.csv" \\
      --train_frames "600;1" \\
      --seeds "0;1;2"

  # full sweep
  python scripts/train_sweep.py \\
      --csv_files "CollectedData_facemap_train.csv;CollectedData_all_train.csv" \\
      --train_frames "600;1" \\
      --seeds "0;1;2" \\
      --backbones "resnet50_animal_ap10k"

  # safe to re-run after interruption
  python scripts/train_sweep.py --skip_existing ...
"""

import argparse
import shutil
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from mouse_pose.paths import load_paths, repo_root

_paths      = load_paths()
DATA_DIR    = Path(_paths["data_dir"])
RESULTS_DIR = Path(_paths["results_dir"])

CONFIG_FILE = repo_root() / "configs" / "model.yaml"

# Every model is evaluated against each dataset's test CSV. Pixel error is NaN
# for keypoints absent from a given dataset; the plotting script handles filtering.
EVAL_DATASETS = ["facemap", "ibl", "cheese-2d"]


# ── helpers ───────────────────────────────────────────────────────────────────

def csv_stem(csv_file: str) -> str:
    """CollectedData_<tag>_train.csv -> <tag>_train"""
    return Path(csv_file).stem.split("_", 1)[1]


def losses_tag(losses: list[str]) -> str:
    return "supervised" if not losses else "+".join(sorted(losses))


def sanitize(s: str) -> str:
    return s.replace("/", "_").replace(".", "_")


def make_output_dir(csv_file, backbone, train_frames, seed, losses) -> Path:
    return (
        RESULTS_DIR
        / csv_stem(csv_file)
        / losses_tag(losses)
        / f"tf{train_frames}"
        / sanitize(backbone)
        / f"seed{seed}"
    )


def make_command(csv_file, backbone, train_frames, seed, losses, output_dir, debug) -> list[str]:
    lr = 5e-5 if "vit" in backbone else 1e-3
    losses_hydra = f"[{','.join(losses)}]"

    overrides = [
        f"data.data_dir={DATA_DIR}",
        f"data.csv_file={csv_file}",
        f"model.backbone={backbone}",
        f"model.losses_to_use={losses_hydra}",
        f"training.train_frames={train_frames}",
        f"training.rng_seed_data_pt={seed}",
        f"training.optimizer_params.learning_rate={lr}",
    ]

    if "vitb_sam" in backbone:
        overrides.append("training.train_batch_size=16")

    if debug:
        overrides += [
            "training.check_val_every_n_epoch=1",
            "training.max_epochs=3",
            "training.unfreezing_epoch=1",
            "eval.predict_vids_after_training=false",
        ]

    return (
        ["litpose", "train", str(CONFIG_FILE),
         "--output_dir", str(output_dir),
         "--overrides"] + overrides
    )


def evaluate_model(output_dir: Path, csv_file: str) -> None:
    """Evaluate a trained model against every per-dataset test CSV."""
    from lightning_pose.api import Model
    from lightning_pose.metrics import pixel_error

    print(f"  Loading model...")
    model = Model.from_dir(output_dir)

    for eval_name in EVAL_DATASETS:
        test_csv = DATA_DIR / f"CollectedData_{eval_name}_test.csv"

        if not test_csv.exists():
            print(f"  WARNING: {test_csv} not found — skipping {eval_name} eval")
            continue

        print(f"  Predicting on {eval_name} test set ({test_csv.name})...")
        result = model.predict_on_label_csv(
            csv_file=test_csv,
            data_dir=DATA_DIR,
            compute_metrics=False,
        )
        preds_df = result.predictions

        labels_df = pd.read_csv(test_csv, header=[0, 1, 2], index_col=0)
        if labels_df.index[0] == labels_df.index.name:
            labels_df = labels_df.iloc[1:]

        shared_idx = preds_df.index.intersection(labels_df.index)
        if len(shared_idx) == 0:
            print(f"  WARNING: no shared frames between predictions and labels — skipping")
            continue
        preds_df  = preds_df.loc[shared_idx]
        labels_df = labels_df.loc[shared_idx]
        n = len(shared_idx)

        kps = labels_df.columns.get_level_values(1).unique().tolist()
        xy  = ["x", "y"]
        preds_arr  = preds_df.loc[:,  preds_df.columns.get_level_values(2).isin(xy)].to_numpy().reshape(n, len(kps), 2)
        labels_arr = labels_df.loc[:, labels_df.columns.get_level_values(2).isin(xy)].to_numpy().reshape(n, len(kps), 2)

        error    = pixel_error(labels_arr, preds_arr)
        error_df = pd.DataFrame(error, index=shared_idx, columns=kps)

        save_dir = output_dir / "eval" / eval_name
        save_dir.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(save_dir / "predictions.csv")
        error_df.to_csv(save_dir / "pixel_error.csv")

        mean_err = float(np.nanmean(error))
        print(f"    Mean pixel error: {mean_err:.2f} px  →  {save_dir}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Head-fixed LP training sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv_files", default="CollectedData_all_train.csv",
        help='semicolon-separated CSV filenames relative to data_dir',
    )
    parser.add_argument(
        "--train_frames", default="1",
        help='semicolon-separated frame counts; 1 = all frames',
    )
    parser.add_argument("--seeds",        default="0",                   help='semicolon-separated rng seeds, e.g. "0;1;2"')
    parser.add_argument("--backbones",    default="resnet50_animal_ap10k", help='semicolon-separated backbone names')
    parser.add_argument("--losses_to_use", default="",                   help='comma-separated loss names; empty = supervised only')
    parser.add_argument("--debug",         action="store_true",          help="Smoke-test run (3 epochs)")
    parser.add_argument("--dry_run",       action="store_true",          help="Print commands without running")
    parser.add_argument("--skip_existing", action="store_true",          help="Skip combos whose output dir already exists")
    parser.add_argument("--eval_only",     action="store_true",          help="Skip training; only run evaluation on existing model dirs")
    args = parser.parse_args()

    csv_files    = args.csv_files.split(";")
    train_frames = args.train_frames.split(";")
    seeds        = args.seeds.split(";")
    backbones    = args.backbones.split(";")
    losses       = [l for l in args.losses_to_use.split(",") if l]

    combos = list(product(csv_files, backbones, train_frames, seeds))
    print(f"Total jobs: {len(combos)}")

    if args.skip_existing and not args.eval_only:
        combos = [c for c in combos if not make_output_dir(*c, losses).exists()]
        print(f"After skipping existing: {len(combos)} remaining")

    for csv_file, backbone, train_frames_n, seed in combos:
        output_dir = make_output_dir(csv_file, backbone, train_frames_n, seed, losses)
        label = f"{csv_stem(csv_file)} | tf={train_frames_n} | {backbone} | seed={seed}"
        print(f"\n── {label}")

        if not args.eval_only:
            cmd = make_command(csv_file, backbone, train_frames_n, seed, losses, output_dir, args.debug)
            print("   " + " ".join(cmd))

            if not args.dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"  ERROR: training failed (exit {e.returncode}), skipping eval...")
                    continue

        if not args.dry_run:
            if not output_dir.exists():
                print(f"  WARNING: output dir not found, skipping eval: {output_dir}")
                continue
            try:
                evaluate_model(output_dir, csv_file)
            except Exception as e:
                print(f"  ERROR: evaluation failed: {e}")
            for fname in ("predictions.csv", "predictions_pixel_error.csv"):
                p = output_dir / fname
                if p.exists():
                    p.unlink()
            image_preds = output_dir / "image_preds"
            if image_preds.exists():
                shutil.rmtree(image_preds)

    if args.dry_run:
        print(f"\n(dry run — {len(combos)} commands printed, nothing executed)")


if __name__ == "__main__":
    main()
