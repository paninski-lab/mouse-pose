#!/usr/bin/env python3
"""
Cartesian training sweep over head-fixed datasets using the litpose CLI —
local, sequential.

Loops over all combinations of csv_files × backbones × train_frames × seeds
and calls `litpose train` for each, then evaluates in-process.

For remote, parallel execution on Lightning AI, see train_sweep_lightning.py —
it shares all combo/naming/command-building logic with this script via
mouse_pose.train, so the two only differ in how a command actually gets run.

Output lands at:
  <results_dir>/<tag>/<losses_tag>/tf<N>/<backbone>/seed<N>/
  └── eval/<dataset_name>/pixel_error.csv   ← per-model evaluation results

Merged CSVs (produced by build_dataset.py --tag <tag>) are evaluated against
all per-dataset test sets. Pixel error is NaN for keypoints absent from a dataset.

Usage examples:
  # dry run to preview all commands
  python scripts/train_sweep.py --dry_run \\
      --csv_files "CollectedData_facemap-600_train.csv;CollectedData_face+ibl+cheese_train.csv" \\
      --train_frames "200;400;600" \\
      --seeds "0;1;2"

  # full sweep
  python scripts/train_sweep.py \\
      --csv_files "CollectedData_facemap-600_train.csv;CollectedData_face+ibl+cheese_train.csv" \\
      --train_frames "200;400;600" \\
      --seeds "0;1;2" \\
      --backbones "vits_dino"

  # safe to re-run after interruption
  python scripts/train_sweep.py --skip_existing ...
"""

import argparse
import subprocess

from mouse_pose.train import (
    build_combos,
    csv_stem,
    evaluate_model,
    make_output_dir,
    make_train_command,
    parse_semicolon_list,
)


def main():
    parser = argparse.ArgumentParser(
        description="Head-fixed LP training sweep (local, sequential)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv_files", default="CollectedData_face+ibl+cheese_train.csv",
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

    csv_files    = parse_semicolon_list(args.csv_files)
    train_frames = parse_semicolon_list(args.train_frames)
    seeds        = parse_semicolon_list(args.seeds)
    backbones    = parse_semicolon_list(args.backbones)
    losses       = [l for l in args.losses_to_use.split(",") if l]

    combos = build_combos(csv_files, backbones, train_frames, seeds)
    print(f"Total jobs: {len(combos)}")

    if args.skip_existing and not args.eval_only:
        combos = [c for c in combos if not make_output_dir(*c, losses).exists()]
        print(f"After skipping existing: {len(combos)} remaining")

    for csv_file, backbone, train_frames_n, seed in combos:
        output_dir = make_output_dir(csv_file, backbone, train_frames_n, seed, losses)
        label = f"{csv_stem(csv_file)} | tf={train_frames_n} | {backbone} | seed={seed}"
        print(f"\n── {label}")

        if not args.eval_only:
            cmd = make_train_command(csv_file, backbone, train_frames_n, seed, losses, output_dir, args.debug)
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

    if args.dry_run:
        print(f"\n(dry run — {len(combos)} commands printed, nothing executed)")


if __name__ == "__main__":
    main()
