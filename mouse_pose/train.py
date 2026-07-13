"""
Shared utilities for Lightning Pose training sweeps over the head-fixed dataset.

Used by both `scripts/train_sweep.py` (local, sequential) and
`scripts/train_sweep_lightning.py` (Lightning AI, parallel), so combo
generation, naming, and command-building stay identical across backends —
only *how* a command gets executed differs between the two scripts.

Also invocable standalone to run evaluation only. This exists because a
Lightning AI Job is an independent remote process: there's no "come back to
this process after training finishes" step like the local sweep script has,
so evaluation has to be its own chainable command:

    python -m mouse_pose.train --output_dir <dir> --csv_file <CollectedData_..._train.csv>
"""

import argparse
import shutil
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


# ── naming / output dirs ─────────────────────────────────────────────────────

def csv_stem(csv_file: str) -> str:
    """CollectedData_<tag>_train.csv -> <tag>_train"""
    return Path(csv_file).stem.split("_", 1)[1]


def losses_tag(losses: list[str]) -> str:
    return "supervised" if not losses else "+".join(sorted(losses))


def sanitize(s: str) -> str:
    return str(s).replace("/", "_").replace(".", "_")


def make_output_dir(csv_file, backbone, train_frames, seed, losses) -> Path:
    return (
        RESULTS_DIR
        / csv_stem(csv_file)
        / losses_tag(losses)
        / f"tf{train_frames}"
        / sanitize(backbone)
        / f"seed{seed}"
    )


def make_job_name(csv_file, backbone, train_frames, seed, losses) -> str:
    """Lightning job name for one sweep combo (unused locally, but shares the
    same inputs as make_output_dir so job name <-> output dir stay traceable)."""
    return "__".join([
        sanitize(csv_stem(csv_file)),
        sanitize(backbone),
        losses_tag(losses),
        f"tf{train_frames}",
        f"s{seed}",
    ])


# ── sweep combo generation ───────────────────────────────────────────────────

def parse_semicolon_list(s: str) -> list[str]:
    return s.split(";")


def build_combos(csv_files, backbones, train_frames, seeds) -> list[tuple]:
    """Cartesian product of one sweep, in (csv_file, backbone, train_frames, seed) order."""
    return list(product(csv_files, backbones, train_frames, seeds))


# ── command building ─────────────────────────────────────────────────────────

def make_train_command(
    csv_file, backbone, train_frames, seed, losses, output_dir, debug=False,
) -> list[str]:
    """Build the `litpose train ...` argv for one sweep combo."""
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


def make_eval_command(output_dir, csv_file) -> list[str]:
    """Build the standalone-evaluation argv for one sweep combo (chained onto
    make_train_command's output with `&&` for a single Lightning job command)."""
    return [
        "python", "-m", "mouse_pose.train",
        "--output_dir", str(output_dir),
        "--csv_file", str(csv_file),
    ]


def make_extract_command() -> str:
    """Shell snippet that extracts DATA_DIR from a sibling `.tar` archive if
    DATA_DIR doesn't already exist, meant to run once at the start of each
    Lightning job (chained before make_train_command's output with `&&`).

    Each Lightning Job is an isolated snapshot of the launching Studio's own
    filesystem, not a shared mount multiple jobs write into concurrently — so
    unlike a HuggingFace-style dataset cache on shared storage, there's no race
    to guard against here. Every job independently checking "does this exist
    yet" and extracting its own local copy if not is safe to duplicate as-is.

    Returns a shell string (not an argv list, unlike the other make_*_command
    functions) since it needs `||` — only meaningful for Lightning jobs; the
    local sequential script never calls this because DATA_DIR is expected to
    already exist on disk for local runs.
    """
    archive = f"{DATA_DIR}.tar"
    return f'test -d "{DATA_DIR}" || tar -xf "{archive}" -C "{DATA_DIR.parent}"'


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(output_dir: Path, csv_file: str) -> None:
    """Evaluate a trained model against every per-dataset test CSV, then clean
    up the scratch prediction files litpose leaves behind in output_dir and
    delete model checkpoints (*.ckpt) — evaluation predictions/pixel-errors
    under output_dir/eval/ are what's kept long-term, not the weights."""
    from lightning_pose.api import Model
    from lightning_pose.metrics import pixel_error

    output_dir = Path(output_dir)

    print("  Loading model...")
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
            print("  WARNING: no shared frames between predictions and labels — skipping")
            continue
        preds_df  = preds_df.loc[shared_idx]
        labels_df = labels_df.loc[shared_idx]
        n = len(shared_idx)

        kps = labels_df.columns.get_level_values(1).unique().tolist()
        xy  = ["x", "y"]
        preds_cols  = preds_df.columns.get_level_values(2).isin(xy)
        labels_cols = labels_df.columns.get_level_values(2).isin(xy)
        preds_arr  = preds_df.loc[:, preds_cols].to_numpy().reshape(n, len(kps), 2)
        labels_arr = labels_df.loc[:, labels_cols].to_numpy().reshape(n, len(kps), 2)

        error    = pixel_error(labels_arr, preds_arr)
        error_df = pd.DataFrame(error, index=shared_idx, columns=kps)

        save_dir = output_dir / "eval" / eval_name
        save_dir.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(save_dir / "predictions.csv")
        error_df.to_csv(save_dir / "pixel_error.csv")

        mean_err = float(np.nanmean(error))
        print(f"    Mean pixel error: {mean_err:.2f} px  →  {save_dir}")

    for fname in ("predictions.csv", "predictions_pixel_error.csv"):
        p = output_dir / fname
        if p.exists():
            p.unlink()
    image_preds = output_dir / "image_preds"
    if image_preds.exists():
        shutil.rmtree(image_preds)

    n_deleted = 0
    for ckpt in output_dir.rglob("*.ckpt"):
        ckpt.unlink()
        n_deleted += 1
    print(f"  Deleted {n_deleted} checkpoint file(s)")


def _main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained head-fixed model against every per-dataset test CSV",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Model output_dir passed to `litpose train`",
    )
    parser.add_argument(
        "--csv_file", required=True,
        help="Train CSV filename the model was trained on (for logging only)",
    )
    args = parser.parse_args()
    evaluate_model(args.output_dir, args.csv_file)


if __name__ == "__main__":
    _main()
