# Replicating the head-fixed training sweep

`results/head-fixed_v1` is the completed sweep from before this dataset rebuild — trained on
`data/head-fixed_v1`, which used the now-deprecated `ibl-paw` (wrist-only) dataset. This doc gives
the exact commands to reproduce the *same sweep design* (same tags, `train_frames`, seeds, backbone)
against **`data/head-fixed_v2`** (the relabeled `ibl` dataset), so the two are comparable — plus the
commands for the new **full/unbalanced** comparison that didn't exist as a sweep before (see
[`build_dataset.md`](build_dataset.md#design-principle-why-both-a-balanced-and-an-unbalanced-version-exist)
for what "balanced" vs. "full" means and why both are useful).

If you actually want to re-run the literal old sweep unchanged (e.g. to sanity-check the
`train_sweep.py`/`train_sweep_lightning.py` refactor didn't change behavior), swap `ibl` back to
`ibl-paw` in the CSV filenames below and point `paths.yaml` at `data/head-fixed_v1` instead.

## The balanced sweep design (reproduces `results/head-fixed_v1`)

Reconstructed from what's actually on disk in `results/head-fixed_v1` (`find results/head-fixed_v1
-mindepth 4 -maxdepth 4 -type d`) — it is **not** a full cartesian product across all seven tags.
Single-dataset baselines got the full `train_frames` sweep; merges only got the higher end of it
(most likely to conserve compute, since merges have 2-3x the wall-clock cost per point). Tag names
below use the current `-600` naming (see `build_dataset.md`) — at the time `results/head-fixed_v1`
was trained, these same balanced tags were named without the suffix (`face+cheese` etc.), since the
full/unbalanced versions didn't exist yet and there was no ambiguity to disambiguate:

| Group | Tags | `--train_frames` | `--seeds` | `--backbones` | Jobs |
|---|---|---|---|---|---|
| Single-dataset baselines | `facemap-600`, `ibl-600`, `cheese-2d-600` | `200;400;600` | `0;1;2` | `vits_dino` | 27 |
| Pairwise merges | `face+cheese-600`, `face+ibl-600`, `cheese+ibl-600` | `400;600` | `0;1;2` | `vits_dino` | 18 |
| Triple merge | `face+ibl+cheese-600` | `600` | `0;1;2` | `vits_dino` | 3 |

**48 jobs total**, all `losses_to_use` empty (supervised only) — matches the 48 `train_status.json`
files under `results/head-fixed_v1`. See [`build_dataset.md`](build_dataset.md) for why the
`train_frames` ceiling is 600 and why 200 only applies to single-dataset baselines.

## Prerequisites

`paths.yaml` must have `data_dir` pointing at `data/head-fixed_v2`. For `results_dir`, follow the
versioning convention from the main README: point it at the unversioned `results/head-fixed` scratch
path, run the sweep, then rename to `results/head-fixed_v2` once everything finishes (see
[After training](#after-training) below).

```yaml
raw_dir: /media/mattw/poseinterface/_raw
data_dir: /media/mattw/poseinterface/data/head-fixed_v2
results_dir: /media/mattw/poseinterface/results/head-fixed
```

## Commands — local, sequential (`train_sweep.py`)

Always `--dry_run` first to preview the exact `litpose train` commands before committing to a
multi-hour run:

```bash
conda run -n pose python scripts/train_sweep.py --dry_run \
    --csv_files "CollectedData_facemap-600_train.csv;CollectedData_ibl-600_train.csv;CollectedData_cheese-2d-600_train.csv" \
    --train_frames "200;400;600" \
    --seeds "0;1;2" \
    --backbones "vits_dino"
```

Then the three groups for real, in any order (`--skip_existing` makes each safe to re-run if
interrupted):

```bash
# 1. Single-dataset baselines
conda run -n pose python scripts/train_sweep.py \
    --csv_files "CollectedData_facemap-600_train.csv;CollectedData_ibl-600_train.csv;CollectedData_cheese-2d-600_train.csv" \
    --train_frames "200;400;600" \
    --seeds "0;1;2" \
    --backbones "vits_dino" \
    --skip_existing

# 2. Pairwise merges
conda run -n pose python scripts/train_sweep.py \
    --csv_files "CollectedData_face+cheese-600_train.csv;CollectedData_face+ibl-600_train.csv;CollectedData_cheese+ibl-600_train.csv" \
    --train_frames "400;600" \
    --seeds "0;1;2" \
    --backbones "vits_dino" \
    --skip_existing

# 3. Triple merge
conda run -n pose python scripts/train_sweep.py \
    --csv_files "CollectedData_face+ibl+cheese-600_train.csv" \
    --train_frames "600" \
    --seeds "0;1;2" \
    --backbones "vits_dino" \
    --skip_existing
```

## Commands — Lightning AI, parallel (`train_sweep_lightning.py`)

Identical CSV/`train_frames`/seed/backbone arguments — only the launch mechanism differs.
`--machine` wasn't a consideration in the original run (it was trained locally) — `L4` below is a
reasonable default for `vits_dino`, not a reproduction of anything, adjust to what's available/cheap
on your account.

**One-time setup on the Studio:** archive `data/head-fixed_v2` and upload the archive (not the
10k-file extracted directory — file-by-file transfer/snapshotting is what was slow) to wherever this
machine's `paths.yaml` has `data_dir` pointing:

```bash
tar -cf head-fixed_v2.tar -C data head-fixed_v2
# upload head-fixed_v2.tar to the Studio, next to where data_dir should end up
```

Each job extracts this archive into place itself if `data_dir` doesn't already exist when it starts
(see `make_extract_command` in `mouse_pose/train.py`) — safe to do independently in every job since
each Lightning Job is an isolated snapshot of the Studio's filesystem, not a shared mount. `paths.yaml`
also needs `results_dir` pointing at storage that outlives an individual job (e.g. a teamspace-mounted
drive), since results need to survive after the job's compute is torn down — unlike `data_dir`, this
one *should* be shared/persistent across jobs.

```bash
pip install -e ".[lightning]"

# 1. Single-dataset baselines
python scripts/train_sweep_lightning.py \
    --csv_files "CollectedData_facemap-600_train.csv;CollectedData_ibl-600_train.csv;CollectedData_cheese-2d-600_train.csv" \
    --train_frames "200;400;600" \
    --seeds "0;1;2" \
    --backbones "vits_dino" \
    --machine L4 \
    --skip_existing

# 2. Pairwise merges
python scripts/train_sweep_lightning.py \
    --csv_files "CollectedData_face+cheese-600_train.csv;CollectedData_face+ibl-600_train.csv;CollectedData_cheese+ibl-600_train.csv" \
    --train_frames "400;600" \
    --seeds "0;1;2" \
    --backbones "vits_dino" \
    --machine L4 \
    --skip_existing

# 3. Triple merge
python scripts/train_sweep_lightning.py \
    --csv_files "CollectedData_face+ibl+cheese-600_train.csv" \
    --train_frames "600" \
    --seeds "0;1;2" \
    --backbones "vits_dino" \
    --machine L4 \
    --skip_existing
```

All three groups can be launched together (Lightning runs each Job independently, so there's no
sequencing benefit to doing them one at a time the way the local script needs). 48 jobs on a single
`L4` each is worth checking against your account's concurrent-job limit before launching all at
once.

## The full/unbalanced sweep (new comparison, not a replication of anything)

Same 6 tags as above, minus the `-600` suffix, plus the 3 single-dataset tags that already existed
before this session (`facemap`, `ibl`, `cheese-2d` — no new CSVs needed for those). Every condition
uses `--train_frames 1` (Lightning Pose's "use every frame in the CSV" convention) since there's no
common pool to sweep a learning curve over the way the `-600` tags have — see `build_dataset.md` for
why 200/400/600 doesn't apply here. 21 jobs (7 tags × 3 seeds) at one `train_frames` value each:

```bash
# dry run first
conda run -n pose python scripts/train_sweep.py --dry_run \
    --csv_files "CollectedData_facemap_train.csv;CollectedData_ibl_train.csv;CollectedData_cheese-2d_train.csv;CollectedData_face+cheese_train.csv;CollectedData_face+ibl_train.csv;CollectedData_cheese+ibl_train.csv;CollectedData_face+ibl+cheese_train.csv" \
    --train_frames "1" \
    --seeds "0;1;2" \
    --backbones "vits_dino"

# local, sequential
conda run -n pose python scripts/train_sweep.py \
    --csv_files "CollectedData_facemap_train.csv;CollectedData_ibl_train.csv;CollectedData_cheese-2d_train.csv;CollectedData_face+cheese_train.csv;CollectedData_face+ibl_train.csv;CollectedData_cheese+ibl_train.csv;CollectedData_face+ibl+cheese_train.csv" \
    --train_frames "1" \
    --seeds "0;1;2" \
    --backbones "vits_dino" \
    --skip_existing

# Lightning AI, parallel
python scripts/train_sweep_lightning.py \
    --csv_files "CollectedData_facemap_train.csv;CollectedData_ibl_train.csv;CollectedData_cheese-2d_train.csv;CollectedData_face+cheese_train.csv;CollectedData_face+ibl_train.csv;CollectedData_cheese+ibl_train.csv;CollectedData_face+ibl+cheese_train.csv" \
    --train_frames "1" \
    --seeds "0;1;2" \
    --backbones "vits_dino" \
    --machine L4 \
    --skip_existing
```

Note the triple-merge job here (`face+ibl+cheese`, 8427 frames) will take meaningfully longer per
step than the balanced `-600` conditions (max 1800 frames) — it's the same `min_steps`/`max_steps`
in `configs/model.yaml`, just more data per epoch.

## After training

Freeze the run the same way `data/head-fixed_v2` was frozen — rename the scratch results directory
to match, keeping data/results versions in lockstep:

```bash
mv results/head-fixed results/head-fixed_v2
```
