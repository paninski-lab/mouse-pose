# Head-Fixed Combined Pose Estimation Dataset

Pipeline for merging multiple labeled datasets into a single standardized training set for Lightning Pose.

---

## Workflow

### 1. Add a new raw dataset

Place the raw dataset under `_raw/<dataset-name>/` with the standard DLC layout:

```
_raw/<dataset-name>/
  labeled-data/<session>/<frame>.png
  CollectedData.csv        ← train split
  CollectedData_test.csv   ← test split
  dataset_config.yaml      ← you create this (see below)
```

### 2. Convert the dataset (run once per dataset)

```bash
python code/convert_dataset.py --dataset <dataset-name>
```

Outputs to `data/head-fixed/`:
- `CollectedData_<dataset>_train.csv`
- `CollectedData_<dataset>_test.csv`
- `labeled-data/<dataset>/<session>/<frame>.png`  (all images copied)

Re-running is safe — images are skipped if already present.

### 3. Build a merged training set (run as needed)

```bash
# All three datasets, 400 frames each
python code/build_dataset.py --tag all --n_frames 400

# Two-dataset subsets
python code/build_dataset.py --tag face+ibl   --datasets facemap ibl-paw   --n_frames 400
python code/build_dataset.py --tag face+cheese --datasets facemap cheese-2d --n_frames 400
python code/build_dataset.py --tag cheese+ibl  --datasets cheese-2d ibl-paw --n_frames 400
```

Outputs to `data/head-fixed/`:
- `CollectedData_<tag>_train.csv`
- `CollectedData_<tag>_test.csv`

Frame selection is reproducible: the same `--seed` + dataset name always produces the same frames, regardless of which other datasets are included. The default seed is 42.

### 4. Train

```bash
# Dry run to preview commands
python code/train_sweep.py --dry_run \
    --csv_files "CollectedData_facemap_train.csv;CollectedData_all_train.csv" \
    --train_frames "400;1" \
    --seeds "0;1;2"

# Full sweep
python code/train_sweep.py \
    --csv_files "CollectedData_facemap_train.csv;CollectedData_all_train.csv" \
    --train_frames "400;1" \
    --seeds "0;1;2" \
    --backbones "resnet50_animal_ap10k"
```

Results land at `results/head-fixed/<tag>/<losses>/tf<N>/<backbone>/seed<N>/`.  
Evaluation runs automatically after each model against every per-dataset test CSV.

---

## Currently converted datasets

| Dataset    | Train frames | Test frames | Notes |
|------------|-------------|-------------|-------|
| facemap    | 1800        | 100         | left-view only; bilateral kps lateralized via `{side}` |
| ibl-paw    | 5963        | 1446        | wrist only; `_right` sessions pre-flipped → relabeled left |
| cheese-2d  | 665         | 291         | all four views (L/R/BC/TC); custom visibility post-processing |

## Currently built merged datasets

| Tag          | Datasets              | Train frames |
|--------------|-----------------------|-------------|
| `all`        | facemap + ibl + cheese | 1200 (400×3) |
| `face+ibl`   | facemap + ibl-paw     | 800 (400×2) |
| `face+cheese`| facemap + cheese-2d   | 800 (400×2) |
| `cheese+ibl` | cheese-2d + ibl-paw   | 800 (400×2) |

---

## Architecture

### Two-step pipeline

```
_raw/<dataset>/                    data/head-fixed/
  dataset_config.yaml  ──convert──▶  CollectedData_<dataset>_{train,test}.csv
  CollectedData.csv                  labeled-data/<dataset>/...
  CollectedData_test.csv
                                   ──build──▶  CollectedData_<tag>_{train,test}.csv
```

**convert** (`convert_dataset.py`) is slow (copies images) and run once per dataset.  
**build** (`build_dataset.py`) is fast (CSV only) and run freely for experimentation.

### Canonical keypoint vocabulary (`keypoints.yaml`)

Single source of truth for all 37 keypoint names and their ordering. Every output CSV — per-dataset and merged — has columns in this order. Datasets that don't label a keypoint carry `visible=0` for it.

### Visibility convention

| Value | Meaning |
|-------|---------|
| `2`   | Keypoint is labeled in this frame |
| `1`   | Keypoint belongs to this dataset but is unlabeled in this frame (e.g. wrong-side view) |
| `0`   | Keypoint is not part of this dataset |

Lightning Pose's loss function is visibility-aware: `vis=0` frames are excluded from loss for that keypoint. This means single-dataset and merged models all share the same LP config (`config_head-fixed.yaml`, 37 keypoints).

### `dataset_config.yaml` format

```yaml
exclude:
  keypoints: [kp1, kp2]   # source keypoints to drop entirely
  sessions: [sess1, ...]   # sessions to exclude from both splits

keypoints:
  <source_name>: <canonical_name>         # direct rename
  <source_name>: <canonical_{side}_name>  # lateralized: expands to _left + _right

sessions:
  <session_name>: left | right | null     # drives lateralization
```

**Lateralization:** if the canonical target contains `{side}`, `convert_dataset.py` expands it into two columns (`_left`, `_right`). The session's `side` value determines which column gets real coordinates; the other gets `NaN` / `vis=1`.

**`null` sessions** have no dominant side. Lateral keypoints are inapplicable; only midline keypoints get filled. Useful for head-on camera views (cheese-2d BC/TC sessions).

### Per-dataset post-processing (`convert_dataset.py`)

Some datasets need custom visibility logic beyond the standard lateralization rules. These are implemented as functions registered in `POST_PROCESS` at the top of `convert_dataset.py`:

```python
POST_PROCESS: dict[str, Callable] = {
    "cheese-2d": _post_process_cheese2d,
}
```

Each function receives the fully-processed DataFrame and the dataset config, and returns a modified DataFrame. To add post-processing for a new dataset, define a function and add one entry to `POST_PROCESS`.

**cheese-2d specifics:** Missing labels in this dataset are annotation gaps, not occlusion. Post-processing promotes `vis=1 → vis=0` for keypoints that should be visible given the session's viewpoint: visible-side keypoints in left/right sessions, and a fixed set of frontally-visible keypoints in null sessions.

### Directory layout

```
poseinterface/
  _raw/
    <dataset>/
      dataset_config.yaml
      labeled-data/<session>/<frame>.png
      CollectedData.csv
      CollectedData_test.csv

  code/
    keypoints.yaml            canonical keypoint vocabulary
    convert_dataset.py        per-dataset conversion (run once)
    build_dataset.py          subsampling + merging (run freely)
    train_sweep.py            LP training sweep + evaluation
    plots/
      plot_keypoints.py       keypoint overlay visualization

  data/head-fixed/
    config_head-fixed.yaml    LP model config (37 keypoints)
    labeled-data/
      <dataset>/<session>/<frame>.png
    CollectedData_<dataset>_{train,test}.csv   per-dataset (from convert)
    CollectedData_<tag>_{train,test}.csv       merged (from build)

  results/head-fixed/
    <tag>/<losses>/tf<N>/<backbone>/seed<N>/
      eval/<dataset>/
        predictions.csv
        pixel_error.csv
```

### Adding a new dataset

1. Place raw data under `_raw/<name>/` with the standard DLC layout
2. Create `_raw/<name>/dataset_config.yaml` (see format above)
3. Add `<name>` to `EVAL_DATASETS` in `train_sweep.py`
4. Run `python code/convert_dataset.py --dataset <name>`
5. If custom visibility logic is needed, add a function to `POST_PROCESS` in `convert_dataset.py`
6. Rebuild any merged datasets with `build_dataset.py`
