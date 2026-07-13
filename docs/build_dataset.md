# Building the head-fixed combined dataset

This walks through the exact commands used to build `data/head-fixed_v2`, and — more
importantly — *why* the dataset is shaped the way it is. The goal of the head-fixed
experiment is to answer: **does adding more datasets to the training mix improve or
degrade pose estimation performance**, per dataset and overall? Everything about how
these CSVs are built follows from making that comparison fair.

---

## Design principle: a balanced per-dataset frame budget

If a merged dataset (e.g. `face+ibl+cheese`) simply concatenated every available frame
from every source dataset, a merged model's advantage over a single-dataset model would
be a tangle of two effects: (1) it saw more *total* frames, and (2) it saw more *varied*
frames. We only care about (2) — whether cross-dataset variety improves generalization —
so every dataset that goes into a merge contributes the **same number of frames**.
Concretely, `face+ibl` (facemap + ibl, 600 frames each) has the same total training set
size as `facemap-600` alone, so any pixel-error difference between them is attributable
to the source mix, not the amount of data.

This is also why every single-dataset condition has a matching `<dataset>-600` tag
(`facemap-600`, `ibl-600`, `cheese-2d-600`) *in addition to* the full unsampled
`<dataset>` CSV (`facemap`, `ibl`, `cheese-2d`, which each contain that dataset's entire
train split). The `-600` versions are the actual single-dataset baselines used in
comparisons against merges — they're capped at the same pool size so the "does merging
help" comparison is apples-to-apples. The full/uncapped CSVs exist for the cases where
you *do* want a dataset's full available labels (e.g. as an upper-bound reference).

## Two layers of "frame count": build-time pool vs. train-time subsample

There are two separate knobs, easy to conflate:

1. **`build_dataset.py --n_frames`** — sets the size of the CSV *pool* written to disk.
   This is a one-time, fairly slow-to-regenerate artifact.
2. **`train_sweep.py --train_frames`** — Lightning Pose's `training.train_frames` config
   value, which subsamples *from* that pool at the start of each training run.

We build the pool at `--n_frames 600` and then sweep `--train_frames 200,400,600` at
training time, rather than building three separate 200/400/600 CSVs per tag. This means
the pool size must be **at least as large as the largest value you intend to sweep over**.
600 is also the practical ceiling here: `cheese-2d` only has 665 labeled train frames
total, so 600 is the largest round number that still fits inside every source dataset's
available pool (facemap: 1800, ibl: 5962, cheese-2d: 665) while leaving room for the
train/val split.

**200 frames is the most informative single point for comparing dataset combinations.**
It's an arbitrary but standard choice — small enough that a single dataset alone gives
"reasonable but not great" performance, which is exactly the regime where mixing in other
datasets has the most room to show a benefit. At 600 frames a single dataset is often
already performing well, so the marginal value of merging is harder to see. If you only
have budget to train one point per tag, use `--train_frames 200`; sweep 200/400/600 when
you want the full learning-curve picture (this is what `results/head-fixed_v1` did).

## Test sets are never subsampled

`build_dataset.py` always writes the *full*, unsampled test split for every dataset in a
tag — only the train split is balanced. Evaluation should use every available labeled
test frame for statistical power; there's no reason to throw test frames away for
balance, since evaluation happens per-dataset (`eval/<dataset>/pixel_error.csv`) rather
than on the merged test set as a whole.

---

## Commands run to build `data/head-fixed_v2`

`build_dataset.py` has no `--data_dir` override (unlike `convert_dataset.py`) — it always
writes to whatever `data_dir` in `paths.yaml` currently points at. The convention is to
leave `paths.yaml` pointed at the unversioned working path (`data/head-fixed`), run the
full pipeline there, then rename the directory once everything succeeds:

```bash
# 1. Convert each raw dataset (slow — copies images). Run once each, or whenever a
#    dataset's raw labels change (this rebuild was triggered by relabeling `ibl`).
conda run -n pose python scripts/convert_dataset.py --dataset facemap
conda run -n pose python scripts/convert_dataset.py --dataset ibl
conda run -n pose python scripts/convert_dataset.py --dataset cheese-2d

# 2. Single-dataset baselines, capped to the shared pool size.
conda run -n pose python scripts/build_dataset.py --tag facemap-600   --datasets facemap             --n_frames 600
conda run -n pose python scripts/build_dataset.py --tag ibl-600       --datasets ibl                 --n_frames 600
conda run -n pose python scripts/build_dataset.py --tag cheese-2d-600 --datasets cheese-2d            --n_frames 600

# 3. Pairwise and triple merges, same pool size per dataset.
conda run -n pose python scripts/build_dataset.py --tag face+cheese     --datasets facemap cheese-2d     --n_frames 600
conda run -n pose python scripts/build_dataset.py --tag face+ibl       --datasets facemap ibl           --n_frames 600
conda run -n pose python scripts/build_dataset.py --tag cheese+ibl     --datasets cheese-2d ibl         --n_frames 600
conda run -n pose python scripts/build_dataset.py --tag face+ibl+cheese --datasets facemap ibl cheese-2d --n_frames 600

# 4. Freeze this build as a version.
mv data/head-fixed data/head-fixed_v2
```

Seed is left at the `build_dataset.py` default (42) throughout, so frame selection is
reproducible and comparable across tags — the docstring in `build_dataset.py` guarantees
the same `--seed` + dataset name always samples the same frames regardless of which other
datasets are included in a given tag.

### Result: 10 tags, 20 CSVs

| Tag | Datasets | Train frames | Purpose |
|---|---|---|---|
| `facemap` / `ibl` / `cheese-2d` | single (full) | 1800 / 5962 / 665 | upper-bound reference, not used in the balanced comparison |
| `facemap-600` / `ibl-600` / `cheese-2d-600` | single (capped) | 600 / 600 / 600 | single-dataset baselines |
| `face+cheese` / `face+ibl` / `cheese+ibl` | pairwise | 1200 | two-dataset merges |
| `face+ibl+cheese` | all three | 1800 | three-dataset merge |

Note there's no `all` tag — in the previous build (`data/head-fixed_v1`) `all` and
`face+ibl+cheese` were identical (both the facemap+ibl+cheese-2d merge), just under two
different names. `face+ibl+cheese` is kept since it follows the same `<short>+<short>`
naming convention as the pairwise tags.

## Why `ibl` and not `ibl-paw`

`ibl-paw` (wrist-only) is deprecated as of this rebuild — `ibl` is a strict superset
(wrist + pupil_center + nose_tip + tongue_end, and as of July 2026 fully human-reviewed
rather than pseudo-labeled) and should always be used instead of `ibl-paw` going forward.
`_raw/ibl-paw` still exists on disk only because it's the input for regenerating `ibl`'s
face-keypoint pseudo-labels (see `scripts/preprocessing/ibl-face/README.md`) — don't
convert/build with it directly.
