# Building the head-fixed combined dataset

This walks through the exact commands used to build `data/head-fixed_v2`, and — more
importantly — *why* the dataset is shaped the way it is. The goal of the head-fixed
experiment is to answer: **does adding more datasets to the training mix improve or
degrade pose estimation performance**, per dataset and overall? Everything about how
these CSVs are built follows from making that comparison fair.

---

## Naming convention: bare tag = full, `-600` = balanced

Every tag comes in two flavors, and the suffix tells you which:

- **Bare (`facemap`, `face+ibl`, `face+ibl+cheese`, ...)** — every dataset contributes
  *all* of its available train frames, unbalanced. Built with `build_dataset.py --n_frames -1`.
- **`-600` (`facemap-600`, `face+ibl-600`, `face+ibl+cheese-600`, ...)** — every dataset
  contributes the *same* number of frames (600), balanced. Built with `--n_frames 600`.

This one rule applies uniformly to single-dataset and merge tags alike — there's no
special case for merges.

## Design principle: why both a balanced and an unbalanced version exist

These answer two different questions, and neither is strictly "the right one":

- **Balanced (`-600`) answers: does cross-dataset *variety* help, independent of quantity?**
  If a merged dataset simply concatenated every available frame from every source
  dataset, a merged model's advantage over a single-dataset model would be a tangle of
  two effects: (1) it saw more *total* frames, and (2) it saw more *varied* frames. The
  `-600` tags isolate (2) by giving every dataset in a tag the same frame budget —
  `face+ibl-600` (600 facemap + 600 ibl) has the same total training-set size as
  `facemap-600` alone, so any pixel-error difference between them is attributable to the
  source mix, not the amount of data.
- **Full (bare) answers: what's the best model to actually deploy?** If you don't care
  about controlling for dataset size — you just want the best-performing model you can
  train from everything currently labeled — use every frame from every dataset. This is
  the practically relevant comparison once you've used the balanced numbers to decide
  *which* datasets are worth combining at all.

Concretely: `facemap` (1800 frames, full) vs. `facemap-600` (600 frames, capped) are
*not* a fair single-vs-merged comparison pair — they only differ in size, not source mix.
The apples-to-apples comparisons are `-600` vs `-600` (balanced) and bare vs. bare (full).
Don't compare a bare tag against a `-600` tag and attribute the difference to merging.

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

For the **full (bare)** tags, there's no pool to subsample from beyond what's already
there — pass `--train_frames 1` to `train_sweep.py`/`train_sweep_lightning.py`, which is
Lightning Pose's own convention for "use every frame in the CSV." Sweeping 200/400/600
against a bare tag wouldn't be meaningful since those tags aren't a fixed common pool
across dataset combinations the way the `-600` tags are (see `docs/train_sweep.md`).

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
full pipeline there, then rename the directory once everything succeeds. Adding tags to
an existing frozen version later (as in step 5 below) means pointing `data_dir` at that
version directly instead (`data/head-fixed_v2`), since there's no unversioned scratch dir
to build into anymore once it's been renamed:

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

# 3. Pairwise and triple merges, balanced (same pool size per dataset).
conda run -n pose python scripts/build_dataset.py --tag face+cheese-600     --datasets facemap cheese-2d     --n_frames 600
conda run -n pose python scripts/build_dataset.py --tag face+ibl-600       --datasets facemap ibl           --n_frames 600
conda run -n pose python scripts/build_dataset.py --tag cheese+ibl-600     --datasets cheese-2d ibl         --n_frames 600
conda run -n pose python scripts/build_dataset.py --tag face+ibl+cheese-600 --datasets facemap ibl cheese-2d --n_frames 600

# 4. Freeze this build as a version.
mv data/head-fixed data/head-fixed_v2

# 5. Added later, directly into data/head-fixed_v2: full/unbalanced merges (every frame
#    from every dataset, no cap — see "why both a balanced and unbalanced version exist"
#    above). Single-dataset full CSVs already exist from step 1, no rebuild needed there.
conda run -n pose python scripts/build_dataset.py --tag face+cheese     --datasets facemap cheese-2d     --n_frames -1
conda run -n pose python scripts/build_dataset.py --tag face+ibl       --datasets facemap ibl           --n_frames -1
conda run -n pose python scripts/build_dataset.py --tag cheese+ibl     --datasets cheese-2d ibl         --n_frames -1
conda run -n pose python scripts/build_dataset.py --tag face+ibl+cheese --datasets facemap ibl cheese-2d --n_frames -1
```

Seed is left at the `build_dataset.py` default (42) throughout, so frame selection is
reproducible and comparable across tags — the docstring in `build_dataset.py` guarantees
the same `--seed` + dataset name always samples the same frames regardless of which other
datasets are included in a given tag. (Seed is moot for the `--n_frames -1` full tags —
taking every frame doesn't involve a random choice.)

### Result: 14 tags, 28 CSVs

| Tag | Datasets | Train frames | Purpose |
|---|---|---|---|
| `facemap` / `ibl` / `cheese-2d` | single (full) | 1800 / 5962 / 665 | full-data single-dataset reference |
| `facemap-600` / `ibl-600` / `cheese-2d-600` | single (capped) | 600 / 600 / 600 | balanced single-dataset baselines |
| `face+cheese` / `face+ibl` / `cheese+ibl` | pairwise (full) | 2465 / 7762 / 6627 | full-data two-dataset merges |
| `face+cheese-600` / `face+ibl-600` / `cheese+ibl-600` | pairwise (capped) | 1200 | balanced two-dataset merges |
| `face+ibl+cheese` | all three (full) | 8427 | full-data three-dataset merge |
| `face+ibl+cheese-600` | all three (capped) | 1800 | balanced three-dataset merge |

Note there's no `all` tag — in the previous build (`data/head-fixed_v1`) `all` and
`face+ibl+cheese` were identical (both the facemap+ibl+cheese-2d merge), just under two
different names. `face+ibl+cheese-600` is kept since it follows the same `<short>+<short>`
naming convention as the pairwise tags.

## Why `ibl` and not `ibl-paw`

`ibl-paw` (wrist-only) is deprecated as of this rebuild — `ibl` is a strict superset
(wrist + pupil_center + nose_tip + tongue_end, and as of July 2026 fully human-reviewed
rather than pseudo-labeled) and should always be used instead of `ibl-paw` going forward.
`_raw/ibl-paw` still exists on disk only because it's the input for regenerating `ibl`'s
face-keypoint pseudo-labels (see `scripts/preprocessing/ibl-face/README.md`) — don't
convert/build with it directly.
