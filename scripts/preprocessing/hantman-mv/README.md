# hantman-mv preprocessing

Builds `_raw/hantman-mv` (single-view) from the multi-view DLC project at
`_raw/_dlc/hantman-mv`.

**Status: stage 2 drafted, not run.** `configs/datasets/hantman-mv.yaml` exists and
validates cleanly, and `configs/keypoints.yaml`/`configs/model.yaml` already carry the
8 new canonical keypoints it needs (`d1_tip`–`d4_tip`, lateralized), but
`scripts/convert_dataset.py --dataset hantman-mv` has not been run — see
[`skills/preprocess-new-dataset/README.md`](../../../skills/preprocess-new-dataset/README.md) for what that means. Unlike `hantman`
(the older, 4-keypoint reaching dataset — see `../hantman-sleap/`), `hantman-mv` has a
fuller finger/paw/pellet(+face) skeleton and is a separate dataset entirely; nothing here
merges the two.

## Why a custom converter

The source is already standard DLC format, so no format translation is needed — but it
arrives as **two per-view CSVs** (`CollectedData_side.csv`, `CollectedData_front.csv`)
sharing an identical 17-keypoint schema, and the project's `project.yaml` declares
`view_names: [front, side]`. "Converting to single-view" means:

1. Concatenating the two CSVs — side and front session directories already encode the
   view in their names (e.g. `KPC188_20260110_v038_side` / `..._front`), so there's no
   naming collision and no remapping needed, just a row concat.
2. Writing a `project.yaml` with `view_names: []`, so the LP labeling app treats the
   result as one single-view project instead of a synchronized multi-view one.

No canonical-vocab mapping happens here — the output keypoint names are exactly the
source names (`d1_tip`, `d1_middle`, ..., `wrist`, `pellet`). That mapping is a stage-2
concern (`configs/datasets/hantman-mv.yaml`), not this script's job.

**Only CSV-referenced images are copied**, not whole `labeled-data/<session>/`
directories — each session directory on disk holds many more frames than are labeled
(unlabeled context frames around each label), so a wholesale copy would pull in far
more than what's needed (labeled-data here mixes ~2050 image files against 222 labeled
rows across both views).

## Design notes

- **Split:** subject-level (first `_`-delimited token of the session name, uppercased),
  same convention as `../hantman-sleap/`. Target is the **midpoint** of the requested
  10-15% range (0.125), not the upper edge — the greedy subject-accumulation split only
  ever overshoots its target (it stops as soon as the running total meets it), and with
  as few as ~17 subjects of uneven size, aiming at 0.15 regularly overshot past it in
  practice (one subject alone was 23% of all frames). Aiming at the midpoint leaves
  room for the overshoot while still landing in range.
- **`videos/`** is created empty in the output, matching the convention in other `_raw/`
  datasets (e.g. `kondo/`, `petersen-top/`) — the source's `videos/` has a handful of
  full session recordings (241MB) that aren't needed for LP training (only labeled
  frames are used), so they aren't copied.
- Source `labeled-data/` also contains stray `*.jpgZone.Identifier` files (Windows
  download artifacts) — ignored, not copied.

See [`CHANGELOG.md`](CHANGELOG.md) for keypoint/label history.

## Scripts

| Script | Env | Purpose |
|---|---|---|
| `convert_hantman_mv.py` | `pose` | Concats both view CSVs, splits, copies referenced images, rewrites `project.yaml` |

## Usage

```bash
conda run -n pose python scripts/preprocessing/hantman-mv/convert_hantman_mv.py

# different train/test split seed
conda run -n pose python scripts/preprocessing/hantman-mv/convert_hantman_mv.py --seed 1
```

## Stage 2 (drafted, not run — see status note above)

`configs/datasets/hantman-mv.yaml` keeps `d1_tip`/`d2_tip`/`d3_tip`/`d4_tip` (lateralized,
newly added to `configs/keypoints.yaml`/`configs/model.yaml`), `eye_back`/`eye_top`/
`eye_front`/`eye_bottom` (lateralized, already canonical), `nose_tip`/`nose_bottom`
(midline, already canonical), and the computed `wrist_new` (lateralized to
`wrist_{side}`, already canonical — see the 2026-09-15 entry in `CHANGELOG.md`). All 54 sessions are
declared `right` (single hand, per the raw source). Everything else — the digit
`_middle`/`_base` joints, `hand_middle`/`hand_lateral`/`hand_medial`, source `wrist`,
`pellet` — is excluded.

Only one side was ever filmed, so the default per-split output would mark every `_left`
counterpart of a lateralized keypoint `visible=1` ("in dataset, unlabeled") rather than
`visible=0` ("not part of this dataset") — training on that would teach the model to
predict a suppressed heatmap for a side that was simply never assessed. A
`POST_PROCESS["hantman-mv"]` entry in `scripts/convert_dataset.py` forces every `_left`
column to `visible=0` after the standard split processing, **except** the four
`ear_*_left` columns (see the 2026-09-17 entry in `CHANGELOG.md`), which are
deliberately left at `visible=1` instead. Same pattern as `cheese-2d`'s post-process
function, simpler here since there's only one side/one scoring rule instead of
per-session left/right/null.

To actually run stage 2:
```bash
conda run -n pose python scripts/convert_dataset.py --dataset hantman-mv
python scripts/build_dataset.py --tag <tag> --datasets hantman-mv ...
```
