# hantman preprocessing

Builds `_raw/hantman` (side + front views combined) from raw SLEAP output at
`_raw/_dlc/hantman/{side,front}_v*.slp`.

**Status: stage 1 only** (see [`skills/preprocess-new-dataset/README.md`](../../../skills/preprocess-new-dataset/README.md) for what
that means). A draft `configs/datasets/hantman.yaml` exists — mapping `wrist`/`digit2`/
`digit4` → `_right`, excluding `pellet` — but isn't usable yet since `digit2`/`digit4`
aren't in `configs/keypoints.yaml`/`model.yaml`.

## Why a custom converter

The source `.slp` files are not `.pkg.slp` packages — no embedded frame pixels, only
a `videos_json` reference to the original recording-machine path (e.g.
`D:/tracker_videos/side/JCR129_20240227_side_v002.avi`). Neither `convert_dataset.py`
nor `litpose convert` / `lightning_pose.converters.sleap` can read this directly; frame
pixels have to be pulled from the local `.avi` files instead.

`side_v13.slp` additionally mixes ~24k SLEAP-tracker *predicted* instances in with
~3k human-labeled ones (`frame.user_instances` vs. the rest). Only human-labeled
instances are converted — the tracker output is not a labeled dataset.

## Pipeline steps

For each view (`side`, `front`):

1. Load the `.slp` with `sleap_io.load_slp`.
2. Keep only frames with `user_instances` (drops SLEAP-tracker predictions).
3. Resolve each frame's video locally by basename under
   `_raw/_dlc/hantman/{side,front}/` (the `.slp` stores the original Windows path).
4. Extract the frame's pixels via `sleap_io`'s video reader at the labeled `frame_idx`.
5. Normalize keypoint node names — the two skeletons disagree ("digit 4" in `side` vs.
   "digit4" in `front`) — so both views produce identical column names.

Then, combined across both views:

6. Merge into one output — side and front session names never collide (both include
   the view in the filename), so they coexist under one `labeled-data/`.
7. Split **subject-level, pooled across both views**: subject = the first
   `_`-delimited token of a session name (`{subject}_{date}_{view}_{version}`),
   uppercased so casing slips in the source filenames (`jcr130` vs `JCR130`) don't
   split the same animal across train/test. All of a subject's sessions — side and
   front alike — land in the same split.

Output keypoints: `digit2`, `digit4`, `pellet`, `wrist`. (`pellet` — the reach target,
not a mouse body part — is dropped at the `configs/datasets/hantman.yaml` level, not
here, since exclusion is a `convert_dataset.py`-level concern.)

The source has no `project.yaml` of its own (raw SLEAP has no such concept), so one is
written from scratch — `keypoint_names` from the merged skeleton, `view_names: []`
since both views were merged into one project — plus an empty `videos/`, so the result
opens correctly in the Lightning Pose labeling app.

## Design notes (decided by the user, not inferred)

- **Reaching hand:** all sessions map `wrist`/`digit2`/`digit4` → `_right` in
  `configs/datasets/hantman.yaml` — every hantman mouse reaches with the same
  (right) paw. Not recoverable from a single frame's pixels (mirroring/camera
  convention is ambiguous), so this was confirmed with the user rather than guessed.
- **Test fraction:** ~15%, subject-level, pooled across views — also a user call, not
  a default. See `skills/preprocess-new-dataset/README.md` for why these are asked
  rather than assumed for every new dataset.

## Scripts

| Script | Env | Purpose |
|---|---|---|
| `convert_hantman_sleap.py` | `pose` | Full pipeline; writes `_raw/hantman/{labeled-data,CollectedData.csv,CollectedData_test.csv,project.yaml,videos/}` |

## Usage

```bash
conda run -n pose python scripts/preprocessing/hantman-sleap/convert_hantman_sleap.py

# different train/test split seed
conda run -n pose python scripts/preprocessing/hantman-sleap/convert_hantman_sleap.py --seed 1
```

## Stage 2 (not yet done — for when it's asked for)

Add `digit2_left/right`, `digit4_left/right` to `configs/keypoints.yaml`/`model.yaml`,
then follow the generic template in
[`skills/preprocess-new-dataset/README.md`](../../../skills/preprocess-new-dataset/README.md#documenting-a-stage-1-only-dataset).
