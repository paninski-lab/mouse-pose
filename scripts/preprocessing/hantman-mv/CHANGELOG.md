# hantman-mv dataset changelog

See [`README.md`](README.md) for the conversion pipeline and design decisions.

## Changelog

### 2026-09-26 (MW) (version 0)
- First versioned snapshot of hantman-mv's label CSVs. This predates the
  versioning scheme, so v0 is the best-available current state rather than a
  pristine collaborator copy — it already includes the undocumented
  pre-versioning edits logged above (2026-09-15 new eye/nose keypoints +
  manual labels, derived-then-manually-corrected `wrist_new` column;
  2026-09-17 empty `ear_*` keypoints added). Neither of those additions is
  reproduced by `convert_hantman_mv.py` — see the note at the bottom of this
  changelog.

### 2026-09-22 (MW)
- Ran `scripts/preprocessing/extract_clips.py` (see
  [`cazettes-side/CHANGELOG.md`](../cazettes-side/CHANGELOG.md) for where that script came
  from) on the four videos in `_raw/_dlc/hantman-mv/videos_test/`
  (`KPC190_20260214_v033_front`, `KPC190_20260214_v033_side`,
  `KPC200_20260323_v053_front`, `KPC200_20260323_v053_side`), saving to
  `_raw/hantman-mv/videos_test/` as labeling-candidate review clips.
- These containers report `1 fps` (5998 frames -> ffprobe reads the duration as ~100
  minutes), but the true capture rate is 500 fps (~12s actual duration) — the 1 fps tag
  is wrong metadata, not a real frame rate. This prompted two new options on
  `extract_clips.py`/`make_video_snippet`: `--from-start` (take the first
  `--clip-length` seconds instead of searching for the highest-motion window) and
  `--fps` (override the frame rate used for all time math, and pass `-r <fps>` to
  ffmpeg as an input option so it regenerates timestamps at the true rate instead of
  trusting the container's declared one).
- First tried `--clip-length 1 --skip-start 0 --from-start --fps 500` as a quick check
  (500 fps / 500 frames / 1.0s output, confirmed correct), then `--clip-length 12` for
  the full ~12s clip, before settling on `--clip-length 8` — verified via ffprobe that
  each final output clip is 500 fps / 4000 frames / 8.0s duration, h264/yuv420p/mp4.
  Raw-video review only, not a label change — `CollectedData*.csv` files are
  untouched.

### 2026-09-17 (MW)
- Added `ear_top`, `ear_tip`, `ear_bottom`, and `ear_base` as new keypoints (all rows
  empty — ears are never visible in this view) to `project.yaml` + both
  `CollectedData*.csv` files, and mapped `ear_*: ear_*_{side}` in
  [`configs/datasets/hantman-mv.yaml`](../../../configs/datasets/hantman-mv.yaml). All
  four canonical names already existed in `configs/keypoints.yaml`/`model.yaml`.
- Added an exemption (`_HANTMAN_MV_LEFT_SUPPRESS_EXEMPT`) to `_post_process_hantman_mv`
  in `scripts/convert_dataset.py` so `ear_*_left` stays at the default `visible=1`
  (matching `ear_*_right`) instead of being swept into the blanket `_left → visible=0`
  rule described under "Stage 2" in `README.md`. Without it the ear keypoints would've
  ended up asymmetric (right actively suppressed, left excluded from loss entirely),
  unlike the symmetric treatment in `cazettes-side`/`facemap`. The pre-existing
  `eye_*_left`/digit/wrist `_left` columns are unaffected and still forced to
  `visible=0`.

### 2026-09-15 (MW)
- Added `eye_back`, `eye_top`, `eye_front`, `eye_bottom`, `nose_tip`, and `nose_bottom`
  as new keypoints to `project.yaml` + both `CollectedData*.csv` files, and manually
  labeled them in the LP app (no source ever provided values for them).
- Computed a new `wrist_new` column from three existing keypoints:
  `mid = nanmean(d2_base, d3_base)` (x/y independently), then
  `wrist_new = (mid + hand_middle) / 2` — `NaN` whenever either input is `NaN`.
  183/192 train rows and 24/30 test rows got a real value under this formula; some
  values were **manually corrected afterward**, so the CSVs no longer match the formula
  exactly row-for-row — treat it as how the column originated, not a reproducible
  derivation of its current values. Maps to `wrist_{side}` (reuses existing canonical
  `wrist_left`/`wrist_right` — no new vocab needed).

**Neither addition above is reproduced by `convert_hantman_mv.py`.** Both are manual
edits to the stage-1 output; the raw source (`_raw/_dlc/hantman-mv`) still only has the
original 17 finger/paw/pellet keypoints, and a from-scratch re-run would regenerate only
those and silently drop these additions.
