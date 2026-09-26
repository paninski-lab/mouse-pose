# cazettes-side dataset changelog

Side-view head-fixed mouse recordings (scorer: Behnaz), already in standard DLC
layout — no custom conversion script was needed, so this folder exists only to
track keypoint-level changes to the source labels over time. See
[`configs/datasets/cazettes-side.yaml`](../../../configs/datasets/cazettes-side.yaml)
for the current keypoint mapping.

## Changelog

### 2026-09-26 (MW) (version 0)
- First versioned snapshot of cazettes-side's label CSVs. This predates the
  versioning scheme, so v0 is the best-available current state rather than a
  pristine collaborator copy — it already includes the undocumented
  pre-versioning edits logged above (2026-08-07 `pupilCenter` keypoint added +
  manual `rightPaw`/`leftPaw` adjustments; 2026-09-17 empty `ear_*` keypoints
  added).

### 2026-09-22 (MW)
- Added `mouse_pose/videos.py` (`make_video_snippet` + motion-energy helpers) and the
  general-purpose `scripts/preprocessing/extract_clips.py` wrapper (ported from an
  external script) — picks the highest-motion window in a video and re-encodes it to
  h264/yuv420p/mp4, regardless of source codec/container. Not cazettes-side-specific;
  lives at the top of `scripts/preprocessing/` for reuse across datasets.
- Ran it on the two raw recordings in `_raw/_dlc/cazettes-side/videos-avi/`
  (`FC075_HeadFixFlippingSTIM_Box02_200123a_Body0.avi`,
  `FC076_HeadFixFlippingSTIM_Box02_200225a_Body0.avi`) to produce short high-motion
  clips in `_raw/_dlc/cazettes-side/videos_test/` for review as labeling candidates.
  `FC076` has visible camera movement during setup, so `--skip-start 60` was used
  (applied to both videos) to exclude the first 60s from the motion search; final
  run used `--clip-length 10`. `--skip-start 60` still wasn't enough for `FC076`
  (camera movement continued past that point), so it was re-run alone with
  `--skip-start 120`, landing its clip at 1712.1s. These are raw-video review
  clips, not label changes — `CollectedData.csv`/`CollectedData_test.csv` are
  untouched.

### 2026-09-17 (MW)
- Added `ear_top`, `ear_tip`, `ear_bottom`, and `ear_base` as new keypoints
  (all rows empty — ears are never visible in this side view) to
  `CollectedData.csv` + `CollectedData_test.csv`, `project.yaml`, and the
  `keypoints` mapping in
  [`configs/datasets/cazettes-side.yaml`](../../../configs/datasets/cazettes-side.yaml)
  (`ear_*: ear_*_{side}`). All four canonical names already existed in
  `configs/keypoints.yaml`/`model.yaml`, so no vocab changes were needed.

### 2026-08-07 (MW)
- Added the `pupilCenter` keypoint and manually labeled it in the Lightning Pose
  app — all 830 rows across `CollectedData.csv` + `CollectedData_test.csv` are
  labeled (no NaNs).
- Made small adjustments to the `rightPaw` and `leftPaw` labels.
- Removed the stale per-session `CollectedData_Behnaz.csv` files under
  `_raw/cazettes-side/labeled-data/<session>/` (2026-09-17) — they predated this
  change and had no `pupilCenter` column, out of sync with the top-level
  `CollectedData.csv`/`CollectedData_test.csv`.
