# kondo dataset changelog

Already in standard DLC layout — no custom conversion script was needed, so this
folder exists only to track changes to the source data over time. See
[`configs/datasets/kondo.yaml`](../../../configs/datasets/kondo.yaml) for the
current keypoint mapping.

## Changelog

### 2026-09-26 (MW)
- Manually adjusted the `earroot`/`earlateral`/`eartip` labels to be consistent with
  the ear landmark definitions used in `cheese-2d`.

### 2026-09-22 (MW)
- Ran `scripts/preprocessing/extract_clips.py` (see
  [`cazettes-side/CHANGELOG.md`](../cazettes-side/CHANGELOG.md) for where that script
  came from) on the four raw face-camera recordings in
  `_raw/_dlc/MouseViewsDLC/dlc-projects/mouse-forelimb-v1/videos/`
  (`sub-VG1-GC#125_ses-2024-04-19-task-day2`,
  `sub-VG1-GC#127_ses-2024-04-16-task-day1`,
  `sub-VG1-GC#129_ses-2024-03-07-task-day1`,
  `sub-VG1-GC#130_ses-2024-04-16-task-day1`), using the script's defaults
  (`--skip-start 60`, `--clip-length 15`), to produce short high-motion clips in
  `_raw/kondo/videos_test/` for review as labeling candidates. h264/yuv420p/mp4,
  15s each. This is raw-video review, not a label change —
  `CollectedData.csv`/`CollectedData_test.csv` are untouched.

### 2026-08-17 (MW)
- Manually adjusted the `rightpawcenter` and `leftpawcenter` labels to align them
  more closely with the IBL paw labels.

### 2026-08-14 (MW)
- Initial data (`CollectedData.csv`/`CollectedData_test.csv`, `labeled-data/`,
  `project.yaml`) extracted into `_raw/kondo/` from
  `_raw/_dlc/MouseViewsDLC/dlc-projects/mouse-face-v1`.
