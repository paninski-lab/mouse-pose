# ibl-face preprocessing

Generates pseudo-labels for face keypoints (pupil, nose, tongue) by running the
[iblvideo](https://github.com/int-brain-lab/iblvideo) Lightning Pose pipeline on
sessions from `_raw/ibl-paw`, then merges them with the existing paw labels.

## Why per-session videos

iblvideo's ROI detection computes a single average crop window over an entire video.
Mixing frames from many sessions would produce a meaningless crop, so a separate
short video is built for each session before running the pipeline.

## Pipeline steps

For each session in `_raw/ibl-paw/`:

1. **Build session video** — for every labeled frame, load 2 context frames before
   and 2 after (5-frame chunk) from `_raw/ibl-paw/labeled-data/<session>/`. Frames
   are upscaled 4× (320×256 → 1280×1024) to match iblvideo's LEFT_VIDEO spec.
   Written to `_raw/ibl-face/session_videos/{split}/{session}/videos/_iblrig_leftCamera.raw.mp4`.

2. **Run iblvideo LP pipeline** (`lightning_pose()`) — runs ROI detection then
   specialized networks: eye (×5 ensemble), nose_tip, tongue (×1), paws (×5 ensemble).
   Output parquet: `{session}/alf/_ibl_leftCamera.lightningPose.pqt`.

3. **Extract labeled-frame predictions** — rows `[2::5]` of the parquet correspond
   to the center (labeled) frame of each chunk. x/y coordinates are divided by 4 to
   return to 320×256 space.

4. **Apply threshold** — nose_tip and tongue likelihoods ≥ 0.9 are kept; below that
   are set to NaN. Eye likelihoods are ignored (EKS output is unreliable in the
   version used here); all pupil predictions are always kept.

5. **Compute `pupil_center_r`** — median x/y of the 4 pupil keypoints
   (`pupil_top_r`, `pupil_right_r`, `pupil_bottom_r`, `pupil_left_r`).

6. **Merge** — face pseudo-labels are combined with original paw labels (`paw_l`,
   `paw_r`) from `_raw/ibl-paw`.

Output keypoints: `paw_l`, `paw_r`, `pupil_center_r`, `nose_tip`, `tongue_end_r`, `tongue_end_l`.

## Scripts

| Script | Env | Purpose |
|---|---|---|
| `create_ibl_face_dataset.py` | `iblvideo2` | Run full pipeline; writes CSVs + copies images |
| `plot_ibl_face_check.py` | `iblvideo2` | Render annotated check images after pipeline |

## Usage

```bash
# Full run
conda run -n iblvideo2 python scripts/preprocessing/ibl-face/create_ibl_face_dataset.py

# Re-run without rebuilding videos (e.g. after fixing a config)
conda run -n iblvideo2 python scripts/preprocessing/ibl-face/create_ibl_face_dataset.py --skip_video

# Skip both video and inference (re-extract/merge from existing parquets)
conda run -n iblvideo2 python scripts/preprocessing/ibl-face/create_ibl_face_dataset.py --skip_video --skip_pipeline

# Render check images
conda run -n iblvideo2 python scripts/preprocessing/ibl-face/plot_ibl_face_check.py
```

## After the pipeline

```bash
conda run -n pose python scripts/convert_dataset.py --dataset ibl-face
python scripts/build_dataset.py --tag <tag>
```

## Known issues

- **EKS likelihoods for eye keypoints** are always ~0.002 due to an older EKS version
  that doesn't output valid likelihoods for ensemble models. Predictions are still
  geometrically correct; the threshold is simply not applied to them.
- **paws config patch**: the iblvideo paws model configs shipped with
  `keypoint_names: null`, which causes LP 2.2.0's `PredictionHandler` to crash.
  All 5 configs at `~/.../networks_v2.1/paw2-mic-2021-12-09_{0-4}/config.yaml`
  must have `keypoint_names: [paw_l, paw_r]`.
