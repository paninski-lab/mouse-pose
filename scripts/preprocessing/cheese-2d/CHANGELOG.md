# cheese-2d dataset changelog

Single-view mouse orofacial recordings, already in standard DLC layout — no custom
conversion script was needed, so this folder exists only to track keypoint-level
changes to the source labels over time. See
[`configs/datasets/cheese-2d.yaml`](../../../configs/datasets/cheese-2d.yaml) for the
current keypoint mapping.

## Changelog

### 2026-09-26 (MW) (version 0)
- First versioned snapshot of cheese-2d's label CSVs. This predates the versioning
  scheme, so v0 is the best-available current state rather than a pristine
  collaborator copy — it already includes the undocumented pre-versioning edits
  logged above (the 2026-09-24 LA keypoint infill, 2026-09-25 `POST_PROCESS`
  override removal, and 2026-09-26 manual ear/eye/paw label check).

### 2026-09-26 (MW)
- Manually checked all ear, eye, and paw labels (a mix of manual labels and the
  pseudo-labels from the 2026-09-25 infill), fixed obvious mislabels, and infilled
  remaining missing labels. Adjusted every ear base label to be more consistent.

### 2026-09-25 (MW)
- Removed the `POST_PROCESS["cheese-2d"]` override (`_post_process_cheese2d`) from
  `scripts/convert_dataset.py`. It existed to force `visible=0` on facial keypoints
  that were structurally absent (wrong head orientation for that session) but still
  came through as `visible=1` (annotation gap) from the default per-split processing.
  Now that the (LA) infill below has filled in those gaps directly in the source
  labels, every keypoint that's actually visible in a frame is labeled, so the
  override's correction is no longer needed.

### 2026-09-24 (LA)
- Infilled missing keypoint labels across the source data — including but not limited
  to `wrist_left`/`wrist_right`, added here for the first time — using a model trained
  on `cheese-3d`. Added the `wrist_left`/`wrist_right` mapping entries to
  `configs/datasets/cheese-2d.yaml`.
