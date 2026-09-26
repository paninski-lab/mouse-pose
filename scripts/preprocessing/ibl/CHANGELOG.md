# ibl dataset changelog

See [`README.md`](README.md) for the pseudo-labeling pipeline.

## Changelog

### 2026-09-26 (MW) (version 0)
- First versioned snapshot of ibl's label CSVs. This predates the versioning
  scheme, so v0 is the best-available current state rather than a pristine
  collaborator copy — it already includes the undocumented pre-versioning
  edits logged above (2026-06-18 `nose`/`pupil`/`tongue` keypoints added ahead
  of the iblvideo pseudo-labeling pipeline) plus a full manual review/
  correction of all ~10k frames in the Lightning Pose app (July 2026, not
  previously logged here) — these are now full human annotations, not
  pseudo-labels.

### 2026-06-18 (MW)
- Manually added `nose`, `pupil`, and `tongue` as new keypoints to the `ibl-paw` schema
  ahead of the pseudo-labeling pipeline, which fills them in via iblvideo.
