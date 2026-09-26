# facemap dataset changelog

Single-camera orofacial mouse recordings from the public
[Facemap dataset](https://doi.org/10.25378/janelia.23712957) (Syeda et al. 2024),
already in standard DLC layout — no custom conversion script was needed, so this
folder exists only to track keypoint-level changes to the source labels over
time. See
[`configs/datasets/facemap.yaml`](../../../configs/datasets/facemap.yaml)
for the current keypoint mapping.

## Changelog

### 2026-09-26 (MW) (version 0)
- First versioned snapshot of facemap's label CSVs. This predates the
  versioning scheme, so v0 is the best-available current state rather than a
  pristine collaborator copy — it already includes the undocumented
  pre-versioning edits logged above (2026-09-17 empty `ear_*` keypoints added;
  2026-09-22 manual `pupil` labels added).

### 2026-09-22 (MW)
- Added manual labels for the `pupil` keypoint to `CollectedData.csv` and `CollectedData_test.csv`.
  Added `pupil: pupil_center_{side}` to the `keypoints` mapping in
  [`configs/datasets/facemap.yaml`](../../../configs/datasets/facemap.yaml).
  `pupil_center_left`/`pupil_center_right` already existed in `configs/keypoints.yaml`/
  `model.yaml` (populated via `ibl`), so no vocab changes were needed. `convert_dataset.py` still
  needs to be re-run to pick up the new labels.

### 2026-09-17 (MW)
- Added `ear_top`, `ear_tip`, `ear_bottom`, and `ear_base` as new keypoints
  (all rows empty — ears are never visible in this view) to
  `CollectedData.csv` + `CollectedData_test.csv`, `project.yaml`, and the
  `keypoints` mapping in
  [`configs/datasets/facemap.yaml`](../../../configs/datasets/facemap.yaml)
  (`ear_*: ear_*_{side}`). All four canonical names already existed in
  `configs/keypoints.yaml`/`model.yaml`, so no vocab changes were needed.
