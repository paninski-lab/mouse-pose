# Preprocessing a new dataset

This directory holds one subfolder per dataset that needed custom work *before*
`scripts/convert_dataset.py` could run on it — anything not already in the standard
DLC layout (`labeled-data/<session>/<frame>.png` + `CollectedData.csv` +
`CollectedData_test.csv`, all keypoints as plain `x`/`y` columns, NaN = unlabeled).
See `ibl-face/` (DLC source, but needs pseudo-label generation first),
`hantman-sleap/` (raw SLEAP `.slp` source), and `hantman-mv/` (already DLC-format, but
split across per-view CSVs that need merging into one single-view project) for three
different examples.

## Three separable stages

Turning a raw contributed dataset into something usable here is three distinct stages,
and **doing stage 1 does not commit you to stages 2 or 3**:

1. **Convert to LP format** (this directory's job) — raw source → standard DLC-layout
   `_raw/<name>/`. Output is a standalone, inspectable LP project. This is a complete,
   valid stopping point: sometimes a dataset just needs to be converted and looked at,
   with no decision yet about whether it belongs in the combined corpus.

   Writing a draft `configs/datasets/<name>.yaml` (the keypoint-mapping/exclude/session
   config) is fine to do as part of stage 1, unprompted — it's a single, self-contained
   file that's trivial to delete if the dataset never goes further. This is different
   from the rest of stage 2 below, which threads `<name>` through several shared files
   in ways that are much less clean to undo.
2. **Add to the corpus** — everything that touches shared, multi-dataset state: add new
   keypoints to `configs/keypoints.yaml` / `configs/model.yaml` if the dataset
   introduces any, register `<name>` in `ALL_DATASETS` (`mouse_pose/datasets.py`), then
   run `convert_dataset.py`. This is what actually merges the dataset's semantics into the
   shared vocabulary — it should not happen automatically just because stage 1
   happened. **Ask before starting stage 2**, even if stage 1 just finished in the same
   conversation. `hantman` is a live example of stage 1 done (including a draft
   `configs/datasets/hantman.yaml`), stage 2 deliberately not started.
3. **Rebuild the combined dataset** — re-run `scripts/build_dataset.py` for any merged
   tags that should include the new dataset.

This README covers stage 1. See the main `README.md`'s "Adding a new dataset" section
for stages 2 and 3.

## Questions to ask before converting (stage 1)

Most of stage 1 is mechanical once the shape of the source data is understood. The part
that repeatedly needed a human call — not something inferable from the data alone — is
the list below. **Ask about these up front, before writing any conversion code**, rather
than picking a default and mentioning it after the fact.

1. **New keypoints.** Does the source track anything not already in
   `configs/keypoints.yaml`? If so: what canonical name/section should it get, and —
   separately — is it even a keypoint that belongs in this vocabulary? (The hantman
   conversion tracked a `pellet`, the target object being reached for, not a mouse
   body part — excluded rather than added.) This is answered while drafting stage 1's
   `configs/datasets/<name>.yaml`, but the answer only becomes *real* (a shared
   `configs/keypoints.yaml`/`model.yaml` edit) in stage 2 — the draft config can name a
   canonical target that doesn't exist in the vocabulary yet without breaking anything.

2. **Laterality.** If a tracked point is one-sided in the source (e.g. a single
   `wrist` rather than `wrist_l`/`wrist_r`) but the canonical name is `_left`/`_right`,
   which side does it map to? This is usually *not* recoverable from a single frame's
   pixels — mirroring/camera-orientation conventions and per-subject handedness are
   both invisible without lab context. Don't guess from an image; ask. Also goes into
   the draft `configs/datasets/<name>.yaml`.

   If a dataset only ever assesses one side (e.g. every session lateralized to
   `right`), `convert_dataset.py`'s default per-split processing still marks the
   unassessed `_left` counterpart `visible=1` ("in dataset, unlabeled") rather than
   `visible=0` ("not part of this dataset") — training on that teaches the model to
   predict a suppressed heatmap for a side that was never captured at all. Add a
   `POST_PROCESS["<name>"]` function in `scripts/convert_dataset.py` to force those
   columns to `visible=0`; see `hantman-mv`/`kaufman` for this (every session the same
   side). If a dataset instead varies side *per session* (as `cheese-2d` once did,
   before all its keypoints were fully labeled), the post-process function needs to key
   off the session-to-side mapping in `configs/datasets/<name>.yaml` rather than
   applying one rule dataset-wide.

3. **Multi-view sources.** If the raw data has more than one camera view (or more
   generally, more than one natural sub-grouping), should each view become its own
   dataset entry (like `petersen-side` / `petersen-top`), or should they be merged
   into a single project (like `cheese-2d`, `hantman`, or `hantman-mv`)? This changes
   stage 1's raw directory layout directly (one `_raw/<name>/` or several) — decide
   before writing the conversion script, not after. If merging and the source is
   already DLC-format split across per-view CSVs with matching schemas (as in
   `hantman-mv`), merging is just a row concat — no remapping needed as long as
   session names across views don't collide. If the source also ships a `project.yaml`
   with a `view_names` list, clear it (`view_names: []`) in the output so the LP
   labeling app treats the merged result as single-view.

4. **Train/test split.** If the source doesn't already provide a split (most
   contributed datasets don't), you need to invent one. Ask, don't default silently:
   - For a **subject-level** split specifically, use `mouse_pose.subject_split`
     (`subject_of`, `subject_split`) rather than reimplementing it — it's shared by
     `hantman-sleap/` and `hantman-mv/` already; a third copy shouldn't exist.
   - **Fraction** — what proportion of frames/sessions/subjects should be held out?
   - **Grouping unit** — every dataset in this repo splits so that no group leaks
     across train/test, but *what the group is* varies: session-level is the default
     (one video's frames stay together), but a multi-subject, multi-view dataset may
     need subject-level grouping instead, pooled across views, so the same animal
     never appears in both splits from a different angle. Always check for naming
     inconsistencies (casing, typos) in whatever field the grouping key is derived
     from, since those silently create leaks otherwise.
   - **If given a range** (e.g. "10-15%") rather than an exact number: a greedy
     group-accumulation split (add whole subjects/sessions to the test set, in some
     order, until the target is reached) only ever *overshoots* its target, since it
     stops as soon as the running total meets it. With few groups of uneven size this
     can overshoot by a lot — one subject alone was 23% of all frames in `hantman-mv`'s
     17-subject split. Aim the algorithm's target at the *midpoint* of the requested
     range, not its upper edge, to leave room for the overshoot while still landing
     inside the range; if it still doesn't land in range, try a few other split seeds
     rather than accepting an out-of-range result.

5. **Whether to go past stage 1 at all, right now.** Once the LP-format conversion
   works, don't assume the next move is corpus integration. Ask.

## Stage 1 mechanics

1. Write a `scripts/preprocessing/<name>/` script that reads the raw source and
   produces `_raw/<name>/labeled-data/...` + `CollectedData.csv` +
   `CollectedData_test.csv` in standard DLC format.
   - **Copy only the images a CSV row actually references**, not whole
     `labeled-data/<session>/` directories — a session directory on disk commonly
     holds many more frames than are labeled (unlabeled context frames around each
     label; `hantman-mv`'s source had ~2050 images on disk against 222 labeled rows).
     A wholesale directory copy silently pulls in a lot of unused data.
   - **Always write a `project.yaml`** (`keypoint_names`, `schema_version: 1`,
     `view_names`) into the output `_raw/<name>/`, even if the source didn't have
     one (e.g. a raw SLEAP export) — this is what lets the dataset be opened directly
     in the Lightning Pose labeling app for a stage-1-only dataset. If the source
     shipped its own `project.yaml`, carry its `keypoint_names` forward rather than
     re-deriving them; set `view_names: []` if views were merged (see the multi-view
     question above). Also create an empty `videos/` dir, matching the layout other
     `_raw/` datasets use.
   - Watch for OS/transfer artifacts mixed into the source that aren't real data
     (e.g. Windows `*.jpgZone.Identifier` files) — don't copy them.
2. Spot-check: overlay a converted frame's keypoints on its image and confirm they
   land in the right place — this catches skeleton-ordering and coordinate-system bugs
   that nothing downstream will validate for you (`convert_dataset.py`'s own validation
   only checks names, not values, and only runs in stage 2 anyway).
3. Write `scripts/preprocessing/<name>/README.md` documenting the source format, why a
   custom script was needed, and any design decisions from the questions above,
   including a "Status: stage 1 only" note if stage 2 hasn't happened — it won't be
   obvious from the code alone that a dataset was deliberately left out of the corpus.
   - Keep keypoint/label-level history out of README.md — track it in a sibling
     `scripts/preprocessing/<name>/CHANGELOG.md` instead (dated, most-recent-first
     entries, same style as existing ones). README.md documents the pipeline as it
     currently stands; CHANGELOG.md is the append-only record of what changed and
     when. For a dataset already in standard DLC layout that needed no custom script
     (e.g. `facemap`, `cazettes-side`, `kondo`), CHANGELOG.md may be the only file in
     the folder — skip README.md entirely rather than writing one with nothing to say.

Stages 2 and 3 (corpus integration, rebuilding merged tags) are in the main
`README.md`'s "Adding a new dataset" and "Renaming or deprecating a dataset" sections —
only do those once asked.

## Documenting a stage-1-only dataset

Every `scripts/preprocessing/<name>/README.md` for a dataset that hasn't reached stage 2
yet should say so in one line near the top, rather than restating the stage
model — that explanation lives here, once:

> **Status: stage 1 only.** `_raw/<name>/` is a usable standalone LP project, not yet in
> the combined corpus. See [`scripts/preprocessing/README.md`](../README.md) for what
> stage 2 would involve; don't start it unless asked.

If stage 2 commands are worth spelling out for this specific dataset (e.g. which new
keypoints it would add), use this template rather than re-deriving it:

```bash
# add <name>'s new keypoints to configs/keypoints.yaml / configs/model.yaml (if any),
# add "<name>" to ALL_DATASETS in mouse_pose/datasets.py, then:
conda run -n pose python scripts/convert_dataset.py --dataset <name>
python scripts/build_dataset.py --tag <tag> --datasets <name> ...
```
