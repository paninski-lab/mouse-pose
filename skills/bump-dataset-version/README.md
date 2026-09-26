# Bumping a raw dataset's label version

Each dataset under `_raw/<dataset>/` gets its own version history, independent of
every other dataset. A version is created whenever the label data changes —
new keypoints, pseudo-labels, or manual label corrections. It is **not**
created for changes that don't touch `CollectedData.csv` / `CollectedData_test.csv`
(e.g. a video-review-only changelog entry) — those still get a changelog entry,
just without a `(version N)` tag.

## What a version bump does

Running `scripts/bump_version.py` for a dataset does three things atomically:

1. Copies the current `CollectedData.csv` and `CollectedData_test.csv` into
   `_raw/<dataset>/versions/` as `CollectedData_versionN.csv` and
   `CollectedData_test_versionN.csv`.
2. Writes `N` into `_raw/<dataset>/VERSION.txt`, so anyone looking at the
   dataset can tell at a glance which version the live CSVs currently are.
3. Inserts a `### {date} ({initials}) (version N)` entry at the top of
   `scripts/preprocessing/<dataset>/CHANGELOG.md`.

The live CSVs at the top of `_raw/<dataset>/` are always a mutable working
copy — they can be in a "mixed", in-progress state between bumps. The
`versions/` copies are the immutable record.

## Version numbering

- **Version 0** is the dataset as received — either the collaborator's raw
  labels, unedited, or (for a dataset that already existed before this
  versioning scheme did) the best-available snapshot at the time versioning
  started for it. In the latter case the changelog entry for v0 should say
  explicitly that it may already include undocumented pre-versioning edits,
  so it isn't mistaken for a genuinely pristine collaborator copy.
- **Version 1+** are our own edits on top of that.
- When onboarding a brand-new dataset, run the bump script once immediately
  after the raw data lands in `_raw/<dataset>/`, *before* making any edits —
  that run captures v0.

## Running it

```
python scripts/bump_version.py <dataset> --initials <XX> --message-file <path> [--dry-run]
```

- `<dataset>` must match a directory name under `raw_dir` (see `paths.yaml`)
  **and** have a corresponding `scripts/preprocessing/<dataset>/` directory —
  the script won't guess where to put a new dataset's changelog, so create
  that directory first if it doesn't exist yet.
- `--message-file` points at a small markdown file containing just the bullet
  list body for the changelog entry (no `###` header — the script writes
  that). Write it the same way as existing entries in that dataset's
  `CHANGELOG.md`.
- `--dry-run` prints what would happen (next version number, files touched,
  the changelog entry text) without writing anything.

The script refuses to bump if the live CSVs are byte-identical to the last
snapshot (nothing to version) — this guards against an accidental duplicate
bump when no actual label change was made.

## Deciding when to bump

Whether a given change is "version-worthy" is a judgment call for whoever is
making the edit — there's no automated check for this. As a rule of thumb:
anything that changes what's in `CollectedData.csv` / `CollectedData_test.csv`
(new keypoints, pseudo-labels, manual corrections) gets a version; changes
that don't touch those files (raw video review, notes, etc.) get a plain
changelog entry with no version tag.
