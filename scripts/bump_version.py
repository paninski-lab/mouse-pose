#!/usr/bin/env python
"""Bump the label-data version for a raw dataset.

Snapshots CollectedData.csv / CollectedData_test.csv into versions/, ticks
VERSION.txt, and inserts a changelog entry — all in one step, so the three
never drift out of sync. See ../skills/bump-dataset-version/README.md for the
versioning policy (when to bump, what version 0 means, etc.).
"""
import argparse
import filecmp
import shutil
import sys
from datetime import date
from pathlib import Path

from mouse_pose.paths import load_paths, repo_root

CSV_NAMES = ["CollectedData.csv", "CollectedData_test.csv"]


def snapshot_path(versions_dir: Path, name: str, version: int) -> Path:
    stem, suffix = name.rsplit(".", 1)
    return versions_dir / f"{stem}_version{version}.{suffix}"


def get_current_version(version_file: Path) -> int | None:
    if not version_file.exists():
        return None
    return int(version_file.read_text().strip())


def snapshot_matches_live(raw_dir: Path, versions_dir: Path, version: int) -> bool:
    for name in CSV_NAMES:
        snapshot = snapshot_path(versions_dir, name, version)
        if not snapshot.exists() or not filecmp.cmp(raw_dir / name, snapshot, shallow=False):
            return False
    return True


def insert_changelog_entry(changelog_file: Path, dataset: str, entry: str) -> None:
    marker = "## Changelog\n"
    if changelog_file.exists():
        old_text = changelog_file.read_text()
        if marker not in old_text:
            sys.exit(f"{changelog_file} has no '## Changelog' section — fix the header manually before bumping.")
        head, _, tail = old_text.partition(marker)
        tail = tail.lstrip("\n")
        new_text = f"{head}{marker}\n{entry}\n{tail}" if tail else f"{head}{marker}\n{entry}"
    else:
        new_text = f"# {dataset} dataset changelog\n\n{marker}\n{entry}\n"
    changelog_file.write_text(new_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset name, e.g. 'kondo' (must match a directory under raw_dir)")
    parser.add_argument("--initials", required=True, help="Initials for the changelog entry, e.g. 'MW'")
    parser.add_argument(
        "--message-file",
        required=True,
        type=Path,
        help="Path to a markdown file with the changelog entry body (a '- ...' bullet list)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without writing anything")
    args = parser.parse_args()

    paths = load_paths()
    raw_dir = Path(paths["raw_dir"]) / args.dataset
    if not raw_dir.is_dir():
        sys.exit(f"No such raw dataset directory: {raw_dir}")

    for name in CSV_NAMES:
        if not (raw_dir / name).exists():
            sys.exit(f"Missing {name} in {raw_dir}")

    version_file = raw_dir / "VERSION.txt"
    versions_dir = raw_dir / "versions"
    current_version = get_current_version(version_file)

    if current_version is not None:
        if not versions_dir.is_dir():
            sys.exit(
                f"{version_file} says {current_version} but {versions_dir} is missing — "
                "inconsistent state, fix manually before bumping."
            )
        if snapshot_matches_live(raw_dir, versions_dir, current_version):
            sys.exit(
                f"{args.dataset}: current CSVs are identical to version {current_version} — "
                "refusing to bump a no-op. Edit the CSVs first if a change is actually intended."
            )
        next_version = current_version + 1
    else:
        next_version = 0  # v0 = the dataset as received (collaborator raw labels, or best-available snapshot)

    changelog_dir = repo_root() / "scripts" / "preprocessing" / args.dataset
    if not changelog_dir.is_dir():
        sys.exit(
            f"No preprocessing directory for '{args.dataset}' at {changelog_dir}. "
            "Create it first — this script won't guess where a new dataset's changelog belongs."
        )
    changelog_file = changelog_dir / "CHANGELOG.md"

    message = args.message_file.read_text().rstrip("\n")
    if not message:
        sys.exit(f"{args.message_file} is empty")

    today = date.today().isoformat()
    entry = f"### {today} ({args.initials}) (version {next_version})\n{message}\n"

    print(f"Dataset:  {args.dataset}")
    print(f"Version:  {current_version if current_version is not None else '(none)'} -> {next_version}")
    print(f"Snapshot: {[str(raw_dir / n) for n in CSV_NAMES]} -> {versions_dir}/")
    print(f"Marker:   {version_file}")
    print(f"Log:      {changelog_file}")
    print("---")
    print(entry)
    print("---")

    if args.dry_run:
        print("(dry run — nothing written)")
        return

    versions_dir.mkdir(exist_ok=True)
    for name in CSV_NAMES:
        dst = snapshot_path(versions_dir, name, next_version)
        if dst.exists():
            sys.exit(f"{dst} already exists — refusing to overwrite. Inconsistent state, fix manually.")
        shutil.copy2(raw_dir / name, dst)

    version_file.write_text(f"{next_version}\n")
    insert_changelog_entry(changelog_file, args.dataset, entry)

    print(f"Bumped {args.dataset} to version {next_version}.")


if __name__ == "__main__":
    main()
