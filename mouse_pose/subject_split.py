"""
Subject-level train/test splitting, shared by preprocessing scripts that need to keep
a subject (animal) entirely on one side of the split (e.g. because it appears in
multiple views/sessions and no group should leak across train/test).

Previously duplicated near-identically in scripts/preprocessing/hantman-sleap/ and
scripts/preprocessing/hantman-mv/ -- see skills/preprocess-new-dataset/README.md's
train/test split question for when this applies vs. the default session-level split.
"""

import numpy as np


def subject_of(session: str) -> str:
    """A session name's subject: its first '_'-delimited token, uppercased so casing
    slips in source filenames (e.g. "jcr130" vs "JCR130") don't split one animal
    across train/test."""
    return session.split("_")[0].upper()


def subject_split(
    counts: dict[str, int], seed: int, test_fraction: float
) -> tuple[set[str], set[str]]:
    """Greedily assign whole subjects to the test set (in a seeded shuffle order)
    until test_fraction of the total count is reached, then the rest go to train.

    `counts` maps each subject to its frame/row count. This only ever *overshoots*
    test_fraction (it stops as soon as the running total meets it) -- with few
    subjects of uneven size that can overshoot a lot, so if you were given a target
    *range* (e.g. "10-15%") rather than an exact number, pass the range's midpoint as
    test_fraction, not its upper edge, to leave room for the overshoot.
    """
    subjects = list(counts.keys())
    np.random.default_rng(seed).shuffle(subjects)

    total = sum(counts.values())
    target = total * test_fraction

    test_subjects: set[str] = set()
    test_count = 0
    for s in subjects:
        if test_count >= target:
            break
        test_subjects.add(s)
        test_count += counts[s]

    train_subjects = set(subjects) - test_subjects
    return train_subjects, test_subjects
