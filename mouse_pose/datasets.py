"""
Single source of truth for which datasets are registered in the combined corpus.

Used by both scripts/build_dataset.py (as the default --datasets set) and
mouse_pose/train.py (as the set evaluated against after each training run) --
previously two separately-maintained copies of this list that had to be kept
in sync by hand. This does NOT derive from configs/datasets/*.yaml; adding a
dataset here is still a manual step (see skills/preprocess-new-dataset/README.md,
stage 2) since not every configs/datasets/<name>.yaml is necessarily ready to
be part of default runs yet.
"""

ALL_DATASETS = ["facemap", "ibl", "cheese-2d", "cazettes-side", "kondo", "kaufman"]
