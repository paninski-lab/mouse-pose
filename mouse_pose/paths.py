from pathlib import Path

import yaml


def repo_root() -> Path:
    """Walk up from this file to find the repo root (directory containing paths.yaml)."""
    for p in Path(__file__).parents:
        if (p / "paths.yaml").exists():
            return p
    raise FileNotFoundError(
        "paths.yaml not found. Create it at the repo root — see README for the required format."
    )


def load_paths() -> dict:
    """Load paths.yaml and return as a plain dict."""
    return yaml.safe_load((repo_root() / "paths.yaml").read_text())
