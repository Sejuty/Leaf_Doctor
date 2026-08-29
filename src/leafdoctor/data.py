"""Dataset acquisition.

Replaces the Colab cell that uploaded kaggle.json into the notebook — the cell
whose stored output leaked an API key into public git history. Credentials are
read by the kaggle CLI from ~/.kaggle/kaggle.json and never touch this repo.
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

from .config import DATA_DIR, DATASET_DIR, KAGGLE_DATASET

CREDENTIALS_HELP = """\
Kaggle credentials not found.

  1. Sign in at kaggle.com -> Settings -> API -> "Create New Token"
  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
  3. chmod 600 ~/.kaggle/kaggle.json

Never copy kaggle.json into this repository.
"""


def _credentials_available():
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def ensure_dataset(force=False):
    """Download and unzip PlantVillage into data/PlantVillage, returning its path.

    No-ops if the dataset is already present (it is ~658 MB / 20,638 images).
    """
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()) and not force:
        return DATASET_DIR

    if not _credentials_available():
        raise RuntimeError(CREDENTIALS_HELP)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / "plantdisease.zip"

    if not archive.exists() or force:
        print(f"Downloading {KAGGLE_DATASET} (~658 MB)...", file=sys.stderr)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(DATA_DIR)],
            check=True,
        )

    print(f"Extracting to {DATA_DIR}...", file=sys.stderr)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA_DIR)

    if not DATASET_DIR.exists():
        raise RuntimeError(
            f"Expected {DATASET_DIR} after extraction. The dataset layout may have changed."
        )

    archive.unlink()  # reclaim 658 MB; the extracted images are what we need
    return DATASET_DIR


if __name__ == "__main__":
    print(ensure_dataset())
