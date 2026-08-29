"""Single source of truth for paths, hyperparameters and class labels.

Everything else — the Streamlit app, the training script, the notebook and the
tests — imports from here, so the image size and the label list can never drift
apart between training and inference again.
"""

import json
import re
from pathlib import Path

# Repo root, resolved from this file so nothing depends on the current directory.
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "PlantVillage"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "plant_disease_model_15.h5"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

KAGGLE_DATASET = "emmarex/plantdisease"

# Preprocessing / training hyperparameters. The model was trained at 128x128,
# so IMG_SIZE is a property of the saved weights, not a free knob.
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3
EPOCHS = 10
SEED = 42

# Below this top-1 probability the app warns that the image may not be a
# supported leaf, rather than presenting a confident-looking diagnosis.
LOW_CONFIDENCE_THRESHOLD = 0.60


def load_class_names():
    """Class labels in the exact index order the model outputs.

    Read from models/class_names.json, which is written by train.py from
    ``train_gen.class_indices`` — the .h5 file itself stores no label mapping.
    """
    with open(CLASS_NAMES_PATH) as f:
        return json.load(f)


CLASS_NAMES = load_class_names()


def pretty_label(raw):
    """Turn 'Tomato__Tomato_YellowLeaf__Curl_Virus' into 'Tomato — Yellow Leaf Curl Virus'.

    PlantVillage folder names use inconsistent separators (one, two, or three
    underscores) and repeat the crop name in some classes. Display only — the
    raw string stays the identifier everywhere else.
    """
    # 'YellowLeaf' -> 'Yellow Leaf': some class names run words together.
    split = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw)
    parts = [p for p in split.replace("_", " ").split() if p]
    if not parts:
        return raw

    crop = parts[0]
    rest = parts[1:]
    # 'Tomato_Tomato_mosaic_virus' and 'Pepper_bell_healthy' repeat the crop.
    while rest and rest[0].lower() == crop.lower():
        rest = rest[1:]
    if crop.lower() == "pepper" and rest and rest[0].lower() == "bell":
        crop, rest = "Bell Pepper", rest[1:]

    condition = " ".join(rest).strip() or "Unknown"
    condition = condition[0].upper() + condition[1:]
    return f"{crop} — {condition}"
