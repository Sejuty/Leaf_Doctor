# Leaf Doctor 🌿: Plant Disease Detection App

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://leafdoctor.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A user-friendly web application built to help you identify diseases in your pepper, potato, and tomato plants. Upload an image of a leaf, and the deep learning model predicts the disease.

**[\>\> Visit the Live Application \<\<](https://leafdoctor.streamlit.app/)**

## About The Project

Leaf Doctor uses MobileNetV2 transfer learning to classify 15 plant conditions from an uploaded leaf image. The goal is a simple, accessible tool for gardeners and farmers to quickly diagnose potential issues with their plants.

Trained on the PlantVillage dataset, deployed for free on Streamlit Community Cloud.

## Features

* **Image Upload** — JPG, JPEG and PNG.
* **Real-Time Prediction** — a diagnosis in seconds.
* **Top-3 Results** — the three most likely conditions with confidence bars, not just one label.
* **Low-Confidence Warning** — flags images that probably aren't a supported leaf instead of presenting a confident-looking diagnosis.
* **Responsive UI** — works on desktop and mobile.

## Technologies Used

* **Frontend**: Streamlit
* **Model**: Python, TensorFlow, Keras, OpenCV
* **Deployment**: Streamlit Community Cloud

## Project Layout

```
app.py                     Streamlit UI
src/leafdoctor/
  config.py                paths, hyperparameters, class labels, label formatting
  data.py                  PlantVillage download via the Kaggle CLI
  inference.py             decode / preprocess / predict — shared by app, tests, notebook
  train.py                 training CLI
models/
  plant_disease_model_15.h5
  class_names.json         label order matching the model's output indices
tests/test_inference.py
LeafDoctor.ipynb           local training walkthrough
```

`config.py` is the single source of truth for the image size and label list, so training and inference can't drift apart.

## Running Locally

Requires Python 3.9+ (but not exactly 3.9.7, which Streamlit excludes).

```bash
git clone https://github.com/Sejuty/Leaf_Doctor.git
cd Leaf_Doctor

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

.venv/bin/streamlit run app.py
```

The app runs at `http://localhost:8501`. The trained model ships in `models/` — no download needed.

### Development

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest tests/ -v          # add -m "not slow" to skip model loading
.venv/bin/ruff check .
```

### Retraining

The dataset (~658 MB) is not in the repo. Put your Kaggle token at `~/.kaggle/kaggle.json` (kaggle.com → Settings → API → Create New Token), then:

```bash
.venv/bin/python -m leafdoctor.train --epochs 10
```

This downloads PlantVillage into a gitignored `data/`, trains, and writes the best epoch to `models/`. **Never copy `kaggle.json` into this repository** — the credentials are read from your home directory precisely so they can't be committed.

Training is slow on CPU (hours per epoch over 16,516 images). Use a GPU.

## Model Information

* **Architecture**: MobileNetV2, ImageNet weights, frozen convolutional base + `GlobalAveragePooling2D → Dropout(0.3) → Dense(15, softmax)`
* **Dataset**: [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) — 20,638 images across 15 classes
* **Input**: 128×128 RGB, pixels scaled to `[0, 1]`
* **Accuracy**: **90.51% validation accuracy** after 10 epochs, on an 80/20 split

## Known Limitations

* **Validation, not test.** The 90.51% figure comes from the validation split used for model selection — there is no held-out test set, so it likely overstates real-world accuracy.
* **Class imbalance.** PlantVillage is heavily skewed towards tomato classes, so a single overall accuracy number hides weaker per-class performance. Per-class precision/recall is not yet reported.
* **15 classes only.** Pepper, potato and tomato. Any other leaf — or any non-leaf photo — will still be mapped to one of these 15; the low-confidence warning is a heuristic, not a real rejection class.
* **Lab conditions.** PlantVillage images are single leaves on plain backgrounds. Accuracy on field photos with soil, hands or multiple leaves will be lower.
* **Not agronomic advice.** For guidance only; confirm with an expert before treating a crop.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
