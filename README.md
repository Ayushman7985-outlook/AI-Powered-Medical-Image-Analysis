# AI-Powered Medical Image Analysis System

Educational chest X-ray classifier for **NORMAL** and **PNEUMONIA** using Python, OpenCV, TensorFlow/Keras, scikit-learn, Matplotlib, and Streamlit. It performs real model training, held-out test evaluation, image prediction, and visualization; it is not a static demo.

> **Safety:** This educational prototype is not a medical device or medical advice. Its scores are not clinical certainty. Public-dataset results do not demonstrate clinical reliability. Decisions must be made by qualified healthcare professionals.

## Purpose and objectives

Learn a practical computer-vision pipeline: validate a public image dataset, decode and preprocess images, train one small CNN, save it, evaluate only on the provided test split, inspect plots/metrics, and run inference through a CLI or local web UI. Medical imaging supports clinicians by making internal structures visible; AI can assist image triage or pattern analysis, but errors, dataset bias, shortcuts, and distribution shift make human clinical judgment essential.

**Industry relevance:** demonstrates reproducible ML code, separation of training/validation/test data, sensitivity-aware evaluation, inference, UI integration, testing, and responsible communication. It is a learning portfolio project, not a deployment-ready system.

## Features and chosen approach

- Streaming image batches; shared RGB resize/normalization code for training and prediction.
- Custom 3-block CNN with binary sigmoid output, Adam, binary cross-entropy, early stopping, best-checkpoint saving, and learning-rate reduction.
- Test metrics (accuracy, precision, recall, F1, support, classification report, confusion matrix); PNEUMONIA is positive.
- Training curves, confusion matrix, actual test image comparisons, JSON artifacts, CLI, and Streamlit upload/inference.
- Optional quick subset for plumbing smoke tests; its scores are not meaningful model evaluation.

| Option                           | Libraries / data                                                                                     | Complexity and hardware                                                  | Typical output                                                         |
| -------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| A. Basic CNN (implemented)       | TensorFlow/Keras; labeled class folders                                                              | Beginner; CPU can run but training may be slow; modest image/batch sizes | Learned small CNN, scores and plots                                    |
| B. Lightweight transfer learning | Keras EfficientNet/MobileNet; same labels plus compatible pretrained weights                         | Intermediate; larger model and external weight/license/source review     | Features initialized from a pretrained model and fine-tuned classifier |
| C. Advanced medical pipeline     | Specialized imaging tools, explainability, segmentation, deployment controls and clinical governance | Advanced; substantial compute, data, validation and regulatory effort    | Broader image-analysis workflow, not merely binary classification      |

Option A is selected to keep the learning path transparent, avoid weight downloads and licensing dependencies, and show every CNN stage. This model is not guaranteed to perform better than transfer learning.

## Architecture

```text
Input X-ray → decode/validate → resize + normalize → CNN feature extraction
           → sigmoid classifier → NORMAL / PNEUMONIA + model score
           → evaluation artifacts / CLI / Streamlit
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module responsibilities and failure points; [docs/PROJECT_EXPLANATION.md](docs/PROJECT_EXPLANATION.md) for beginner-level explanations.

## Repository structure

```text
.
├── app.py                    # Streamlit interface (does not train on startup)
├── main.py                   # train / evaluate / predict CLI
├── src/                      # configuration, data, preprocessing, model, workflows
├── tests/                    # controlled unit tests; no full dataset required
├── data/README.md            # dataset source and expected tree
├── notebooks/README.md       # optional exploration guidance
├── models/                   # locally generated Keras model (ignored by Git)
├── outputs/                  # locally generated charts and JSON (ignored by Git)
├── images/                   # manually captured project evidence placeholders
└── docs/                      # explanations, architecture, troubleshooting, interviews
```

## Dataset

Use the public Mendeley Data dataset **Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification**, Kermany, Zhang, and Goldbaum, version 2, DOI [10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2), landing page [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2). It lists CC BY 4.0 and offers a 1.15 GB ChestXRay2017 archive. Review the current dataset page, attribution, license, and usage terms before obtaining/reusing it. The code does **not** download data automatically. This dataset is a research/educational source, not representative of all clinical populations.

Extract the archive into `data/chest_xray/` so the resulting tree is exactly `train/`, `val/`, and `test/`, each with `NORMAL/` and `PNEUMONIA/`. Never move test images into training/validation or silently tune against the test split. Directory checks and unreadable-image reports are implemented in `src/data_loader.py`. Alternate root: `--dataset PATH`.

The exact optional PowerShell download and extraction commands are in [data/README.md](data/README.md); they are provided for the user to run deliberately and are not invoked by the application.

## Installation

Use Python **3.11 or 3.12** for the pinned TensorFlow range in `requirements.txt`. The available system/default interpreters include Python 3.13 and 3.14, which are outside this project's TensorFlow range; explicitly select Python 3.12. A virtual environment isolates project packages from global Python.

**Windows PowerShell** (from repository root):

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

If script activation is restricted, use `.venv312\Scripts\python.exe -m pip install -r requirements-dev.txt` and invoke that interpreter directly; do not weaken machine policy just for this project. A previous `.venv` in this workspace uses Python 3.14; keep it separate and use the fresh `.venv312` for this project.

**macOS/Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

`requirements.txt` is runtime dependencies; `requirements-dev.txt` adds pytest. No pretrained weights are downloaded.

## Run the workflow

Run commands from the repository root. After extracting the dataset:

```powershell
python main.py train
python main.py evaluate
python main.py predict --image "data/chest_xray/test/PNEUMONIA/example.jpeg" --save-result
streamlit run app.py
```

On macOS/Linux, the same commands apply (use `python3` if needed). To run a small pipeline smoke test: `python main.py train --smoke-test`; this uses at most eight examples per class in each split and one epoch by default. It is only a plumbing check. For other dataset/model/output paths use `--dataset`, `--model`, and `--output-dir`; training also accepts `--epochs` and `--batch-size`. CLI help: `python main.py --help` and `python main.py train --help`.

Training uses train and validation only. Evaluation reads the separate test directory and writes **actual computed** metrics. Inference requires a saved model. The Streamlit app never trains automatically and reports clearly when no model exists.

## Outputs

Generated locally under `outputs/` (ignored by Git): `training_history.json`, `training_accuracy.png`, `training_loss.png`, `metrics.json`, `classification_report.json`, `confusion_matrix.png`, `prediction_examples.png`, and optional `prediction_result.json`. Model: `models/chest_xray_cnn.keras` (ignored). These artifacts appear only after the corresponding real operation succeeds. No results or metrics are bundled or fabricated.

Accuracy is the overall correct fraction; precision measures the share of predicted pneumonia cases that are actual pneumonia; recall measures the share of actual pneumonia cases detected; F1 balances precision and recall. A false negative is missed pneumonia and can be consequential; false positives can cause anxiety and unnecessary follow-up. Neither a high metric nor one test split establishes clinical safety. See [docs/PROJECT_EXPLANATION.md](docs/PROJECT_EXPLANATION.md).

## Tests

```powershell
pytest -q
```

Tests use temporary images and controlled labels to cover preprocessing, data validation, metrics, CLI dispatch, CNN forward inference and save/load prediction, plus Streamlit rendering without a trained model. A tiny end-to-end fixture checks training/evaluation artifact plumbing only; its results are not model-performance evidence. Tests do not require the public dataset. Install the development requirements in Python 3.11/3.12 to run the framework integration checks; evaluation of the public test split requires the public dataset.

## Demo simulation and evidence

Follow the 12-stage, explicitly non-clinical simulation in [docs/PROJECT_EXPLANATION.md](docs/PROJECT_EXPLANATION.md). Capture real evidence only after running each stage. Planned image filenames are listed in `images/README.md`; no screenshots, trained weights, dataset, or fabricated charts are included.

## GitHub publication

Suggested repository name: `AI-Powered-Medical-Image-Analysis`.
Suggested description: “Educational deep-learning project for chest X-ray classification using Python, TensorFlow/Keras, and Streamlit, with model evaluation, visualization, and image-based predictions.”
Topics: `medical-image-analysis`, `computer-vision`, `deep-learning`, `tensorflow`, `python`, `streamlit`, `machine-learning`, `chest-xray`.

Commands are templates—review status and remote first; do not overwrite an existing remote or force-push:

```bash
git status --short
git init                         # only if this is not already a Git repository
git add README.md .gitignore requirements.txt requirements-dev.txt src tests docs app.py main.py data/README.md notebooks/README.md images/README.md models/.gitkeep outputs/.gitkeep
git diff --cached --stat
git commit -m "Build educational chest X-ray analysis pipeline"
git branch --show-current
git remote -v
git remote add origin https://github.com/<YOUR-USERNAME>/AI-Powered-Medical-Image-Analysis.git
git push -u origin <CURRENT-BRANCH>
```

If a remote already exists, inspect it and use the matching remote name; don't replace it. Verify README rendering, attribution, ignored dataset/model files, and add genuine demo screenshots after capture. Suggested incremental commits: initial project structure; dataset validation/preprocessing; CNN and training; evaluation/visualization; inference/UI; tests/troubleshooting; final documentation. These are recommendations, not existing commits.

## Limitations and learning outcomes

This educational baseline may overfit, exploit spurious cues, or fail on different scanners, populations, protocols, or prevalence. Dataset split composition and provenance may introduce bias/leakage. No calibration, external validation, prospective study, clinical workflow integration, security/privacy review, or regulatory work is implemented. Do not submit identifiable patient images to this local demo or publish dataset images/models without checking rights. For any real clinical use, substantial independent clinical, technical, ethical, legal, and regulatory work is required.

Learn to organize an ML project, inspect data, use a CNN, reason about held-out evaluation, generate evidence, debug failures, and explain safety limits. Interview answers: [docs/INTERVIEW_QUESTIONS.md](docs/INTERVIEW_QUESTIONS.md). Common fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Suggested seven-day proof build

1. Environment/repository setup — capture setup and tree; commit initial structure.
2. Obtain and validate data — capture dataset tree and validation; commit loader.
3. Preprocessing — capture a real preprocessing output; commit preprocessing/tests.
4. CNN/training pipeline — capture model summary and training start; commit model/training.
5. Training/evaluation — capture actual logs, test metrics, matrix; commit evaluation/charts.
6. Prediction/UI — capture actual uploaded image, result and app; commit inference/app.
7. Documentation/publication — capture final GitHub page; commit docs/tests/demo guide.

This is a suggested schedule, not a claim that these steps or evidence already exist. More detail is in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
