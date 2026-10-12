# Architecture and module responsibilities

## Data flow

```text
Input X-ray image
      |
      v
Image loading and validation (src/preprocessing.py, src/data_loader.py)
      |
      v
RGB conversion → resize to configured size → normalize to [0, 1]
      |
      v
CNN feature extraction (src/model.py)
      |
      v
Sigmoid classification layer (PNEUMONIA score; threshold 0.5)
      |
      v
NORMAL / PNEUMONIA + uncalibrated score
      |
      v
Metrics/plots/JSON (src/evaluate.py, src/visualization.py)
      |
      v
CLI (main.py) or Streamlit UI (app.py)
```

## Source modules

| Module                 | Responsibility                                                                                 | Common failure                                                      |
| ---------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `src/config.py`        | Canonical paths, input shape, batch/epoch defaults, class mapping                              | Wrong root or inconsistent custom settings                          |
| `src/data_loader.py`   | Split/class validation, unreadable-image warnings, deterministic subsets and streaming batches | Missing folders, empty class, unreadable file, memory/IO bottleneck |
| `src/preprocessing.py` | Decode, RGB conversion, resize, normalization, model batch shape                               | Corrupt/empty/unsupported input or invalid pixel range              |
| `src/model.py`         | Compact sigmoid CNN and binary compile configuration                                           | TensorFlow unavailable or incompatible                              |
| `src/train.py`         | Train/validation-only fit, callbacks, model/history/curve persistence                          | Missing split, insufficient memory, write permissions               |
| `src/evaluate.py`      | Test-only evaluation and metric/report/matrix/example outputs                                  | Missing model/test images, sklearn missing                          |
| `src/predict.py`       | Saved model load, shared preprocessing, class/score, optional JSON                             | Missing/invalid model or image                                      |
| `src/visualization.py` | Reusable actual-data charts; no placeholder performance plots                                  | Missing Matplotlib, malformed history/matrix                        |
| `src/utils.py`         | Directory and JSON helpers                                                                     | Permission or disk errors                                           |
| `main.py`              | `train`, `evaluate`, `predict` command dispatch/options                                        | Invalid CLI option or workflow error                                |
| `app.py`               | Upload UI, cached saved model, shared inference and available artifacts                        | Model not trained, unreadable upload, stale/invalid artifact        |

TensorFlow is imported inside model/workflow functions so preprocessing and controlled metric tests can still run without eagerly importing it. No DICOM support, calibration, explainability, segmentation, or hospital integration is implemented.

## Safety/data boundaries

Provided train and validation sets are used during fit; the test split is read only by evaluation. No split fallback is performed. The canonical label order is `NORMAL=0`, `PNEUMONIA=1`; score >= 0.5 maps to PNEUMONIA. The same preprocessing function is used by loader batches and single-image inference. Output charts/metrics are evidence only after the relevant workflow has produced them. A public test split is not an external clinical validation cohort.
