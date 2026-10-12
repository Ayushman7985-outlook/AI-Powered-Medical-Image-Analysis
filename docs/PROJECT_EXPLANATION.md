# Project explanation and local simulation

## In plain language

Medical image analysis means using software to inspect images such as X-rays and identify patterns. Hospitals and diagnostic laboratories use imaging to help qualified professionals investigate a patient's condition. AI can learn associations between labeled examples and image patterns, then prioritize or classify new images. This project teaches binary classification: it outputs NORMAL or PNEUMONIA from a public chest X-ray dataset. The model can be wrong because training examples are limited, labels and images can be noisy, populations and equipment differ, and a neural net can latch onto irrelevant cues. It supports learning, not clinical judgement or a diagnosis.

## Actual implemented pipeline

`Public image dataset → dataset validation → image decoding and preprocessing → provided train/val/test splits → custom CNN → training → saved Keras model → test evaluation → prediction and plots → CLI / Streamlit`

There is no automatic download or training at app startup, and no validation data is replaced with test data.

### Data and preprocessing

Expected folders are `train/NORMAL`, `train/PNEUMONIA`, `val/...`, and `test/...`. `src/config.py` owns class order, image size (160 × 160 RGB), batch size, and default paths. The loader checks every provided split and both classes, looks for supported image extensions, attempts decoding and reports skipped unreadable files. It streams batches rather than retaining all resized images in memory.

Pillow decodes supported image files and converts grayscale and RGBA inputs to RGB; OpenCV resizes images consistently, and pixel values are scaled from 0–255 to float32 in [0, 1]. Resizing creates fixed-size tensors; normalization makes input magnitudes consistent for optimization. Labels are `NORMAL=0`, `PNEUMONIA=1`. Batch creation groups images/labels for efficient training. No augmentation is applied because unnecessary transformations can alter medically relevant structure.

The dataset source's archive may have a different nested top-level folder after extraction; point `--dataset` to the directory containing the three split folders. The implementation expects all three splits and will not invent a validation split or use the test split to tune. If you have no valid provided validation split, obtain a documented patient-level validation split separately before training; don't split images from the same patient across subsets.

### CNN and learning

`src/model.py` defines three convolution blocks (16, 32, 64 filters), each with ReLU and max pooling, followed by global average pooling, dropout, a 64-unit dense layer and one sigmoid output. Convolution filters learn local patterns; pooling reduces spatial detail and computation. The output is a scalar pneumonia model score. Threshold 0.5 maps scores below it to NORMAL and scores at/above it to PNEUMONIA. Training uses binary cross-entropy, Adam, early stopping, best validation-loss checkpointing, and learning-rate reduction. Loss measures prediction error; backpropagation computes how parameters contributed to loss and the optimizer updates them. Validation is used during training to monitor/stop learning, never the test split. Seeds are set where practical but hardware/library kernels can prevent bit-for-bit repeatability.

The score/confidence is an uncalibrated model output. Even a score of 0.99 does not mean 99% clinical certainty or a calibrated patient-specific disease probability.

### Evaluation, prediction, and artifacts

`src/evaluate.py` only evaluates `test`, reports accuracy, precision, recall, F1, class support, classification report, and a confusion matrix. PNEUMONIA is positive. A false negative is an actual pneumonia image predicted NORMAL; recall measures the portion of positive examples detected. False positives can also cause needless follow-up and harm. Metrics depend on threshold and sample composition; no test tuning is done. One public test split cannot establish reliability.

`src/predict.py` uses the same preprocessing function as training, loads the saved `.keras` model, maps its output using the same threshold/classes, and can write one JSON result. Visualization functions draw actual Keras history, actual test labels/predictions, and computed confusion matrix. The app calls these shared prediction functions and displays only existing real metrics/plots.

## Twelve-step local virtual simulation

1. Obtain the public archive at the Mendeley Data page linked in the root README; review its CC BY 4.0 terms and attribution. This simulates acquiring a permitted research dataset, not obtaining hospital data.
2. Extract it so a directory contains `train`, `val`, `test`, with both class folders in each. Keep test files separate.
3. Run `python main.py train --smoke-test` first or `python main.py train` for configured training. The loader validates and reports issues; it never fabricates images.
4. Preprocessing runs as images are loaded: RGB decode, resize, scale, label. This simulates a standardized input interface, not a clinical PACS/DICOM pipeline.
5. The CLI consumes the provided splits. The small subset is only a plumbing smoke test; it is not meaningful evaluation.
6. Training monitors validation and writes model/history/curves. This simulates offline model development, not a regulated training process.
7. Run `python main.py evaluate` to use the held-out test folder once. Do not use its scores to select hyperparameters.
8. Inspect `outputs/metrics.json`, `classification_report.json`, `confusion_matrix.png`, and `prediction_examples.png`. These are generated only by real runs.
9. Select an actual public test image, then run `python main.py predict --image PATH --save-result`.
10. Run `streamlit run app.py`, upload an image, and trigger the actual local model inference. No model means a clear error, not a demo prediction.
11. Capture real screenshots/logs listed in `images/README.md`; avoid including private data, secrets, or unlicensed images.
12. Publish code and docs only after inspecting Git staging and honoring data attribution. This simulates a portfolio demonstration—not patient care, clinical deployment, security review, or validation.

## Three approaches compared

| Approach                       | Libraries/data                                                                      | Difficulty   | Hardware                                         | Output                                              |
| ------------------------------ | ----------------------------------------------------------------------------------- | ------------ | ------------------------------------------------ | --------------------------------------------------- |
| Basic custom CNN (implemented) | Keras + class-labeled X-rays                                                        | Introductory | CPU possible, slower; small batches              | Trainable classifier with transparent layers        |
| Lightweight transfer learning  | Keras + pretrained MobileNet/EfficientNet and provenance-compatible weights         | Intermediate | Often faster convergence; memory/model dependent | Fine-tuned classifier using transferred features    |
| Advanced medical imaging       | Specialized libraries, explainability/segmentation/deployment and governed datasets | Advanced     | Often GPU, extensive validation                  | More complex imaging workflow, not implemented here |

The compact CNN avoids external weight downloads and keeps the educational mechanics explicit. Transfer learning can be a useful later experiment, but introduces weight provenance, licensing, and compatibility decisions. Advanced techniques do not make a model clinically safe by themselves.
