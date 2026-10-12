# Interview preparation: questions and answers

Answers describe the implementation in this repository; no training metrics are claimed unless produced locally.

## Fundamentals

**What is medical image analysis?** Applying computational methods to images such as X-rays to measure or classify patterns that can support professional interpretation. This prototype is educational, not clinical.

**What is image classification?** Assigning an input image to one of a fixed set of categories. Here the classes are NORMAL and PNEUMONIA.

**What is a CNN?** A neural network that learns spatial filters over image regions and combines their outputs into increasingly useful features.

**What do convolution and pooling do?** Convolution filters respond to local patterns; max pooling downsamples feature maps, reducing spatial detail and computation.

**Why normalize images?** The implementation scales pixel values from 0–255 to 0–1, giving optimization a consistent numeric range.

**Why resize?** A batch requires a common shape; every image becomes 160×160 RGB through shared preprocessing.

**What are train, validation, and test sets?** Training updates parameters, validation monitors fit/early stopping, and a separate test split estimates performance after choices are complete. Test data must not tune the model.

## Project implementation

**Why this dataset?** The Mendeley Data chest X-ray dataset is publicly accessible, has labeled train/test material and is commonly used for learning. It is not a representative clinical cohort; check citation, license, and restrictions.

**How are images loaded?** `data_loader.py` validates class-folder layout and streams paths in batches; `preprocessing.py` decodes each through OpenCV.

**How does label encoding work?** `config.py` fixes NORMAL to 0 and PNEUMONIA to 1. This matches the model's sigmoid output and positive-class metrics.

**How is the CNN structured?** Three Conv2D/ReLU + max-pooling blocks, global average pooling, dropout, a 64-unit dense layer, then one sigmoid unit.

**How does training work?** Keras minimizes binary cross-entropy with Adam over train batches, checks validation loss, early-stops/restores best weights, reduces learning rate, and saves the model/history/curves.

**Where is the model saved?** By default `models/chest_xray_cnn.keras`, configurable through `--model`.

**How does prediction work?** The saved Keras model receives the image processed by the exact shared RGB resize/normalization function. Scores >= 0.5 map to PNEUMONIA.

**How is confidence calculated?** It is the model score for predicted class (pneumonia score or its complement). It is uncalibrated, not a clinical probability.

**How are metrics generated?** Evaluation predicts on only the test split; scikit-learn calculates accuracy, PNEUMONIA precision/recall/F1, support, report, and fixed-order confusion matrix.

**How does Streamlit call the model?** Upload bytes go to `predict_image`; the UI caches the loaded model for a specific model path/modified time. It never trains at startup.

## Evaluation and reasoning

**Accuracy?** Correct predictions divided by all examples; class imbalance can make it misleading.

**Precision, recall, F1?** Precision is true positives among predicted positives; recall is true positives among actual positives; F1 is their harmonic mean.

**Confusion matrix?** Counts actual vs predicted class pairs, including false negatives and false positives. Rows are actual, columns predicted here.

**Why may recall matter?** Low pneumonia recall means more actual positive examples are missed. False positives also matter because they can cause needless alarm and follow-up. Clinical trade-offs require domain experts and prospective evidence.

**Class imbalance?** Unequal class counts; a model can look accurate by favoring the majority class. Check support and per-class metrics.

**Overfitting and mitigation?** Memorizing training-specific patterns harms new data. More representative data, regularization, dropout, early stopping, and careful split design can help. The code includes dropout and early stopping, not a guarantee.

**Data leakage?** Information from evaluation or near-duplicate/patient-related examples leaks into training/selection, inflating estimates. Keep patient-level examples separated where metadata allows and never tune to test.

**Distribution shift?** The deployment data differ in population, prevalence, scanner, protocol, or institution; learned associations may fail.

**Why isn't confidence certainty?** Neural scores can be miscalibrated and reflect dataset associations, not a validated probability for an individual patient.

## Module failures and debugging

**Dataset folders missing?** Validation raises actionable paths. Check root passed to `--dataset` contains all train/val/test splits and both classes.

**Images have inconsistent dimensions?** Normal; shared preprocessing resizes them. If decode fails, loader warns/skips corrupt supported files.

**Model fails to load?** Check exact `.keras` path, write completion, TensorFlow compatibility, and use the training version/environment.

**Labels reversed?** Check `CLASS_TO_INDEX` and class directory names: NORMAL=0, PNEUMONIA=1. Do not independently reorder labels.

**TensorFlow import fails?** Check Python 3.11/3.12 and supported platform, venv interpreter and installation output.

**Training accuracy rises, validation falls?** Suspect overfit, patient leakage, small/biased split, or distribution differences. Inspect history and split provenance; test results must not guide tuning.

**Confusion matrix wrong size?** Evaluator explicitly uses `[0,1]`; confirm test support and that no labels were incorrectly encoded.

**Prediction differs from expected preprocessing?** Verify same Settings and saved model; prediction calls shared preprocessing. Check image bytes, shape, RGB conversion, scaling and threshold.

**Streamlit cannot find model?** Train first; launch from repo root or confirm both app and CLI use same `--model` default path.

**Graphs missing?** Check training/evaluation succeeded and real history/test predictions were created. App omits absent charts rather than inventing them.

**Memory excessive?** Loader streams batches. Reduce `--batch-size`; use a smoke test for plumbing only, not conclusions.

## Industry and behavioral discussion

**Architecture decisions?** One central settings object, one preprocessing path, lazy TensorFlow imports, provided splits, explicit CLI, and no app-triggered training reduce inconsistent behavior and make tests possible.

**Limitations?** Small custom CNN, one public dataset, no external validation/calibration/explainability, non-DICOM inputs, no privacy/security/hospital integration, and no clinical governance.

**Ethics/privacy?** Public does not mean unrestricted; follow attribution/license. Avoid identifiable patient data and never publish private images. Be clear about limitations and not overstate scores.

**Deployment readiness?** Not ready. Would require representative multi-site data, patient-level leakage review, external/prospective evaluation, calibrated and clinically chosen thresholds, human factors, monitoring, security/privacy controls, workflow validation, governance and applicable regulatory approval.

**What would you improve?** Verify split provenance, collect more diverse licensed data, run external validation, assess calibration and subgroup performance, compare transfer learning on fixed splits, and conduct clinical/ethics review before any use beyond education.
