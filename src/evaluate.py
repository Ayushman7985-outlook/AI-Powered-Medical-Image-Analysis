"""Independent test-set metrics and evaluation artifacts."""

from pathlib import Path

import numpy as np

from .config import CLASS_NAMES, DEFAULT_SETTINGS, POSITIVE_CLASS, Settings
from .data_loader import apply_quick_subset, validate_dataset
from .utils import write_json
from .visualization import plot_confusion_matrix, plot_prediction_examples


def calculate_metrics(y_true, y_pred) -> tuple[dict, np.ndarray, dict]:
    """Calculate binary metrics; PNEUMONIA (index 1) is the positive class."""
    try:
        from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                     f1_score, precision_score, recall_score)
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for evaluation. Install requirements.txt.") from exc
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "positive_class": POSITIVE_CLASS,
        "sample_count": int(len(y_true)),
        "class_support": {name: int(np.sum(np.asarray(y_true) == index)) for index, name in enumerate(CLASS_NAMES)},
    }
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=list(CLASS_NAMES),
                                   output_dict=True, zero_division=0)
    return metrics, matrix, report


def evaluate_model(settings: Settings = DEFAULT_SETTINGS, *, quick: bool = False) -> dict:
    """Evaluate a saved model only on the supplied test split and save real artifacts."""
    if not settings.model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {settings.model_path}. Run the train command first.")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to load and evaluate the model.") from exc
    splits = validate_dataset(settings.data_dir)
    split = splits["test"]
    if quick:
        split = apply_quick_subset({"test": split}, settings.smoke_test_images_per_class)["test"]
        print("SMOKE TEST: test metrics use a small subset and are not meaningful model evaluation.")
    if not split.paths:
        raise ValueError("The test split contains no readable images.")
    model = tf.keras.models.load_model(settings.model_path)
    from .data_loader import iter_batches

    y_true: list[int] = []
    scores: list[float] = []
    examples: list[np.ndarray] = []
    actual_examples: list[int] = []
    for images, labels in iter_batches(split, settings.batch_size, settings):
        batch_scores = np.asarray(model.predict(images, verbose=0)).reshape(-1)
        scores.extend(float(value) for value in batch_scores)
        y_true.extend(int(value) for value in labels)
        for image in images[:max(0, 8 - len(examples))]:
            examples.append(image)
        actual_examples.extend(int(value) for value in labels[:max(0, 8 - len(actual_examples))])
    y_pred = [int(score >= 0.5) for score in scores]
    metrics, matrix, report = calculate_metrics(y_true, y_pred)
    metrics["smoke_test"] = quick
    output_dir = settings.outputs_dir
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "classification_report.json", report)
    plot_confusion_matrix(matrix, output_dir / "confusion_matrix.png")
    plot_prediction_examples(examples, actual_examples, y_pred[:len(examples)], scores[:len(examples)],
                             output_dir / "prediction_examples.png")
    print("Test-set metrics (PNEUMONIA is the positive class):")
    for key in ("accuracy", "precision", "recall", "f1_score", "sample_count"):
        print(f"  {key}: {metrics[key]}")
    print(f"Saved metrics and figures to {output_dir}")
    return {"metrics": metrics, "confusion_matrix": matrix.tolist(), "classification_report": report}
