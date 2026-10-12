"""Single-image inference using the exact training preprocessing path."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES, DEFAULT_SETTINGS, Settings
from .preprocessing import prepare_model_input
from .utils import write_json


def load_trained_model(model_path: str | Path = DEFAULT_SETTINGS.model_path):
    """Load a saved Keras model, with an actionable missing-file error."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}. Train one with `python main.py train` first.")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for inference. Install requirements.txt.") from exc
    try:
        return tf.keras.models.load_model(path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not load model at {path}: {exc}") from exc


def predict_image(image: str | Path | bytes | bytearray, *, model=None,
                  settings: Settings = DEFAULT_SETTINGS, save_result: bool = False) -> dict:
    """Predict NORMAL/PNEUMONIA and return the model score (not clinical certainty)."""
    model = model if model is not None else load_trained_model(settings.model_path)
    model_input = prepare_model_input(image, settings)
    scores = np.asarray(model.predict(model_input, verbose=0)).reshape(-1)
    if scores.size != 1 or not np.isfinite(scores[0]) or not 0 <= scores[0] <= 1:
        raise ValueError("The model must return one finite sigmoid score in the range [0, 1].")
    pneumonia_score = float(scores[0])
    predicted_index = int(pneumonia_score >= 0.5)
    model_confidence = pneumonia_score if predicted_index == 1 else 1.0 - pneumonia_score
    result = {
        "predicted_class": CLASS_NAMES[predicted_index],
        "model_confidence": model_confidence,
        "pneumonia_model_score": pneumonia_score,
        "threshold": 0.5,
        "confidence_note": "Uncalibrated model score; not a clinical probability or diagnostic certainty.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if save_result:
        write_json(settings.outputs_dir / "prediction_result.json", result)
    return result
