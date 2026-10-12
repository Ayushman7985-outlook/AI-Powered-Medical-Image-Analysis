import numpy as np
import pytest

from src.config import DEFAULT_SETTINGS, Settings
from src.model import build_model
from src.predict import load_trained_model, predict_image
from src.preprocessing import ImagePreprocessingError


class FixedModel:
    def __init__(self, score):
        self.score = score
        self.input_shape_seen = None

    def predict(self, values, verbose=0):
        self.input_shape_seen = values.shape
        return np.array([[self.score]], dtype=np.float32)


def test_prediction_uses_batched_shared_preprocessing_and_label_mapping():
    model = FixedModel(0.8)
    result = predict_image(np.zeros((30, 40), dtype=np.uint8), model=model)
    assert model.input_shape_seen == (1, DEFAULT_SETTINGS.image_height, DEFAULT_SETTINGS.image_width, 3)
    assert result["predicted_class"] == "PNEUMONIA"
    assert result["model_confidence"] == pytest.approx(0.8)
    assert "not a clinical" in result["confidence_note"]


def test_normal_prediction_uses_complementary_confidence():
    result = predict_image(np.zeros((10, 10, 3), dtype=np.uint8), model=FixedModel(0.2))
    assert result["predicted_class"] == "NORMAL"
    assert result["model_confidence"] == pytest.approx(0.8)


def test_invalid_image_fails_and_missing_model_is_actionable(tmp_path):
    with pytest.raises(ImagePreprocessingError):
        predict_image(b"bad", model=FixedModel(0.7))
    with pytest.raises(FileNotFoundError, match="Train one"):
        load_trained_model(tmp_path / "absent.keras")


def test_saved_cnn_loads_and_predicts_after_shared_preprocessing(tmp_path):
    tf = pytest.importorskip("tensorflow")
    settings = Settings(
        image_height=32,
        image_width=32,
        model_path=tmp_path / "small_model.keras",
        outputs_dir=tmp_path / "outputs",
    )
    model = build_model(settings)
    model.save(settings.model_path)

    loaded_model = load_trained_model(settings.model_path)
    result = predict_image(np.zeros((24, 40, 3), dtype=np.uint8), model=loaded_model, settings=settings)

    assert result["predicted_class"] in {"NORMAL", "PNEUMONIA"}
    assert 0 <= result["model_confidence"] <= 1
    assert loaded_model.input_shape == (None, 32, 32, 3)
    assert isinstance(loaded_model, tf.keras.Model)
