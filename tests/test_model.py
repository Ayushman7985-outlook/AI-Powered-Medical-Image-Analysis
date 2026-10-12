import pytest
import numpy as np

from src.config import Settings
from src.model import build_model


def test_model_output_shape_and_binary_configuration():
    pytest.importorskip("tensorflow")
    model = build_model(Settings(image_height=32, image_width=32))
    assert model.output_shape == (None, 1)
    assert model.loss == "binary_crossentropy"
    scores = model.predict(np.zeros((2, 32, 32, 3), dtype=np.float32), verbose=0)
    assert scores.shape == (2, 1)
    assert np.isfinite(scores).all()
