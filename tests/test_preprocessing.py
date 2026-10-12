import cv2
import numpy as np
import pytest

from src.config import Settings
from src.preprocessing import ImagePreprocessingError, preprocess_array, preprocess_image


def small_settings():
    return Settings(image_height=12, image_width=10)


def test_rgb_preprocessing_dimensions_and_normalization():
    image = np.full((8, 20, 3), 128, dtype=np.uint8)
    result = preprocess_array(image, small_settings())
    assert result.shape == (12, 10, 3)
    assert result.dtype == np.float32
    assert np.allclose(result, 128 / 255)


def test_grayscale_array_becomes_three_channels():
    result = preprocess_array(np.full((5, 7), 255, dtype=np.uint8), small_settings())
    assert result.shape == (12, 10, 3)
    assert result.min() == result.max() == 1.0


def test_grayscale_file_decodes_as_rgb(tmp_path):
    path = tmp_path / "gray.png"
    assert cv2.imwrite(str(path), np.zeros((7, 9), dtype=np.uint8))
    result = preprocess_image(path, small_settings())
    assert result.shape == (12, 10, 3)


def test_empty_and_invalid_images_raise_clear_error():
    with pytest.raises(ImagePreprocessingError, match="empty"):
        preprocess_array(np.array([]), small_settings())
    with pytest.raises(ImagePreprocessingError, match="not a readable"):
        preprocess_image(b"not an image", small_settings())


def test_out_of_range_values_are_rejected():
    with pytest.raises(ImagePreprocessingError, match="0–255"):
        preprocess_array(np.full((2, 2, 3), 300), small_settings())
