"""Shared X-ray image decoding and preprocessing for training and inference."""

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import DEFAULT_SETTINGS, Settings


class ImagePreprocessingError(ValueError):
    """Raised when an image cannot be decoded or prepared for the model."""


def load_image(source: str | Path | bytes | bytearray | BinaryIO) -> np.ndarray:
    """Decode an image as an RGB uint8 array, raising a useful error on failure."""
    try:
        if isinstance(source, (bytes, bytearray)):
            if not source:
                raise ImagePreprocessingError("The image file is empty.")
            decoded = Image.open(BytesIO(source))
        else:
            if not isinstance(source, (str, Path)) and hasattr(source, "seek"):
                source.seek(0)
            decoded = Image.open(source)
        with decoded:
            image = np.asarray(decoded.convert("RGB"))
    except UnidentifiedImageError as exc:
        raise ImagePreprocessingError("The file is not a readable supported image.") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ImagePreprocessingError(f"Could not read the image: {exc}") from exc

    if image.size == 0:
        raise ImagePreprocessingError("The file is not a readable supported image.")
    return image


def preprocess_array(image: np.ndarray, settings: Settings = DEFAULT_SETTINGS) -> np.ndarray:
    """Resize an RGB/grayscale array and scale pixels to float32 in [0, 1]."""
    image = np.asarray(image)
    if image.size == 0:
        raise ImagePreprocessingError("The supplied image is empty.")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim != 3 or image.shape[2] != settings.channels:
        raise ImagePreprocessingError("Expected an image with one, three, or four channels.")
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ImagePreprocessingError("Image pixels must be finite numeric values.")
    if image.max(initial=0) > 255 or image.min(initial=0) < 0:
        raise ImagePreprocessingError("Image pixel values must be in the range 0–255.")

    resized = cv2.resize(
        image.astype(np.uint8),
        (settings.image_width, settings.image_height),
        interpolation=cv2.INTER_AREA if image.shape[0] > settings.image_height else cv2.INTER_LINEAR,
    )
    return (resized.astype(np.float32) / 255.0).astype(np.float32, copy=False)


def preprocess_image(source: str | Path | bytes | bytearray | BinaryIO | np.ndarray,
                     settings: Settings = DEFAULT_SETTINGS) -> np.ndarray:
    """Apply the canonical preprocessing and return (height, width, 3)."""
    image = source if isinstance(source, np.ndarray) else load_image(source)
    return preprocess_array(image, settings)


def prepare_model_input(source: str | Path | bytes | bytearray | BinaryIO | np.ndarray,
                        settings: Settings = DEFAULT_SETTINGS) -> np.ndarray:
    """Preprocess an image and add the batch dimension required by Keras."""
    return np.expand_dims(preprocess_image(source, settings), axis=0)
