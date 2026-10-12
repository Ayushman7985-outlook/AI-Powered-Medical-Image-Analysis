"""Central project configuration shared by training, evaluation, and inference."""

from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
POSITIVE_CLASS = "PNEUMONIA"


@dataclass(frozen=True)
class Settings:
    """Paths and model settings; override values rather than duplicating defaults."""

    data_dir: Path = PROJECT_ROOT / "data" / "chest_xray"
    model_path: Path = PROJECT_ROOT / "models" / "chest_xray_cnn.keras"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    image_height: int = 160
    image_width: int = 160
    channels: int = 3
    batch_size: int = 16
    epochs: int = 12
    seed: int = 42
    smoke_test_images_per_class: int = 8

    def with_overrides(self, **values: object) -> "Settings":
        """Return a settings copy with selected fields changed."""
        return replace(self, **values)


DEFAULT_SETTINGS = Settings()
