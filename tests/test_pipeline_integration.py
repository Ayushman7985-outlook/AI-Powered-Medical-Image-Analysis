"""Temporary controlled-image integration check; not a model-quality evaluation."""

import cv2
import numpy as np
import pytest

from src.config import CLASS_NAMES, Settings
from src.evaluate import evaluate_model
from src.predict import predict_image
from src.train import train_model


def test_tiny_temporary_fixture_trains_evaluates_and_predicts(tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("sklearn.metrics")

    data_root = tmp_path / "fixture_data"
    for split in ("train", "val", "test"):
        for class_index, class_name in enumerate(CLASS_NAMES):
            folder = data_root / split / class_name
            folder.mkdir(parents=True)
            image = np.full((40, 40), 32 + class_index * 160, dtype=np.uint8)
            assert cv2.imwrite(str(folder / "fixture.png"), image)

    settings = Settings(
        data_dir=data_root,
        model_path=tmp_path / "models" / "fixture.keras",
        outputs_dir=tmp_path / "outputs",
        image_height=32,
        image_width=32,
        batch_size=2,
        epochs=1,
        smoke_test_images_per_class=1,
    )

    training = train_model(settings, quick=True)
    evaluation = evaluate_model(settings)
    prediction = predict_image(
        data_root / "test" / "NORMAL" / "fixture.png",
        settings=settings,
        save_result=True,
    )

    assert training["model_path"].is_file()
    assert (settings.outputs_dir / "training_history.json").is_file()
    assert (settings.outputs_dir / "training_accuracy.png").is_file()
    assert (settings.outputs_dir / "training_loss.png").is_file()
    assert evaluation["metrics"]["sample_count"] == 2
    assert (settings.outputs_dir / "metrics.json").is_file()
    assert (settings.outputs_dir / "confusion_matrix.png").is_file()
    assert (settings.outputs_dir / "prediction_examples.png").is_file()
    assert prediction["predicted_class"] in CLASS_NAMES
    assert (settings.outputs_dir / "prediction_result.json").is_file()
