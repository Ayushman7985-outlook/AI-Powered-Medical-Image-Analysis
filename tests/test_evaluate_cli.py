import numpy as np
import pytest

import main as cli
from main import build_parser
from src.evaluate import calculate_metrics


def test_metrics_for_controlled_binary_labels():
    try:
        import sklearn.metrics  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"scikit-learn compiled metrics are unavailable in this environment: {exc}")
    metrics, matrix, report = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert matrix.tolist() == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 0.8
    assert metrics["positive_class"] == "PNEUMONIA"
    assert report["PNEUMONIA"]["support"] == 2


def test_cli_argument_parsing():
    args = build_parser().parse_args(["train", "--epochs", "2", "--batch-size", "3", "--smoke-test"])
    assert args.command == "train"
    assert args.epochs == 2
    assert args.batch_size == 3
    assert args.smoke_test
    prediction = build_parser().parse_args(["predict", "--image", "scan.png"])
    assert prediction.image.name == "scan.png"


def test_cli_dispatches_train_evaluate_and_predict(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_train(settings, *, quick, epochs, batch_size):
        calls.append(("train", settings, quick, epochs, batch_size))

    def fake_evaluate(settings, *, quick):
        calls.append(("evaluate", settings, quick))

    def fake_predict(image, *, settings, save_result):
        calls.append(("predict", image, settings, save_result))
        return {"predicted_class": "NORMAL", "model_confidence": 0.9, "confidence_note": "test score"}

    monkeypatch.setattr(cli, "train_model", fake_train)
    monkeypatch.setattr(cli, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(cli, "predict_image", fake_predict)

    assert cli.main(["train", "--dataset", str(tmp_path), "--epochs", "2", "--smoke-test"]) == 0
    assert cli.main(["evaluate", "--dataset", str(tmp_path), "--smoke-test"]) == 0
    assert cli.main(["predict", "--image", str(tmp_path / "scan.png"), "--save-result"]) == 0
    assert [call[0] for call in calls] == ["train", "evaluate", "predict"]
    assert calls[0][2:] == (True, 2, None)
    assert calls[1][2:] == (True,)
    assert calls[2][1] == tmp_path / "scan.png"
    assert calls[2][3] is True
    assert "Predicted class: NORMAL" in capsys.readouterr().out
