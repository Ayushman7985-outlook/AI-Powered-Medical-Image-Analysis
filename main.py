"""Command-line interface for training, evaluating, and predicting."""

import argparse
import sys
from pathlib import Path

from src.config import DEFAULT_SETTINGS
from src.evaluate import evaluate_model
from src.predict import predict_image
from src.train import train_model


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser independently for help output and tests."""
    parser = argparse.ArgumentParser(description="Educational chest X-ray classification (not for medical diagnosis).")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train and save the CNN using train/val only.")
    evaluate = commands.add_parser("evaluate", help="Evaluate the saved model on the held-out test split.")
    predict = commands.add_parser("predict", help="Predict the class of one image using a saved model.")
    for command_parser in (train, evaluate, predict):
        command_parser.add_argument("--dataset", type=Path, default=DEFAULT_SETTINGS.data_dir,
                                    help="Dataset chest_xray directory (default: %(default)s).")
        command_parser.add_argument("--model", type=Path, default=DEFAULT_SETTINGS.model_path,
                                    help="Keras model path (default: %(default)s).")
        command_parser.add_argument("--output-dir", type=Path, default=DEFAULT_SETTINGS.outputs_dir,
                                    help="Directory for generated JSON and charts.")
    train.add_argument("--epochs", type=int, default=None, help="Training epochs (smoke test defaults to 1).")
    train.add_argument("--batch-size", type=int, default=None, help="Batch size (smoke test defaults to 4).")
    train.add_argument("--smoke-test", action="store_true", help="Train one epoch on a small subset; not evaluation.")
    evaluate.add_argument("--smoke-test", action="store_true", help="Evaluate a small test subset; not meaningful metrics.")
    predict.add_argument("--image", required=True, type=Path, help="Path to one supported image file.")
    predict.add_argument("--save-result", action="store_true", help="Write outputs/prediction_result.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected workflow and return a process exit status."""
    args = build_parser().parse_args(argv)
    settings = DEFAULT_SETTINGS.with_overrides(
        data_dir=args.dataset.expanduser(), model_path=args.model.expanduser(),
        outputs_dir=args.output_dir.expanduser(),
    )
    try:
        if args.command == "train":
            train_model(settings, quick=args.smoke_test, epochs=args.epochs, batch_size=args.batch_size)
        elif args.command == "evaluate":
            evaluate_model(settings, quick=args.smoke_test)
        else:
            result = predict_image(args.image, settings=settings, save_result=args.save_result)
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Model confidence (uncalibrated): {result['model_confidence']:.1%}")
            print(result["confidence_note"])
        return 0
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
