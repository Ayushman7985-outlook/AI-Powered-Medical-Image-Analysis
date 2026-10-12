"""Reusable plots generated from real training, evaluation, and prediction data."""

from pathlib import Path
from typing import Sequence

import numpy as np

from .config import CLASS_NAMES, DEFAULT_SETTINGS
from .utils import ensure_directory


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required to generate charts. Install requirements.txt.") from exc
    return plt


def plot_training_history(history: dict, output_dir: str | Path) -> tuple[Path, Path]:
    """Save accuracy and loss curves from Keras History values."""
    plt = _pyplot()
    destination = ensure_directory(output_dir)
    paths = []
    for metric, title, filename in (("accuracy", "Training and validation accuracy", "training_accuracy.png"),
                                    ("loss", "Training and validation loss", "training_loss.png")):
        train_key = metric if metric in history else f"binary_{metric}"
        validation_key = f"val_{train_key}"
        if train_key not in history or validation_key not in history:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(history[train_key], label="Training")
        ax.plot(history[validation_key], label="Validation")
        ax.set(title=title, xlabel="Epoch", ylabel=metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = destination / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    if len(paths) != 2:
        raise ValueError("Training history must contain accuracy/loss and their validation values.")
    return paths[0], paths[1]


def plot_confusion_matrix(matrix: np.ndarray, output_path: str | Path) -> Path:
    """Save an annotated confusion matrix using the fixed class ordering."""
    plt = _pyplot()
    matrix = np.asarray(matrix)
    if matrix.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 confusion matrix, received {matrix.shape}.")
    path = Path(output_path)
    ensure_directory(path.parent)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(xticks=range(2), yticks=range(2), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted label", ylabel="Actual label", title="Test-set confusion matrix")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(int(matrix[row, column])), ha="center", va="center",
                    color="white" if matrix[row, column] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prediction_examples(images: Sequence[np.ndarray], actual: Sequence[int], predicted: Sequence[int],
                             scores: Sequence[float], output_path: str | Path) -> Path:
    """Save a small real test-set image grid with actual labels and model scores."""
    plt = _pyplot()
    count = min(len(images), len(actual), len(predicted), len(scores), 8)
    if count == 0:
        raise ValueError("At least one prediction example is required.")
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.5 * rows), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= count:
            ax.axis("off")
            continue
        ax.imshow(np.clip(images[index], 0, 1))
        ax.set_title(f"Actual: {CLASS_NAMES[int(actual[index])] }\nPredicted: {CLASS_NAMES[int(predicted[index])]} ({scores[index]:.2f})")
        ax.axis("off")
    fig.suptitle("Test-set examples — model score is not clinical certainty")
    fig.tight_layout()
    path = Path(output_path)
    ensure_directory(path.parent)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
