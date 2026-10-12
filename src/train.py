"""Training orchestration for the educational chest X-ray CNN."""

import time
from pathlib import Path

from .config import DEFAULT_SETTINGS, Settings
from .data_loader import apply_quick_subset, as_tf_dataset, validate_dataset
from .model import build_model
from .utils import ensure_directory, write_json
from .visualization import plot_training_history


def train_model(settings: Settings = DEFAULT_SETTINGS, *, quick: bool = False, epochs: int | None = None,
                batch_size: int | None = None) -> dict:
    """Validate, train, and save a Keras model plus its actual history and charts."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is unavailable. Use a supported Python environment and install requirements.txt.") from exc
    effective = settings.with_overrides(
        epochs=epochs if epochs is not None else (1 if quick else settings.epochs),
        batch_size=batch_size if batch_size is not None else (4 if quick else settings.batch_size),
    )
    splits = validate_dataset(effective.data_dir)
    if quick:
        splits = apply_quick_subset(splits, effective.smoke_test_images_per_class)
        print("SMOKE TEST: a small subset is being used; this is not meaningful model evaluation.")
    train_split, validation_split = splits["train"], splits["val"]
    if min(train_split.counts.values()) < 1 or min(validation_split.counts.values()) < 1:
        raise ValueError("Training and validation splits must each contain readable images from both classes.")
    ensure_directory(effective.model_path.parent)
    ensure_directory(effective.outputs_dir)
    model = build_model(effective)
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(effective.model_path), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    start = time.perf_counter()
    history = model.fit(
        as_tf_dataset(train_split, effective.batch_size, effective, shuffle=True),
        steps_per_epoch=(len(train_split.paths) + effective.batch_size - 1) // effective.batch_size,
        validation_data=as_tf_dataset(validation_split, effective.batch_size, effective),
        validation_steps=(len(validation_split.paths) + effective.batch_size - 1) // effective.batch_size,
        epochs=effective.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    duration = time.perf_counter() - start
    model.save(effective.model_path)
    history_data = {key: [float(item) for item in values] for key, values in history.history.items()}
    write_json(effective.outputs_dir / "training_history.json", {
        "history": history_data, "duration_seconds": duration, "epochs_completed": len(history.epoch),
        "smoke_test": quick, "train_counts": train_split.counts, "validation_counts": validation_split.counts,
    })
    plot_training_history(history_data, effective.outputs_dir)
    print(f"Training completed in {duration:.1f}s. Model saved to {effective.model_path}")
    return {"model_path": effective.model_path, "history": history_data, "duration_seconds": duration}
