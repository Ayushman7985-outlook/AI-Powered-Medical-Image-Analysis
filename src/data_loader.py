"""Dataset structure validation and memory-conscious TensorFlow input streams."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import CLASS_NAMES, CLASS_TO_INDEX, DEFAULT_SETTINGS, SUPPORTED_EXTENSIONS, Settings
from .preprocessing import ImagePreprocessingError, preprocess_image

SPLIT_NAMES = ("train", "val", "test")


class DatasetValidationError(ValueError):
    """Raised when the dataset layout cannot be used safely."""


@dataclass(frozen=True)
class DatasetSplit:
    """Image paths and integer labels for one provided dataset split."""

    name: str
    paths: tuple[Path, ...]
    labels: tuple[int, ...]
    counts: dict[str, int]


def _readable_image(path: Path) -> bool:
    try:
        preprocess_image(path)
        return True
    except (ImagePreprocessingError, OSError):
        return False


def validate_dataset(data_dir: str | Path, *, strict_images: bool = False) -> dict[str, DatasetSplit]:
    """Validate train/val/test class folders and omit unreadable images with warnings."""
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise DatasetValidationError(
            f"Dataset directory not found: {root}. Download the public dataset and pass its chest_xray folder."
        )
    splits: dict[str, DatasetSplit] = {}
    problems: list[str] = []
    for split_name in SPLIT_NAMES:
        split_dir = root / split_name
        if not split_dir.is_dir():
            problems.append(f"Missing split directory: {split_dir}")
            continue
        paths: list[Path] = []
        labels: list[int] = []
        counts: dict[str, int] = {}
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                problems.append(f"Missing class directory: {class_dir}")
                counts[class_name] = 0
                continue
            candidates = sorted(p for p in class_dir.rglob("*")
                                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
            if not candidates:
                problems.append(f"No supported images in: {class_dir}")
            good = [path for path in candidates if _readable_image(path)]
            bad = len(candidates) - len(good)
            if bad:
                message = f"{bad} unreadable image(s) in {class_dir}; they will be skipped."
                if strict_images:
                    problems.append(message)
                else:
                    print(f"Warning: {message}")
            if candidates and not good:
                problems.append(f"No readable images remain in: {class_dir}")
            counts[class_name] = len(good)
            paths.extend(good)
            labels.extend([CLASS_TO_INDEX[class_name]] * len(good))
        if paths:
            order = np.argsort(np.asarray([str(path) for path in paths]))
            splits[split_name] = DatasetSplit(
                split_name,
                tuple(paths[int(i)] for i in order),
                tuple(labels[int(i)] for i in order),
                counts,
            )
    if problems:
        raise DatasetValidationError("Dataset validation failed:\n- " + "\n- ".join(problems))
    return splits


def apply_quick_subset(splits: dict[str, DatasetSplit], images_per_class: int) -> dict[str, DatasetSplit]:
    """Select a deterministic per-class subset from each split for smoke tests."""
    if images_per_class < 1:
        raise ValueError("images_per_class must be at least 1.")
    subset: dict[str, DatasetSplit] = {}
    for name, split in splits.items():
        selected_paths: list[Path] = []
        selected_labels: list[int] = []
        counts: dict[str, int] = {}
        for class_name, class_index in CLASS_TO_INDEX.items():
            pairs = [(path, label) for path, label in zip(split.paths, split.labels) if label == class_index]
            pairs = pairs[:images_per_class]
            counts[class_name] = len(pairs)
            selected_paths.extend(path for path, _ in pairs)
            selected_labels.extend(label for _, label in pairs)
        if any(count == 0 for count in counts.values()):
            raise DatasetValidationError(f"Quick subset for {name} is missing at least one class: {counts}")
        subset[name] = DatasetSplit(name, tuple(selected_paths), tuple(selected_labels), counts)
    return subset


def iter_batches(split: DatasetSplit, batch_size: int, settings: Settings = DEFAULT_SETTINGS,
                 *, shuffle: bool = False, seed: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield preprocessed batches without loading the full image collection in memory."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    indices = np.arange(len(split.paths))
    if shuffle:
        np.random.default_rng(seed if seed is not None else settings.seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        images: list[np.ndarray] = []
        labels: list[int] = []
        for index in batch_indices:
            try:
                images.append(preprocess_image(split.paths[int(index)], settings))
            except (ImagePreprocessingError, OSError) as exc:
                raise DatasetValidationError(f"Image became unreadable during loading: {split.paths[int(index)]}: {exc}") from exc
            labels.append(split.labels[int(index)])
        yield np.stack(images), np.asarray(labels, dtype=np.float32)


def as_tf_dataset(split: DatasetSplit, batch_size: int, settings: Settings = DEFAULT_SETTINGS,
                  *, shuffle: bool = False):
    """Wrap the shared Python preprocessor in a streaming tf.data.Dataset."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for training and evaluation. Install requirements.txt.") from exc

    output_signature = (
        tf.TensorSpec((None, settings.image_height, settings.image_width, settings.channels), tf.float32),
        tf.TensorSpec((None,), tf.float32),
    )

    def generator():
        while True:
            yield from iter_batches(split, batch_size, settings, shuffle=shuffle, seed=settings.seed)

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.prefetch(tf.data.AUTOTUNE)
