import cv2
import numpy as np
import pytest

from src.config import CLASS_NAMES, CLASS_TO_INDEX
from src.data_loader import DatasetValidationError, apply_quick_subset, validate_dataset


def make_dataset(root, *, omit=None, empty=None):
    omit = omit or set()
    empty = empty or set()
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            if (split, class_name) in omit:
                continue
            folder = root / split / class_name
            folder.mkdir(parents=True)
            if (split, class_name) not in empty:
                assert cv2.imwrite(str(folder / "sample.png"), np.zeros((6, 6), dtype=np.uint8))
    return root


def test_class_mapping_is_consistent():
    assert CLASS_TO_INDEX == {"NORMAL": 0, "PNEUMONIA": 1}


def test_valid_dataset_and_counts(tmp_path):
    splits = validate_dataset(make_dataset(tmp_path))
    assert set(splits) == {"train", "val", "test"}
    assert splits["test"].counts == {"NORMAL": 1, "PNEUMONIA": 1}


def test_missing_dataset_directory_fails(tmp_path):
    with pytest.raises(DatasetValidationError, match="not found"):
        validate_dataset(tmp_path / "missing")


def test_missing_class_and_empty_folder_fail(tmp_path):
    with pytest.raises(DatasetValidationError, match="Missing class directory"):
        validate_dataset(make_dataset(tmp_path / "missing", omit={("test", "NORMAL")}))
    empty_root = tmp_path / "empty"
    make_dataset(empty_root, empty={("val", "PNEUMONIA")})
    with pytest.raises(DatasetValidationError, match="No supported images"):
        validate_dataset(empty_root)


def test_quick_subset_keeps_each_class(tmp_path):
    splits = validate_dataset(make_dataset(tmp_path))
    subset = apply_quick_subset(splits, 1)
    assert subset["train"].counts == {"NORMAL": 1, "PNEUMONIA": 1}
