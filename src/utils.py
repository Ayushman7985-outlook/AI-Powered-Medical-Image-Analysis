"""Small shared utilities for optional dependencies and safe JSON output."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_directory(path: str | Path) -> Path:
    """Create and return a directory, with a friendly permissions error."""
    directory = Path(path).expanduser()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Cannot create output directory {directory}: {exc}") from exc
    return directory


def json_safe(value: Any) -> Any:
    """Convert NumPy and Path values into JSON-serializable equivalents."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: str | Path, value: Any) -> Path:
    """Write indented JSON and return its path."""
    destination = Path(path)
    ensure_directory(destination.parent)
    destination.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination
