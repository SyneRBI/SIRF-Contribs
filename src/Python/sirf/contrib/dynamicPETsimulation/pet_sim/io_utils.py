# SPDX-License-Identifier: Apache-2.0

"""HDF5, JSON, logging, and sample-directory helpers."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_logging(log_path, verbose=False):
    logger = logging.getLogger(f"pet_sim.{log_path.parent.name}")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def close_logger(logger):
    """Close per-sample file handles so long batches do not exhaust descriptors."""
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, default=_json_default) + "\n")


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_completed_sample(sample_dir, h5_name):
    status_path = sample_dir / "status.json"
    h5_path = sample_dir / h5_name
    if not status_path.exists() or not h5_path.exists():
        return False
    try:
        return read_json(status_path).get("state") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def write_h5_dataset(h5_path, dataset_name, data, *, compression=True, dtype=None):
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data)
    if dtype is not None:
        array = array.astype(dtype)
    with h5py.File(h5_path, "a") as handle:
        group, name = _h5_parent(handle, dataset_name)
        if name in group:
            del group[name]
        kwargs = {"compression": "gzip", "shuffle": True} if compression and array.ndim else {}
        group.create_dataset(name, data=array, **kwargs)


def recreate_h5_dataset(handle, dataset_name, shape, dtype, *, compression=True):
    group, name = _h5_parent(handle, dataset_name)
    if name in group:
        del group[name]
    kwargs = {"compression": "gzip", "shuffle": True} if compression else {}
    return group.create_dataset(name, shape=shape, dtype=dtype, **kwargs)


def write_h5_attrs(h5_path, group_name, attrs):
    with h5py.File(h5_path, "a") as handle:
        group = handle.require_group(group_name.strip("/"))
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple)):
                group.attrs[key] = json.dumps(value, ensure_ascii=False, default=_json_default)
            elif isinstance(value, Path):
                group.attrs[key] = str(value)
            elif value is None:
                group.attrs[key] = "null"
            else:
                group.attrs[key] = value


def validate_finite(name, array, *, require_positive= False):
    array = np.asarray(array)
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} contains NaN or Inf")
    if require_positive and not np.any(array > 0):
        raise RuntimeError(f"{name} has no positive values")


def array_summary(array):
    array = np.asarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "sum": float(np.nansum(array))
    }


@contextmanager
def sample_stage(status_path,status,stage_name):
    status["stage"] = stage_name
    status["updated_at"] = utc_now()
    write_json_atomic(status_path, status)
    yield


def _h5_parent(handle, dataset_name):
    parts = dataset_name.strip("/").split("/")
    group: h5py.Group = handle
    for part in parts[:-1]:
        group = group.require_group(part)
    return group, parts[-1]


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")
