# SPDX-License-Identifier: Apache-2.0

"""Load immutable AIF, XCAT label, and kinetic lookup inputs once per batch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat

from .config import InputsConfig


@dataclass(frozen=True)
class InputData:
    label_full_xyz: np.ndarray
    aif_time_min: np.ndarray
    aif_cp: np.ndarray
    regions: dict[int, dict[str, float | str]]


def load_inputs(config):
    aif_path = _existing(config.aif_mat)
    xcat_path = _existing(config.xcat_mat)
    lookup_path = _existing(config.lookup_table)

    aif_mat = loadmat(aif_path, struct_as_record=False, squeeze_me=True)
    if "aif" not in aif_mat:
        raise KeyError(f"{aif_path} does not contain MATLAB variable 'aif'")
    aif = aif_mat["aif"]
    aif_cp = np.asarray(aif.dat, dtype=np.float64).ravel()
    aif_time_min = np.asarray(aif.tt, dtype=np.float64).ravel()
    if aif_cp.shape != aif_time_min.shape or aif_cp.size < 2:
        raise ValueError("AIF time and concentration arrays must have the same nontrivial shape")
    order = np.argsort(aif_time_min)
    aif_time_min = aif_time_min[order]
    aif_cp = aif_cp[order]
    if np.any(np.diff(aif_time_min) <= 0):
        raise ValueError("AIF time values must be unique and strictly increasing")
    if not np.all(np.isfinite(aif_cp)) or np.any(aif_cp < 0):
        raise ValueError("AIF concentrations must be finite and nonnegative")

    xcat_mat = loadmat(xcat_path, struct_as_record=False, squeeze_me=True)
    if "xcat_dat" not in xcat_mat:
        raise KeyError(f"{xcat_path} does not contain MATLAB variable 'xcat_dat'")
    label_full_xyz = np.asarray(xcat_mat["xcat_dat"], dtype=np.int16)
    if label_full_xyz.ndim != 3:
        raise ValueError("XCAT label must be a 3-D array in x,y,z order")

    regions = load_xcat_lookup_table(lookup_path)
    return InputData(
        label_full_xyz=label_full_xyz,
        aif_time_min=aif_time_min,
        aif_cp=aif_cp,
        regions=regions
    )


def parse_numbers(text):
    text = text.replace("D", "E").replace("d", "e")
    values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return [float(value) for value in values]


def load_xcat_lookup_table(path):
    regions: dict[int, dict[str, float | str]] = {}
    pattern = re.compile(r"^\s*(?P<name>\w+)\s*=\s*(?P<label>\d+)\s*#\s*(?P<rest>.*)$")
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("%", "//")):
                continue
            match = pattern.match(stripped)
            if match is None:
                continue
            rest = match.group("rest")
            description, number_text = rest.split(";", 1) if ";" in rest else (rest, "")
            values = parse_numbers(number_text)
            if len(values) < 5:
                raise ValueError(f"lookup line {line_number} has fewer than five kinetic parameters")
            k1, k2, k3, k4, vb = values[-5:]
            label = int(match.group("label"))
            regions[label] = {
                "name": match.group("name"),
                "desc": description.strip(),
                "k1": float(k1),
                "k2": float(k2),
                "k3": float(k3),
                "k4": float(k4),
                "Vb": float(vb)
            }
    if not regions:
        raise ValueError(f"no regions could be parsed from {path}")
    return regions


def _existing(path_text):
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path
