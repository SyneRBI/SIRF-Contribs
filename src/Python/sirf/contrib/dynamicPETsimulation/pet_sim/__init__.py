# SPDX-License-Identifier: Apache-2.0

"""Configurable batch PET simulation pipeline derived from data_simulation_v13."""

from .config import SimulationConfig, load_config

__all__ = ["SimulationConfig", "load_config", "run_batch"]


def run_batch(*args, **kwargs):
    """Lazily import heavy HDF5/SIRF-facing modules only for a real run."""
    from .batch import run_batch as _run_batch

    return _run_batch(*args, **kwargs)
