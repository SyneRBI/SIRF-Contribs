# SPDX-License-Identifier: Apache-2.0

"""Configuration loading, validation, and command-line overrides."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

DEFAULT_ROI_CLASSES = {
    1: "Body", 2: "Body", 3: "Body", 4: "Body", 5: "Body",
    6: "Body", 7: "Body", 8: "Body", 9: "Body", 10: "Body",
    11: "Body", 12: "Body", 13: "Body", 14: "Body", 15: "Lung",
    16: "Body", 17: "Body", 18: "Body", 20: "Body", 21: "Body",
    22: "Body", 23: "Body", 24: "Body", 25: "Body", 26: "Bone",
    27: "Bone", 28: "Bone", 29: "Body", 30: "Bone", 31: "Body",
    32: "Body", 33: "Body", 34: "Body", 35: "Body", 36: "Body",
    37: "Body", 38: "Body", 39: "Body", 45: "Body", 46: "Body",
    47: "Air", 52: "Air", 70: "Body", 71: "Body", 72: "Body",
    73: "Lung", 74: "Lung", 75: "Lung"
}


@dataclass
class InputsConfig:
    aif_mat: str = "input/AIF_FDG.mat"
    xcat_mat: str = "input/XCAT_Mask_400x400x900_act_NoLes.mat"
    lookup_table: str = "input/XCAT_mask_look_up_table.txt"


@dataclass
class BatchConfig:
    num_samples: int = 1
    start_index: int = 0
    master_seed: int = 20260715
    output_root: str = "batch_output"
    resume: bool = True
    fail_fast: bool = False


@dataclass
class AnatomyConfig:
    target_organ: str = "liver"
    organ_id: int = 13
    bed_z_size: int = 127
    voxel_cm_xyz: list[float] = field(default_factory=lambda: [0.203642, 0.203642, 0.2025])
    max_initial_translation_cm_xyz: list[float] = field(default_factory=lambda: [5.0, 5.0, 4.0])
    initial_scale_range: list[float] = field(default_factory=lambda: [0.95, 1.05])
    max_initial_rotation_deg_xyz: list[float] = field(default_factory=lambda: [2.0, 2.0, 2.0])


@dataclass
class FramesConfig:
    n_frames: int = 10
    duration_seconds: float = 60.0
    start_scan_seconds: float = 20.0 * 60.0
    end_scan_seconds: float = 60.0 * 60.0
    points_per_min: int = 100


@dataclass
class KineticsConfig:
    use_missing_replacement: bool = True
    replacement_name: str = "body_activity"


@dataclass
class LesionConfig:
    base_parameters: dict[str, float] = field(
        default_factory=lambda: {
            "k1": 1.056,
            "k2": 1.029,
            "k3": 0.320,
            "k4": 0.0,
            "Vb": 0.205
        }
    )
    parameter_multiplier_ranges: dict[str, list[float]] = field(
        default_factory=lambda: {
            "k1": [0.90, 1.10],
            "k2": [0.90, 1.10],
            "k3": [0.85, 1.15],
            "k4": [1.00, 1.00],
            "Vb": [0.90, 1.10]
        }
    )
    base_radius_xyz: list[float] = field(
        default_factory=lambda: [6.0, 6.0, 4.0]
    )
    radius_multiplier_range: list[float] = field(
        default_factory=lambda: [0.20, 1.20]
    )
    min_distance_margin_voxels: float = 1.0
    attenuation_class: str = "Body"


@dataclass
class AttenuationConfig:
    mu_cm: dict[str, float] = field(
        default_factory=lambda: {
            "Air": 0.0,
            "Body": 0.0927,
            "Lung": 0.0267,
            "Bone": 0.1305
        }
    )
    roi_classes: dict[int, str] = field(
        default_factory=lambda: deepcopy(DEFAULT_ROI_CLASSES)
    )
    unknown_nonzero_default: str = "Body"


@dataclass
class MotionConfig:
    min_events: int = 1
    max_events: int = 1
    max_translation_cm_xyz: list[float] = field(default_factory=lambda: [5.0, 5.0, 4.0])
    max_rotation_deg_xyz: list[float] = field(default_factory=lambda: [3.0, 3.0, 3.0])


@dataclass
class ScannerConfig:
    name: str = "Siemens mMR"
    span: int = 11
    max_ring_diff: int = 60


@dataclass
class ProjectionConfig:
    add_background: bool = True
    background_fraction: float = 0.05
    noise_scaling_factor: float = 20.0
    save_noisy_projections: bool = True


@dataclass
class ReconstructionConfig:
    num_subsets: int = 7
    num_subiterations: int = 63
    save_interfile: bool = False


@dataclass
class PatlakConfig:
    start_min: float = 20.0
    end_min: float = 60.0
    cp_epsilon: float = 1e-8


@dataclass
class OutputConfig:
    h5_name: str = "project_pet_dynamic.h5"
    compression: bool = True
    keep_failed_sample: bool = True


@dataclass
class SimulationConfig:
    inputs: InputsConfig = field(default_factory=InputsConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    anatomy: AnatomyConfig = field(default_factory=AnatomyConfig)
    frames: FramesConfig = field(default_factory=FramesConfig)
    kinetics: KineticsConfig = field(default_factory=KineticsConfig)
    lesion: LesionConfig = field(default_factory=LesionConfig)
    attenuation: AttenuationConfig = field(default_factory=AttenuationConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    patlak: PatlakConfig = field(default_factory=PatlakConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SimulationConfig":
        raw = deepcopy(raw)
        raw_roi = raw.get("attenuation", {}).get("roi_classes")
        if raw_roi is not None:
            raw["attenuation"]["roi_classes"] = {int(key): value for key, value in raw_roi.items()}
        data = deep_merge(asdict(cls()), raw)
        attenuation = dict(data["attenuation"])
        attenuation["roi_classes"] = {int(key): value for key, value in attenuation["roi_classes"].items()}
        config = cls(
            inputs=InputsConfig(**data["inputs"]),
            batch=BatchConfig(**data["batch"]),
            anatomy=AnatomyConfig(**data["anatomy"]),
            frames=FramesConfig(**data["frames"]),
            kinetics=KineticsConfig(**data["kinetics"]),
            lesion=LesionConfig(**data["lesion"]),
            attenuation=AttenuationConfig(**attenuation),
            motion=MotionConfig(**data["motion"]),
            scanner=ScannerConfig(**data["scanner"]),
            projection=ProjectionConfig(**data["projection"]),
            reconstruction=ReconstructionConfig(**data["reconstruction"]),
            patlak=PatlakConfig(**data["patlak"]),
            output=OutputConfig(**data["output"])
        )
        config.validate()
        return config

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.batch.num_samples < 1:
            raise ValueError("batch.num_samples must be at least 1")
        if self.batch.start_index < 0:
            raise ValueError("batch.start_index cannot be negative")
        if self.anatomy.bed_z_size < 1:
            raise ValueError("anatomy.bed_z_size must be positive")
        _require_len3("anatomy.voxel_cm_xyz", self.anatomy.voxel_cm_xyz, positive=True)
        _require_len3(
            "anatomy.max_initial_translation_cm_xyz",
            self.anatomy.max_initial_translation_cm_xyz,
            nonnegative=True
        )
        _require_len3(
            "anatomy.max_initial_rotation_deg_xyz",
            self.anatomy.max_initial_rotation_deg_xyz,
            nonnegative=True
        )
        _require_range("anatomy.initial_scale_range", self.anatomy.initial_scale_range, positive=True)
        if self.frames.n_frames < 2:
            raise ValueError("frames.n_frames must be at least 2")
        if self.frames.duration_seconds <= 0 or self.frames.points_per_min < 2:
            raise ValueError("frame duration and points_per_min must be positive")
        available = self.frames.end_scan_seconds - self.frames.start_scan_seconds
        required = self.frames.n_frames * self.frames.duration_seconds
        if available <= 0 or required > available:
            raise ValueError("scan interval cannot contain all requested frames")
        if not 1 <= self.motion.min_events <= self.motion.max_events:
            raise ValueError("motion requires 1 <= min_events <= max_events")
        if self.motion.max_events > self.frames.n_frames - 1:
            raise ValueError("motion.max_events cannot exceed n_frames - 1")
        _require_len3("motion.max_translation_cm_xyz", self.motion.max_translation_cm_xyz, nonnegative=True)
        _require_len3("motion.max_rotation_deg_xyz", self.motion.max_rotation_deg_xyz, nonnegative=True)
        _require_len3("lesion.base_radius_xyz", self.lesion.base_radius_xyz, positive=True)
        _require_range("lesion.radius_multiplier_range", self.lesion.radius_multiplier_range, positive=True)
        required_parameters = {"k1", "k2", "k3", "k4", "Vb"}
        if set(self.lesion.base_parameters) != required_parameters:
            raise ValueError("lesion.base_parameters must contain k1, k2, k3, k4, Vb")
        if set(self.lesion.parameter_multiplier_ranges) != required_parameters:
            raise ValueError("lesion.parameter_multiplier_ranges must contain k1, k2, k3, k4, Vb")
        for name, limits in self.lesion.parameter_multiplier_ranges.items():
            _require_range(f"lesion.parameter_multiplier_ranges.{name}", limits, nonnegative=True)
        if self.lesion.attenuation_class not in self.attenuation.mu_cm:
            raise ValueError("lesion attenuation class is absent from attenuation.mu_cm")
        if self.attenuation.unknown_nonzero_default not in self.attenuation.mu_cm:
            raise ValueError("unknown attenuation class is absent from attenuation.mu_cm")
        if self.projection.background_fraction < 0:
            raise ValueError("projection.background_fraction cannot be negative")
        if self.projection.noise_scaling_factor <= 0:
            raise ValueError("projection.noise_scaling_factor must be positive")
        if self.reconstruction.num_subsets < 1 or self.reconstruction.num_subiterations < 1:
            raise ValueError("reconstruction iteration counts must be positive")
        if not self.patlak.start_min < self.patlak.end_min:
            raise ValueError("patlak.start_min must be smaller than patlak.end_min")


def deep_merge(base, update):
    """Recursively merge a user config into the complete default config."""
    merged = deepcopy(base)
    for key, value in update.items():
        if key not in merged:
            raise KeyError(f"unknown configuration key: {key}")
        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path, overrides):
    raw: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    if overrides:
        raw = deepcopy(raw)
        for expression in overrides:
            apply_override(raw, expression)
    return SimulationConfig.from_dict(raw)


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )


def apply_override(raw, expression):
    """Apply ``section.key=JSON_VALUE`` to a partially specified config."""
    if "=" not in expression:
        raise ValueError(f"invalid --set value: {expression!r}")
    dotted_key, text_value = expression.split("=", 1)
    keys = [part for part in dotted_key.split(".") if part]
    if not keys:
        raise ValueError(f"invalid --set key: {expression!r}")
    try:
        value = json.loads(text_value)
    except json.JSONDecodeError:
        value = text_value
    target = raw
    for key in keys[:-1]:
        existing = target.get(key)
        if existing is None:
            target[key] = {}
        elif not isinstance(existing, dict):
            raise ValueError(f"cannot set nested key below {key!r}")
        target = target[key]
    target[keys[-1]] = value


def sample_seed(master_seed, sample_index):
    """Derive a stable per-sample seed, independent of batch slicing."""
    import numpy as np

    sequence = np.random.SeedSequence([int(master_seed), int(sample_index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _require_len3(name, values, *, positive=False, nonnegative=False) -> None:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if positive and any(float(value) <= 0 for value in values):
        raise ValueError(f"{name} values must be positive")
    if nonnegative and any(float(value) < 0 for value in values):
        raise ValueError(f"{name} values cannot be negative")


def _require_range(name, values, *, positive=False, nonnegative=False) -> None:
    if len(values) != 2 or float(values[0]) > float(values[1]):
        raise ValueError(f"{name} must be [minimum, maximum]")
    if positive and float(values[0]) <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and float(values[0]) < 0:
        raise ValueError(f"{name} cannot be negative")
