# SPDX-License-Identifier: Apache-2.0

"""Label-first anatomy augmentation, lesion placement, and attenuation maps."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from scipy.ndimage import affine_transform, distance_transform_edt

from .config import AnatomyConfig, AttenuationConfig, LesionConfig


@dataclass
class LesionSample:
    parameters: dict[str, float]
    radius_xyz: np.ndarray


@dataclass
class AnatomySample:
    label_bed_xyz: np.ndarray
    organ_mask_xyz: np.ndarray
    lesion_mask_xyz: np.ndarray
    lesion_center_xyz: tuple[int, int, int]
    crop_start_xyz: np.ndarray
    crop_end_xyz: np.ndarray
    initial_translation_cm_xyz: np.ndarray
    initial_scale_factor: float
    initial_rotation_deg_xyz: np.ndarray
    initial_scipy_matrix_zyx: np.ndarray


def sample_lesion(config, rng):
    parameters: dict[str, float] = {}
    for name, base_value in config.base_parameters.items():
        low, high = config.parameter_multiplier_ranges[name]
        value = float(base_value) * float(rng.uniform(low, high))
        if name == "Vb":
            value = float(np.clip(value, 0.0, 1.0))
        else:
            value = max(value, 0.0)
        parameters[name] = value
    radius_low, radius_high = config.radius_multiplier_range
    radius_xyz = np.asarray(config.base_radius_xyz, dtype=np.float64) * rng.uniform(radius_low, radius_high, size=3)
    return LesionSample(parameters=parameters, radius_xyz=radius_xyz)


def generate_anatomy(label_full_xyz, anatomy_config, lesion_config, lesion_sample, rng, logger):
    organ_mask_full = np.asarray(label_full_xyz) == anatomy_config.organ_id
    if not np.any(organ_mask_full):
        raise ValueError(f"organ_id={anatomy_config.organ_id} is absent from the XCAT label")

    x_idx, y_idx, z_idx = np.where(organ_mask_full)
    nx, ny, nz = label_full_xyz.shape
    z_center = int(round((int(z_idx.min()) + int(z_idx.max())) / 2.0))
    z_start, z_end = clamp_start(z_center, anatomy_config.bed_z_size, nz)
    crop_start_xyz = np.asarray([0, 0, z_start], dtype=np.int32)
    crop_end_xyz = np.asarray([nx, ny, z_end], dtype=np.int32)
    label_original = np.asarray(label_full_xyz[:, :, z_start:z_end], dtype=np.int16).copy()

    translation_cm_xyz = rng.uniform(-np.asarray(anatomy_config.max_initial_translation_cm_xyz, dtype=np.float64),
                                     np.asarray(anatomy_config.max_initial_translation_cm_xyz,
                                                dtype=np.float64)).astype(np.float32)
    scale_factor = float(rng.uniform(*anatomy_config.initial_scale_range))
    rotation_deg_xyz = rng.uniform(
        -np.asarray(anatomy_config.max_initial_rotation_deg_xyz, dtype=np.float64),
        np.asarray(anatomy_config.max_initial_rotation_deg_xyz, dtype=np.float64)
    ).astype(np.float32)

    voxel_cm_xyz = np.asarray(anatomy_config.voxel_cm_xyz, dtype=np.float64)
    translation_vox_xyz = translation_cm_xyz / voxel_cm_xyz
    rotation_xyz = rotation_matrix_xyz(rotation_deg_xyz)
    forward_linear_xyz = scale_factor * rotation_xyz
    inverse_linear_xyz = np.linalg.inv(forward_linear_xyz)
    center_xyz = (np.asarray(label_original.shape, dtype=np.float64) - 1.0) / 2.0
    offset_xyz = center_xyz - inverse_linear_xyz @ (center_xyz + translation_vox_xyz)
    scipy_matrix_xyz = np.eye(4, dtype=np.float64)
    scipy_matrix_xyz[:3, :3] = inverse_linear_xyz
    scipy_matrix_xyz[:3, 3] = offset_xyz

    label_bed_xyz = affine_transform(
        label_original,
        matrix=scipy_matrix_xyz,
        output_shape=label_original.shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False
    ).astype(np.int16)
    organ_mask_xyz = label_bed_xyz == anatomy_config.organ_id
    if not np.any(organ_mask_xyz):
        raise RuntimeError("initial anatomy transform moved the target organ completely outside the FOV")

    lesion_mask_xyz, lesion_center_xyz = make_random_ellipsoid_lesion_inside_organ(
        organ_mask_xyz,
        lesion_sample.radius_xyz,
        rng,
        min_distance_margin=lesion_config.min_distance_margin_voxels,
        logger=logger
    )

    xyz_from_zyx = np.asarray([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    scipy_matrix_zyx = xyz_from_zyx.T @ scipy_matrix_xyz @ xyz_from_zyx

    logger.info(
        "initial anatomy: translation_cm_xyz=%s scale=%.6f rotation_deg_xyz=%s",
        translation_cm_xyz.tolist(),
        scale_factor,
        rotation_deg_xyz.tolist()
    )
    return AnatomySample(
        label_bed_xyz=label_bed_xyz,
        organ_mask_xyz=organ_mask_xyz,
        lesion_mask_xyz=lesion_mask_xyz,
        lesion_center_xyz=lesion_center_xyz,
        crop_start_xyz=crop_start_xyz,
        crop_end_xyz=crop_end_xyz,
        initial_translation_cm_xyz=translation_cm_xyz,
        initial_scale_factor=scale_factor,
        initial_rotation_deg_xyz=rotation_deg_xyz,
        initial_scipy_matrix_zyx=scipy_matrix_zyx
    )


def make_attenuation_map(label_xyz, lesion_mask_xyz, attenuation_config, lesion_config, logger):
    label_xyz = np.asarray(label_xyz)
    attenuation = np.zeros(label_xyz.shape, dtype=np.float32)
    missing: list[int] = []
    for label in np.unique(label_xyz).astype(int):
        if label == 0:
            class_name = "Air"
        elif label in attenuation_config.roi_classes:
            class_name = attenuation_config.roi_classes[label]
        else:
            class_name = attenuation_config.unknown_nonzero_default
            missing.append(label)
        if class_name not in attenuation_config.mu_cm:
            raise ValueError(f"attenuation class {class_name!r} has no coefficient")
        attenuation[label_xyz == label] = attenuation_config.mu_cm[class_name]
    attenuation[np.asarray(lesion_mask_xyz, dtype=bool)] = attenuation_config.mu_cm[lesion_config.attenuation_class]
    if missing:
        logger.warning(
            "labels without explicit attenuation class use %s: %s",
            attenuation_config.unknown_nonzero_default,
            missing
        )
    return attenuation


def make_random_ellipsoid_lesion_inside_organ(organ_mask_xyz, radius_xyz, rng, *, min_distance_margin, logger):
    organ_mask_xyz = np.asarray(organ_mask_xyz, dtype=bool)
    rx, ry, rz = np.asarray(radius_xyz, dtype=np.float64)
    distance = distance_transform_edt(organ_mask_xyz)
    valid_center_mask = distance >= max(rx, ry, rz) + float(min_distance_margin)
    if not np.any(valid_center_mask):
        logger.warning("organ is too small to guarantee the full lesion ellipsoid; clipping to organ")
        valid_center_mask = organ_mask_xyz
    centers = np.argwhere(valid_center_mask)
    center = centers[int(rng.integers(0, centers.shape[0]))]
    cx, cy, cz = (int(value) for value in center)
    nx, ny, nz = organ_mask_xyz.shape
    x_grid, y_grid, z_grid = np.ogrid[:nx, :ny, :nz]
    lesion = ((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2 + ((z_grid - cz) / rz) ** 2 <= 1.0
    lesion &= organ_mask_xyz
    if not np.any(lesion):
        raise RuntimeError("generated lesion is empty")
    return lesion, (cx, cy, cz)


def rotation_matrix_xyz(rotation_deg_xyz):
    rx, ry, rz = np.deg2rad(rotation_deg_xyz)
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])
    rotation_x = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rotation_y = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rotation_z = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rotation_z @ rotation_y @ rotation_x


def clamp_start(center, crop_size, full_size):
    if crop_size > full_size:
        raise ValueError("crop size cannot exceed full size")
    start = int(round(center - crop_size / 2.0))
    start = max(0, min(start, full_size - crop_size))
    return start, start + crop_size
