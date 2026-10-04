"""Persistent rigid-motion events represented by composed SciPy affine matrices."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform

from .config import MotionConfig


@dataclass
class MotionPlan:
    selected_frame_indices: np.ndarray
    event_translation_cm_xyz: np.ndarray
    event_rotation_deg_xyz: np.ndarray
    cumulative_translation_cm_xyz: np.ndarray
    cumulative_rotation_deg_xyz: np.ndarray
    event_scipy_matrix_t44: np.ndarray
    cumulative_scipy_matrix_t44: np.ndarray


def generate_motion_plan(n_frames, image_shape_zyx, voxel_cm_xyz, config, rng):
    candidate_frames = np.arange(1, n_frames, dtype=np.int32)
    event_count = int(rng.integers(config.min_events, config.max_events + 1))
    selected = np.sort(rng.choice(candidate_frames, size=event_count, replace=False)).astype(np.int32)
    event_translation = np.zeros((n_frames, 3), dtype=np.float32)
    event_rotation = np.zeros((n_frames, 3), dtype=np.float32)
    max_translation = np.asarray(config.max_translation_cm_xyz, dtype=np.float64)
    max_rotation = np.asarray(config.max_rotation_deg_xyz, dtype=np.float64)
    event_translation[selected] = rng.uniform(-max_translation, max_translation, size=(event_count, 3)).astype(
        np.float32)
    event_rotation[selected] = rng.uniform(-max_rotation, max_rotation, size=(event_count, 3)).astype(np.float32)

    event_matrices = np.zeros((n_frames, 4, 4), dtype=np.float64)
    cumulative_matrices = np.zeros_like(event_matrices)
    current = np.eye(4, dtype=np.float64)
    for frame in range(n_frames):
        translation_vox_zyx = translation_cm_xyz_to_vox_zyx(event_translation[frame], voxel_cm_xyz)
        event_matrices[frame] = make_scipy_affine_matrix_zyx(
            image_shape_zyx,
            translation_vox_zyx,
            event_rotation[frame],
            scale_factor=1.0
        )
        # SciPy matrices are output-to-input maps. This is the v13 composition rule.
        current = current @ event_matrices[frame]
        cumulative_matrices[frame] = current

    cumulative_translation = np.zeros((n_frames, 3), dtype=np.float32)
    cumulative_rotation = np.zeros((n_frames, 3), dtype=np.float32)
    for frame in range(n_frames):
        cumulative_translation[frame], cumulative_rotation[frame] = (
            motion_parameters_from_scipy_matrix_zyx(
                cumulative_matrices[frame], image_shape_zyx, voxel_cm_xyz
            )
        )
    return MotionPlan(
        selected_frame_indices=selected,
        event_translation_cm_xyz=event_translation,
        event_rotation_deg_xyz=event_rotation,
        cumulative_translation_cm_xyz=cumulative_translation,
        cumulative_rotation_deg_xyz=cumulative_rotation,
        event_scipy_matrix_t44=event_matrices,
        cumulative_scipy_matrix_t44=cumulative_matrices
    )


def apply_motion_to_h5(project_h5, plan, *, compression, logger):
    import h5py

    from .io_utils import recreate_h5_dataset, validate_finite

    with h5py.File(project_h5, "a") as handle:
        clean = handle["gt/emi_clean_tzyx"]
        attenuation = handle["gt/atn_static_zyx"][:].astype(np.float32)
        moved_emission = recreate_h5_dataset(
            handle,
            "motion/emi_motion_tzyx",
            clean.shape,
            np.float32,
            compression=compression
        )
        moved_attenuation = recreate_h5_dataset(
            handle,
            "motion/atn_motion_tzyx",
            clean.shape,
            np.float32,
            compression=compression
        )
        for frame in range(clean.shape[0]):
            matrix = plan.cumulative_scipy_matrix_t44[frame]
            emission_frame = apply_scipy_affine_matrix_zyx(clean[frame].astype(np.float32), matrix, order=1).astype(
                np.float32)
            attenuation_frame = apply_scipy_affine_matrix_zyx(attenuation, matrix, order=1).astype(np.float32)
            validate_finite(f"motion emission frame {frame}", emission_frame)
            validate_finite(f"motion attenuation frame {frame}", attenuation_frame)
            moved_emission[frame] = emission_frame
            moved_attenuation[frame] = attenuation_frame
            handle.flush()
            logger.info("motion frame %d/%d", frame + 1, clean.shape[0])
            del emission_frame, attenuation_frame
            gc.collect()


def save_motion_metadata(project_h5, plan, config, *, initial_translation_cm_xyz, initial_scale_factor,
                         initial_rotation_deg_xyz, initial_scipy_matrix_zyx):
    from .io_utils import write_h5_attrs, write_h5_dataset

    datasets = {
        "initial_translation_cm_xyz": initial_translation_cm_xyz,
        "initial_scale_factor": np.asarray(initial_scale_factor),
        "initial_rotation_deg_xyz": initial_rotation_deg_xyz,
        "initial_label_scipy_matrix_zyx": initial_scipy_matrix_zyx,
        "selected_frame_indices": plan.selected_frame_indices,
        "event_translation_cm_xyz": plan.event_translation_cm_xyz,
        "event_rotation_deg_xyz": plan.event_rotation_deg_xyz,
        "translation_cm_xyz": plan.cumulative_translation_cm_xyz,
        "rotation_deg_xyz": plan.cumulative_rotation_deg_xyz,
        "event_scipy_matrix_t44": plan.event_scipy_matrix_t44,
        "cumulative_scipy_matrix_t44": plan.cumulative_scipy_matrix_t44
    }
    for name, value in datasets.items():
        write_h5_dataset(
            project_h5,
            f"motion/{name}",
            value,
            compression=value.ndim > 2 if isinstance(value, np.ndarray) else False
        )
    write_h5_attrs(
        project_h5,
        "motion",
        {
            "motion_model": "label-first initial anatomy augmentation; persistent incremental rigid-motion events",
            "selected_frame_meaning": "frame where a new motion event occurs; new pose persists",
            "first_frame_has_motion_event": False,
            "minimum_number_of_random_motion_frames": config.min_events,
            "maximum_number_of_random_motion_frames": config.max_events,
            "actual_number_of_random_motion_frames": len(plan.selected_frame_indices),
            "event_matrix_composition": "current_cumulative_matrix @ new_event_matrix",
            "scipy_matrix_convention": "output-to-input backward mapping",
            "image_interpolation": "one linear interpolation from motion-free GT using the composed cumulative matrix",
            "event_motion_has_scaling": False
        }
    )


def rotation_matrix_zyx_from_xyz_degrees(rotation_deg_xyz):
    rx, ry, rz = np.deg2rad(rotation_deg_xyz)
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])
    rotation_x = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rotation_y = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rotation_z = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    rotation_xyz = rotation_z @ rotation_y @ rotation_x
    permutation = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
    return permutation.T @ rotation_xyz @ permutation


def translation_cm_xyz_to_vox_zyx(translation_cm_xyz, voxel_cm_xyz):
    tx, ty, tz = np.asarray(translation_cm_xyz, dtype=np.float64)
    vx, vy, vz = np.asarray(voxel_cm_xyz, dtype=np.float64)
    return np.asarray([tz / vz, ty / vy, tx / vx], dtype=np.float64)


def make_scipy_affine_matrix_zyx(image_shape_zyx, translation_vox_zyx, rotation_deg_xyz, *, scale_factor):
    rotation = rotation_matrix_zyx_from_xyz_degrees(rotation_deg_xyz)
    inverse_linear = np.linalg.inv(float(scale_factor) * rotation)
    center = (np.asarray(image_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    offset = center - inverse_linear @ (center + translation_vox_zyx)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = inverse_linear
    matrix[:3, 3] = offset
    return matrix


def apply_scipy_affine_matrix_zyx(image, scipy_matrix, *, order, cval=0.0):
    image = np.asarray(image)
    return affine_transform(
        image,
        matrix=np.asarray(scipy_matrix, dtype=np.float64),
        output_shape=image.shape,
        order=order,
        mode="constant",
        cval=cval,
        prefilter=False
    )


def motion_parameters_from_scipy_matrix_zyx(scipy_matrix, image_shape_zyx, voxel_cm_xyz):
    forward = np.linalg.inv(np.asarray(scipy_matrix, dtype=np.float64))
    center_zyx = (np.asarray(image_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    moved_center = (forward @ np.append(center_zyx, 1.0))[:3]
    translation_vox_zyx = moved_center - center_zyx
    vx, vy, vz = np.asarray(voxel_cm_xyz, dtype=np.float64)
    translation_cm_xyz = np.asarray([translation_vox_zyx[2] * vx, translation_vox_zyx[1] * vy, translation_vox_zyx[0] * vz],dtype=np.float32)
    permutation = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
    rotation_xyz = permutation @ forward[:3, :3] @ permutation.T
    ry = np.arcsin(np.clip(-rotation_xyz[2, 0], -1.0, 1.0))
    if abs(np.cos(ry)) > 1e-8:
        rx = np.arctan2(rotation_xyz[2, 1], rotation_xyz[2, 2])
        rz = np.arctan2(rotation_xyz[1, 0], rotation_xyz[0, 0])
    else:
        rx = np.arctan2(-rotation_xyz[1, 2], rotation_xyz[1, 1])
        rz = 0.0
    return translation_cm_xyz, np.rad2deg([rx, ry, rz]).astype(np.float32)
