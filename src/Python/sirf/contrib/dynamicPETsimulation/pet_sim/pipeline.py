# SPDX-License-Identifier: Apache-2.0

"""One-sample pipeline orchestration with explicit stage boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
from pathlib import Path

import h5py
import numpy as np

from .anatomy import AnatomySample, LesionSample, generate_anatomy, make_attenuation_map, sample_lesion
from .config import SimulationConfig
from .inputs import InputData
from .io_utils import recreate_h5_dataset, sample_stage, utc_now, write_h5_attrs, write_h5_dataset, write_json_atomic
from .kinetics import KineticsSample, generate_kinetics
from .motion import apply_motion_to_h5, generate_motion_plan, save_motion_metadata
from .patlak import run_patlak
from .sirf_ops import cleanup_noisy_projections, forward_project_and_add_noise, initialise_sirf, reconstruct_dynamic


@dataclass(frozen=True)
class SamplePaths:
    sample_dir: Path
    project_h5: Path
    noisy_projection_dir: Path
    reconstruction_interfile_dir: Path
    status_json: Path
    config_json: Path
    log_file: Path

    @classmethod
    def create(cls, sample_dir, h5_name):
        return cls(
            sample_dir=sample_dir,
            project_h5=sample_dir / h5_name,
            noisy_projection_dir=sample_dir / "sirf_noisy_projection",
            reconstruction_interfile_dir=sample_dir / "sirf_osem_reconstruction",
            status_json=sample_dir / "status.json",
            config_json=sample_dir / "sample_config.json",
            log_file=sample_dir / "run.log"
        )


def run_sample(inputs, config, sample_index, seed, paths, logger):
    paths.sample_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "state": "running",
        "sample_index": sample_index,
        "seed": seed,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "stage": "initialising"
    }
    write_json_atomic(paths.status_json, status)
    write_json_atomic(
        paths.config_json,
        {
            "sample_index": sample_index,
            "sample_seed": seed,
            "simulation_config": config.to_dict()
        }
    )

    rng = np.random.default_rng(seed)

    with sample_stage(paths.status_json, status, "anatomy"):
        lesion = sample_lesion(config.lesion, rng)
        anatomy = generate_anatomy(inputs.label_full_xyz, config.anatomy, config.lesion, lesion, rng, logger)
        attenuation_xyz = make_attenuation_map(anatomy.label_bed_xyz, anatomy.lesion_mask_xyz, config.attenuation,
                                               config.lesion, logger)

    with sample_stage(paths.status_json, status, "kinetics"):
        kinetics = generate_kinetics(anatomy.label_bed_xyz, config.anatomy.organ_id, inputs.regions, lesion.parameters,
                                     inputs.aif_time_min, inputs.aif_cp, config.frames, config.kinetics, logger)

    with sample_stage(paths.status_json, status, "source_hdf5"):
        write_source_h5(paths.project_h5, config, sample_index, seed, anatomy, lesion, kinetics, attenuation_xyz,
                        inputs, logger)
        del attenuation_xyz
        gc.collect()

    # (x,y,z) -> (z,y,x)
    source_shape_zyx = tuple(int(value) for value in anatomy.label_bed_xyz.shape[::-1])
    with sample_stage(paths.status_json, status, "motion"):
        motion_plan = generate_motion_plan(kinetics.n_frames, source_shape_zyx, config.anatomy.voxel_cm_xyz,
                                           config.motion, rng)
        save_motion_metadata(paths.project_h5, motion_plan, config.motion,
                             initial_translation_cm_xyz=anatomy.initial_translation_cm_xyz,
                             initial_scale_factor=anatomy.initial_scale_factor,
                             initial_rotation_deg_xyz=anatomy.initial_rotation_deg_xyz,
                             initial_scipy_matrix_zyx=anatomy.initial_scipy_matrix_zyx)
        apply_motion_to_h5(paths.project_h5, motion_plan, compression=config.output.compression, logger=logger)
        del motion_plan
        gc.collect()

    with sample_stage(paths.status_json, status, "sirf_geometry"):
        sirf_context = initialise_sirf(paths.project_h5, config.scanner, config.anatomy.voxel_cm_xyz)

    noise_seed_base = int(rng.integers(1, 2 ** 31 - config.frames.n_frames - 1))
    with sample_stage(paths.status_json, status, "forward_projection_and_noise"):
        projection_result = forward_project_and_add_noise(paths.project_h5, sirf_context, config.projection,
                                                          paths.noisy_projection_dir, noise_seed_base, logger)

    with sample_stage(paths.status_json, status, "reconstruction"):
        reconstruct_dynamic(paths.project_h5, sirf_context, projection_result, config.projection, config.reconstruction,
                            paths.reconstruction_interfile_dir, compression=config.output.compression, logger=logger)

    with sample_stage(paths.status_json, status, "patlak_and_unet"):
        run_patlak(
            paths.project_h5,
            sirf_context,
            config.patlak,
            inputs.aif_time_min,
            inputs.aif_cp,
            config.frames.points_per_min,
            lesion.parameters,
            compression=config.output.compression,
            logger=logger
        )

    cleanup_noisy_projections(
        paths.noisy_projection_dir,
        config.projection.save_noisy_projections
    )
    status.update(
        {
            "state": "completed",
            "stage": "completed",
            "updated_at": utc_now(),
            "completed_at": utc_now(),
            "project_h5": str(paths.project_h5),
            "lesion_parameters": lesion.parameters,
            "lesion_radius_xyz": lesion.radius_xyz
        }
    )
    write_json_atomic(paths.status_json, status)
    return status


def write_source_h5(project_h5, config, sample_index, seed, anatomy, lesion, kinetics, attenuation_xyz, inputs, logger):
    if project_h5.exists():
        project_h5.unlink()
    label_zyx = np.transpose(anatomy.label_bed_xyz, (2, 1, 0)).astype(np.int32)
    organ_mask_zyx = np.transpose(anatomy.organ_mask_xyz, (2, 1, 0)).astype(np.uint8)
    lesion_mask_zyx = np.transpose(anatomy.lesion_mask_xyz, (2, 1, 0)).astype(np.uint8)
    attenuation_zyx = np.transpose(attenuation_xyz, (2, 1, 0)).astype(np.float32)
    dynamic_shape = (kinetics.n_frames,) + label_zyx.shape
    with h5py.File(project_h5, "w") as handle:
        emission_dataset = recreate_h5_dataset(
            handle,
            "gt/emi_clean_tzyx",
            dynamic_shape,
            np.float32,
            compression=config.output.compression
        )
        for frame in range(kinetics.n_frames):
            emission_xyz = kinetics.emission_frame_xyz(
                anatomy.label_bed_xyz,
                anatomy.lesion_mask_xyz,
                frame
            )
            emission_dataset[frame] = np.transpose(emission_xyz, (2, 1, 0))
            logger.info("clean emission frame %d/%d", frame + 1, kinetics.n_frames)
            del emission_xyz
        handle.flush()

    datasets = {
        "gt/atn_static_zyx": attenuation_zyx,
        "gt/label_bed_zyx": label_zyx,
        "masks/organ_mask_zyx": organ_mask_zyx,
        "masks/lesion_mask_zyx": lesion_mask_zyx,
        "aif/time_min": inputs.aif_time_min,
        "aif/cp": inputs.aif_cp,
        "frames/start_min": kinetics.frame_start_min,
        "frames/end_min": kinetics.frame_end_min,
        "frames/mid_min": kinetics.frame_mid_min,
        "frames/duration_seconds": np.full(kinetics.n_frames, config.frames.duration_seconds, dtype=np.float32),
        "tac/label_ids": kinetics.label_ids,
        "tac/frame_by_label": kinetics.tac_frame_by_label,
        "tac/kinetic_source_label_ids": kinetics.kinetic_source_label_ids,
        "tac/kinetic_parameters_by_label": kinetics.kinetic_parameters_by_label,
        "tac/organ_frame": kinetics.organ_tac,
        "tac/lesion_frame": kinetics.lesion_tac,
        "tac/organ_continuous_time_min": kinetics.organ_continuous_time_min,
        "tac/organ_continuous_activity": kinetics.organ_continuous_activity,
        "tac/lesion_continuous_time_min": kinetics.lesion_continuous_time_min,
        "tac/lesion_continuous_activity": kinetics.lesion_continuous_activity,
        "tac/labels_without_own_kinetics": kinetics.labels_without_own_kinetics,
        "tac/labels_mapped_to_replacement": kinetics.labels_mapped_to_replacement,
        "tac/unfilled_nonzero_labels": kinetics.unfilled_nonzero_labels,
        "metadata/crop_start_xyz": anatomy.crop_start_xyz,
        "metadata/crop_end_xyz": anatomy.crop_end_xyz,
        "metadata/lesion_center_xyz": np.asarray(anatomy.lesion_center_xyz),
        "metadata/lesion_radius_xyz": lesion.radius_xyz,
        "metadata/source_voxel_mm_zyx": np.asarray(config.anatomy.voxel_cm_xyz[::-1], dtype=np.float32) * 10.0
    }
    uncompressed = {
        "tac/label_ids",
        "tac/kinetic_source_label_ids",
        "metadata/crop_start_xyz",
        "metadata/crop_end_xyz",
        "metadata/lesion_center_xyz",
        "metadata/lesion_radius_xyz",
        "metadata/source_voxel_mm_zyx"
    }
    for name, value in datasets.items():
        write_h5_dataset(
            project_h5,
            name,
            value,
            compression=config.output.compression and name not in uncompressed
        )
    write_h5_attrs(
        project_h5,
        "metadata",
        {
            "pipeline_version": "batch-v1",
            "source_notebook": "data_simulation_v13.ipynb",
            "sample_index": sample_index,
            "sample_seed": seed,
            "target_organ": config.anatomy.target_organ,
            "organ_id": config.anatomy.organ_id,
            "bed_z_size": config.anatomy.bed_z_size,
            "source_data_order": "z,y,x",
            "dynamic_data_order": "t,z,y,x",
            "kinetic_parameter_order": ["k1", "k2", "k3", "k4", "Vb"],
            "lesion_parameters": lesion.parameters,
            "lesion_attenuation_class": config.lesion.attenuation_class,
            "missing_kinetic_replacement_label": kinetics.replacement_label,
            "label_first_anatomy": True
        }
    )
