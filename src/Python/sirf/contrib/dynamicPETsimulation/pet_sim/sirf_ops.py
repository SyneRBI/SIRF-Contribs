# SPDX-License-Identifier: Apache-2.0

"""SIRF geometry, forward projection, Poisson noise, and OSEM reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import importlib
import logging
from pathlib import Path
import shutil
from types import ModuleType

import h5py
import numpy as np

from .config import (
    ProjectionConfig,
    ReconstructionConfig,
    ScannerConfig,
)
from .io_utils import (
    recreate_h5_dataset,
    validate_finite,
    write_h5_attrs,
    write_h5_dataset,
)


@dataclass
class SirfContext:
    pet: ModuleType
    acquisition_template: object
    source_template: object
    target_template: object
    source_shape_zyx: tuple[int, int, int]
    target_shape_zyx: tuple[int, int, int]
    source_voxel_mm_zyx: np.ndarray
    target_voxel_mm_zyx: np.ndarray


@dataclass
class ProjectionResult:
    noisy_paths: list[Path]
    emission_only_sums: np.ndarray
    clean_sums: np.ndarray
    noisy_sums: np.ndarray
    background_values: np.ndarray
    noise_seed_base: int


def initialise_sirf(project_h5, scanner, voxel_cm_xyz):
    try:
        pet = importlib.import_module("sirf.STIR")
    except ImportError as error:
        raise RuntimeError(
            "SIRF/STIR is required for projection and reconstruction. Run this project inside a configured SIRF environment."
        ) from error
    acquisition_template = pet.AcquisitionData(
        scanner.name,
        span=scanner.span,
        max_ring_diff=scanner.max_ring_diff
    )
    target_template = acquisition_template.create_uniform_image(0.0)
    target_shape = tuple(int(value) for value in target_template.as_array().shape)
    target_voxel = np.asarray(target_template.voxel_sizes(), dtype=np.float32)
    with h5py.File(project_h5, "r") as handle:
        source_shape = tuple(int(value) for value in handle["motion/emi_motion_tzyx"].shape[1:])
    voxel_x, voxel_y, voxel_z = voxel_cm_xyz
    source_voxel = np.asarray([voxel_z * 10.0, voxel_y * 10.0, voxel_x * 10.0], dtype=np.float32)
    source_template = pet.ImageData()
    source_template.initialise(dim=source_shape, vsize=tuple(float(value) for value in source_voxel))
    context = SirfContext(
        pet=pet,
        acquisition_template=acquisition_template,
        source_template=source_template,
        target_template=target_template,
        source_shape_zyx=source_shape,
        target_shape_zyx=target_shape,
        source_voxel_mm_zyx=source_voxel,
        target_voxel_mm_zyx=target_voxel
    )
    write_h5_dataset(project_h5, "sirf/source_shape_zyx", source_shape, compression=False, dtype=np.int32)
    write_h5_dataset(project_h5, "sirf/target_shape_zyx", target_shape, compression=False, dtype=np.int32)
    write_h5_dataset(project_h5, "sirf/source_voxel_mm_zyx", source_voxel, compression=False)
    write_h5_dataset(project_h5, "sirf/target_voxel_mm_zyx", target_voxel, compression=False)
    write_h5_attrs(
        project_h5,
        "sirf",
        {
            "scanner_name": scanner.name,
            "span": scanner.span,
            "max_ring_diff": scanner.max_ring_diff,
            "source_order": "z,y,x",
            "target_order": "z,y,x",
        }
    )
    return context


def forward_project_and_add_noise(project_h5, context, projection_config, noisy_dir, noise_seed_base, logger):
    shutil.rmtree(noisy_dir, ignore_errors=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(project_h5, "r") as handle:
        n_frames = int(handle["motion/emi_motion_tzyx"].shape[0])
    noise_generator = context.pet.PoissonNoiseGenerator(
        scaling_factor=projection_config.noise_scaling_factor,
        preserve_mean=True
    )
    noisy_paths: list[Path] = []
    emission_sums: list[float] = []
    clean_sums: list[float] = []
    noisy_sums: list[float] = []
    background_values: list[float] = []
    for frame in range(n_frames):
        logger.info("forward/noise frame %d/%d", frame + 1, n_frames)
        emission_raw, attenuation_raw, emission_target, attenuation_target = load_motion_target_images(project_h5,
                                                                                                       context, frame)
        emission_target_array = emission_target.as_array().astype(np.float32)
        attenuation_target_array = attenuation_target.as_array().astype(np.float32)
        _validate_target("emission", emission_target_array, context.target_shape_zyx, positive=True)
        _validate_target("attenuation", attenuation_target_array, context.target_shape_zyx)
        acquisition_model = build_acquisition_model_with_attenuation(context, attenuation_target, emission_target)
        emission_projection = acquisition_model.forward(emission_target)
        emission_array = emission_projection.as_array()
        validate_finite("emission projection", emission_array, require_positive=True)
        if np.any(emission_array < -1e-6):
            raise RuntimeError("emission projection contains negative values")
        background_value = float(
            emission_projection.max()) * projection_config.background_fraction if projection_config.add_background else 0.0

        background = emission_projection.get_uniform_copy(background_value)
        clean_projection = emission_projection.clone()
        clean_projection += background
        clean_array = clean_projection.as_array()
        validate_finite("clean projection", clean_array, require_positive=True)
        observed_background = clean_array - emission_array
        if not np.allclose(observed_background, background_value, rtol=1e-4, atol=1e-6):
            raise RuntimeError("uniform background was not added correctly")
        noise_generator.set_seed(int(noise_seed_base + frame))
        noisy_projection = noise_generator.generate_noisy_data(clean_projection)
        noisy_array = noisy_projection.as_array()
        validate_finite("noisy projection", noisy_array, require_positive=True)
        if np.any(noisy_array < -1e-6):
            raise RuntimeError("noisy projection contains negative values")
        noisy_path = noisy_dir / f"noisy_proj_frame_{frame + 1:03d}.hs"
        noisy_projection.write(str(noisy_path))
        noisy_paths.append(noisy_path)
        emission_sums.append(float(np.nansum(emission_array)))
        clean_sums.append(float(np.nansum(clean_array)))
        noisy_sums.append(float(np.nansum(noisy_array)))
        background_values.append(background_value)
        del (
            emission_raw,
            attenuation_raw,
            emission_target,
            attenuation_target,
            emission_target_array,
            attenuation_target_array,
            acquisition_model,
            emission_projection,
            emission_array,
            background,
            clean_projection,
            clean_array,
            observed_background,
            noisy_projection,
            noisy_array
        )
        gc.collect()

    result = ProjectionResult(
        noisy_paths=noisy_paths,
        emission_only_sums=np.asarray(emission_sums, dtype=np.float32),
        clean_sums=np.asarray(clean_sums, dtype=np.float32),
        noisy_sums=np.asarray(noisy_sums, dtype=np.float32),
        background_values=np.asarray(background_values, dtype=np.float32),
        noise_seed_base=int(noise_seed_base)
    )
    write_h5_dataset(project_h5, "projection/emission_only_sum", result.emission_only_sums)
    write_h5_dataset(project_h5, "projection/clean_sum", result.clean_sums)
    write_h5_dataset(project_h5, "projection/noisy_sum", result.noisy_sums)
    write_h5_dataset(project_h5, "projection/background_uniform_value", result.background_values)
    write_h5_dataset(
        project_h5,
        "projection/background_fraction",
        np.asarray([projection_config.background_fraction], dtype=np.float32),
        compression=False
    )
    write_h5_dataset(
        project_h5,
        "projection/noise_seed_base",
        np.asarray([noise_seed_base], dtype=np.int64),
        compression=False
    )
    write_h5_dataset(
        project_h5,
        "projection/noise_scaling_factor",
        np.asarray([projection_config.noise_scaling_factor], dtype=np.float32),
        compression=False
    )
    write_h5_attrs(
        project_h5,
        "projection",
        {
            "clean_projection_saved": False,
            "clean_projection_definition": "attenuated emission projection + uniform additive background",
            "forward_model": "AcquisitionModelUsingParallelproj with attenuation sensitivity",
            "add_background": projection_config.add_background,
            "background_model": "uniform background per frame",
            "background_value_rule": "max(emission-only projection) * background_fraction",
            "noise_model": "SIRF PoissonNoiseGenerator",
            "preserve_mean": True,
            "noise_seed_rule": "sample noise_seed_base + zero-based frame index",
            "noisy_projection_dir": str(noisy_dir)
        }
    )
    return result


def reconstruct_dynamic(project_h5, context, projection_result, projection_config, reconstruction_config,
                        recon_interfile_dir, *, compression, logger):
    n_frames = len(projection_result.noisy_paths)
    with h5py.File(project_h5, "r") as handle:
        attenuation_source_array = handle["gt/atn_static_zyx"][:].astype(np.float32)
    attenuation_source = make_sirf_image_from_zyx(attenuation_source_array, context.source_template)
    attenuation_target = zoom_to_template(attenuation_source, context.target_template, scaling="preserve_values")
    attenuation_target_array = attenuation_target.as_array().astype(np.float32)
    _validate_target("motion-free attenuation", attenuation_target_array, context.target_shape_zyx)
    reconstruction_shape = (n_frames,) + context.target_shape_zyx
    with h5py.File(project_h5, "a") as handle:
        recreate_h5_dataset(
            handle,
            "recon/osem_tzyx",
            reconstruction_shape,
            np.float32,
            compression=compression
        )
    if reconstruction_config.save_interfile:
        recon_interfile_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_sums: list[float] = []
    for frame, noisy_path in enumerate(projection_result.noisy_paths):
        logger.info("OSEM frame %d/%d", frame + 1, n_frames)
        noisy_projection = context.pet.AcquisitionData(str(noisy_path))
        acquisition_model = build_acquisition_model_with_attenuation(
            context,
            attenuation_target,
            context.target_template,
            acquisition_template=noisy_projection
        )
        background = None
        if projection_config.add_background:
            background = noisy_projection.get_uniform_copy(float(projection_result.background_values[frame]))
            acquisition_model.set_background_term(background)
        initial_image = context.target_template.get_uniform_copy(1.0)
        objective = context.pet.make_Poisson_loglikelihood(noisy_projection)
        objective.set_acquisition_model(acquisition_model)
        reconstructor = context.pet.OSMAPOSLReconstructor()
        reconstructor.set_objective_function(objective)
        reconstructor.set_num_subsets(reconstruction_config.num_subsets)
        reconstructor.set_num_subiterations(reconstruction_config.num_subiterations)
        reconstructor.set_up(initial_image)
        reconstructor.set_current_estimate(initial_image)
        reconstructor.process()
        reconstructed_image = reconstructor.get_output()
        reconstructed_array = reconstructed_image.as_array().astype(np.float32)
        _validate_target(f"OSEM frame {frame}", reconstructed_array, context.target_shape_zyx, positive=True)
        if np.any(reconstructed_array < -1e-6):
            raise RuntimeError(f"OSEM frame {frame} contains negative values")
        with h5py.File(project_h5, "a") as handle:
            handle["recon/osem_tzyx"][frame] = reconstructed_array
            handle.flush()
        reconstruction_sums.append(float(np.nansum(reconstructed_array)))
        if reconstruction_config.save_interfile:
            reconstructed_image.write(str(recon_interfile_dir / f"osem_recon_frame_{frame + 1:03d}.hv"))
        del (
            noisy_projection,
            acquisition_model,
            initial_image,
            objective,
            reconstructor,
            reconstructed_image,
            reconstructed_array
        )
        if background is not None:
            del background
        gc.collect()
    write_h5_dataset(
        project_h5,
        "recon/osem_sum",
        np.asarray(reconstruction_sums, dtype=np.float32),
    )
    write_h5_dataset(
        project_h5,
        "recon/attenuation_motion_free_target_zyx",
        attenuation_target_array
    )
    write_h5_attrs(
        project_h5,
        "recon",
        {
            "osem_saved": True,
            "osem_num_subsets": reconstruction_config.num_subsets,
            "osem_num_subiterations": reconstruction_config.num_subiterations,
            "osem_interfile_saved": reconstruction_config.save_interfile,
            "attenuation_used": "motion-free attenuation for all frames",
            "attenuation_source": "gt/atn_static_zyx",
            "background_used": projection_config.add_background,
            "background_source": "projection/background_uniform_value" if projection_config.add_background else "none"
        }
    )


def cleanup_noisy_projections(noisy_dir, save_noisy_projections):
    if not save_noisy_projections:
        shutil.rmtree(noisy_dir, ignore_errors=True)


def make_sirf_image_from_zyx(array_zyx, template_image):
    array_zyx = np.asarray(array_zyx, dtype=np.float32)
    expected_shape = template_image.as_array().shape
    if array_zyx.shape != expected_shape:
        raise ValueError(f"source array shape {array_zyx.shape} != template shape {expected_shape}")
    image = template_image.clone()
    image.fill(array_zyx)
    return image


def zoom_to_template(source_image, target_image, *, scaling):
    if hasattr(source_image, "zoom_image_as_template"):
        return source_image.zoom_image_as_template(target_image, scaling=scaling)
    if not hasattr(source_image, "zoom_image"):
        raise RuntimeError("SIRF ImageData lacks both zoom_image_as_template and zoom_image")
    source_voxel = np.asarray(source_image.voxel_sizes(), dtype=np.float32)
    target_voxel = np.asarray(target_image.voxel_sizes(), dtype=np.float32)
    zooms = tuple(float(value) for value in source_voxel / target_voxel)
    target_size = tuple(int(value) for value in target_image.as_array().shape)
    offsets = (0.0, 0.0, 0.0)
    try:
        return source_image.zoom_image(
            zooms=zooms,
            offsets_in_mm=offsets,
            size=target_size,
            scaling=scaling
        )
    except TypeError:
        return source_image.zoom_image(zooms, offsets, target_size, scaling)


def load_motion_target_images(project_h5, context, frame_index):
    with h5py.File(project_h5, "r") as handle:
        emission = handle["motion/emi_motion_tzyx"][frame_index].astype(np.float32)
        attenuation = handle["motion/atn_motion_tzyx"][frame_index].astype(np.float32)
    emission_source = make_sirf_image_from_zyx(emission, context.source_template)
    attenuation_source = make_sirf_image_from_zyx(attenuation, context.source_template)
    return (
        emission,
        attenuation,
        zoom_to_template(emission_source, context.target_template, scaling="preserve_values"),
        zoom_to_template(attenuation_source, context.target_template, scaling="preserve_values")
    )


def build_acquisition_model_with_attenuation(context, attenuation_target, image_template, acquisition_template=None):
    if acquisition_template is None:
        acquisition_template = context.acquisition_template
    attenuation_model = context.pet.AcquisitionModelUsingParallelproj()
    attenuation_model.set_up(acquisition_template, attenuation_target)
    sensitivity = context.pet.AcquisitionSensitivityModel(attenuation_target, attenuation_model)
    sensitivity.set_up(acquisition_template)
    acquisition_model = context.pet.AcquisitionModelUsingParallelproj()
    acquisition_model.set_acquisition_sensitivity(sensitivity)
    acquisition_model.set_up(acquisition_template, image_template)
    return acquisition_model


def _validate_target(name, array, expected_shape, *, positive=False):
    if array.shape != expected_shape:
        raise RuntimeError(f"{name} shape {array.shape} != expected {expected_shape}")
    validate_finite(name, array, require_positive=positive)
