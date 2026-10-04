# SPDX-License-Identifier: Apache-2.0

"""Frame-average Patlak fitting and formula-derived Ki/Vd ground truth."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import trapezoid

from .config import PatlakConfig

if TYPE_CHECKING:
    from .sirf_ops import SirfContext


def run_patlak(project_h5, context, config, aif_time_min, aif_cp, points_per_min, lesion_parameters, *, compression,
               logger):
    import h5py

    from .io_utils import recreate_h5_dataset, write_h5_attrs, write_h5_dataset
    from .sirf_ops import make_sirf_image_from_zyx, zoom_to_template

    with h5py.File(project_h5, "r") as handle:
        frame_start_all = handle["frames/start_min"][:].astype(np.float64)
        frame_end_all = handle["frames/end_min"][:].astype(np.float64)
        frame_mid_all = handle["frames/mid_min"][:].astype(np.float64)
    frame_indices = np.where((frame_start_all >= config.start_min) & (frame_end_all <= config.end_min))[0].astype(
        np.int32)
    if frame_indices.size < 2:
        raise RuntimeError("fewer than two complete frames are available for Patlak fitting")
    starts = frame_start_all[frame_indices]
    ends = frame_end_all[frame_indices]
    mids = frame_mid_all[frame_indices]
    cp_average, cumulative_aif_average = average_aif_terms_over_frames(starts, ends, aif_time_min, aif_cp,
                                                                       points_per_min)
    patlak_x, slope_weights, intercept_weights = make_patlak_regression_weights(cp_average, cumulative_aif_average,
                                                                                config.cp_epsilon)

    reconstructed_ki = np.zeros(context.target_shape_zyx, dtype=np.float32)
    reconstructed_vd = np.zeros(context.target_shape_zyx, dtype=np.float32)
    with h5py.File(project_h5, "r") as handle:
        reconstruction = handle["recon/osem_tzyx"]
        expected = (len(frame_mid_all),) + context.target_shape_zyx
        if reconstruction.shape != expected:
            raise RuntimeError(f"reconstruction shape {reconstruction.shape} != expected {expected}")
        for local_index, frame_index in enumerate(frame_indices):
            frame = reconstruction[int(frame_index)].astype(np.float32)
            accumulate_patlak_frame(reconstructed_ki, reconstructed_vd, frame, cp_average[local_index],
                                    slope_weights[local_index], intercept_weights[local_index])
            del frame
            gc.collect()

    with h5py.File(project_h5, "r") as handle:
        label = handle["gt/label_bed_zyx"][:].astype(np.int32)
        lesion_mask = handle["masks/lesion_mask_zyx"][:].astype(bool)
        label_ids = handle["tac/label_ids"][:].astype(np.int32)
        parameters = handle["tac/kinetic_parameters_by_label"][:].astype(np.float64)
    if parameters.shape != (len(label_ids), 5):
        raise RuntimeError("kinetic parameter matrix must have columns k1,k2,k3,k4,Vb")
    ground_truth_ki_source = np.zeros(label.shape, dtype=np.float32)
    ground_truth_vd_source = np.zeros(label.shape, dtype=np.float32)
    nonzero_k4_labels: list[int] = []
    for row, label_id in enumerate(label_ids):
        k1, k2, k3, k4, vb = parameters[row]
        if abs(float(k4)) > 1e-8:
            nonzero_k4_labels.append(int(label_id))
        ki_value, vd_value = kinetic_parameters_to_patlak_gt(k1, k2, k3, vb)
        mask = label == int(label_id)
        ground_truth_ki_source[mask] = ki_value
        ground_truth_vd_source[mask] = vd_value
    lesion_ki, lesion_vd = kinetic_parameters_to_patlak_gt(
        lesion_parameters["k1"],
        lesion_parameters["k2"],
        lesion_parameters["k3"],
        lesion_parameters["Vb"]
    )
    ground_truth_ki_source[lesion_mask] = lesion_ki
    ground_truth_vd_source[lesion_mask] = lesion_vd
    if nonzero_k4_labels:
        logger.warning("nonzero k4 labels still use the requested irreversible formula: %s", nonzero_k4_labels)

    ground_truth_ki_target = zoom_to_template(
        make_sirf_image_from_zyx(ground_truth_ki_source, context.source_template),
        context.target_template,
        scaling="preserve_values"
    ).as_array().astype(np.float32)
    ground_truth_vd_target = zoom_to_template(
        make_sirf_image_from_zyx(ground_truth_vd_source, context.source_template),
        context.target_template,
        scaling="preserve_values"
    ).as_array().astype(np.float32)

    maps = {
        "reconstructed_ki_zyx": reconstructed_ki,
        "reconstructed_vd_zyx": reconstructed_vd,
        "ground_truth_ki_source_zyx": ground_truth_ki_source,
        "ground_truth_vd_source_zyx": ground_truth_vd_source,
        "ground_truth_ki_zyx": ground_truth_ki_target,
        "ground_truth_vd_zyx": ground_truth_vd_target
    }
    for name, array in maps.items():
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"Patlak map {name} contains NaN or Inf")
        write_h5_dataset(project_h5, f"patlak/{name}", array, compression=compression)
    if np.any(ground_truth_ki_source < -1e-8) or np.any(ground_truth_vd_source < -1e-8):
        raise RuntimeError("formula-derived Patlak ground truth contains negative values")

    fit_data = {
        "frame_indices": frame_indices,
        "frame_start_min": starts,
        "frame_end_min": ends,
        "frame_mid_min": mids,
        "cp_frame_average": cp_average,
        "cumulative_aif_frame_average": cumulative_aif_average,
        "x": patlak_x
    }
    for name, array in fit_data.items():
        write_h5_dataset(
            project_h5,
            f"patlak/{name}",
            array,
            compression=False,
            dtype=np.int32 if name == "frame_indices" else np.float32
        )

    unet_shape = (2,) + context.target_shape_zyx
    with h5py.File(project_h5, "a") as handle:
        unet_input = recreate_h5_dataset(
            handle,
            "unet/input_czyx",
            unet_shape,
            np.float32,
            compression=compression
        )
        unet_target = recreate_h5_dataset(
            handle,
            "unet/target_czyx",
            unet_shape,
            np.float32,
            compression=compression
        )
        unet_input[0] = reconstructed_ki
        unet_input[1] = reconstructed_vd
        unet_target[0] = ground_truth_ki_target
        unet_target[1] = ground_truth_vd_target
        handle.flush()
    write_h5_attrs(
        project_h5,
        "patlak",
        {
            "model": "standard irreversible Patlak",
            "fit_equation": "Ct_frame_average/Cp_frame_average = Ki * cumulative_AIF_frame_average/Cp_frame_average + Vd",
            "fit_method": "voxelwise OLS for reconstructed maps only",
            "fit_start_min": config.start_min,
            "fit_end_min": config.end_min,
            "ct_time_definition": "frame average",
            "cp_time_definition": "frame average",
            "cumulative_aif_time_definition": "frame average",
            "ground_truth_source": "direct formulas from transformed-label kinetic parameters",
            "ground_truth_ki_formula": "(1 - Vb) * K1 * k3 / (k2 + k3)",
            "ground_truth_vd_formula": "(1 - Vb) * K1 * k2 / (k2 + k3)^2 + Vb",
            "ground_truth_zero_rate_rule": "K1=k2=k3=0: Ki=0, Vd=Vb",
            "ground_truth_zero_rate_assumption": "zero initial tissue activity; blood term Vb*Cp",
            "negative_reconstructed_values_clipped": False
        }
    )
    write_h5_attrs(
        project_h5,
        "unet",
        {
            "input_dataset": "unet/input_czyx",
            "target_dataset": "unet/target_czyx",
            "data_order": "channel,z,y,x",
            "channel_names": ["Ki", "Vd"],
            "input_source": "fitted reconstructed Ki/Vd from motion-corrupted OSEM frames",
            "target_source": "formula GT Ki/Vd resampled to the scanner grid",
            "normalisation_applied": False,
            "dtype": "float32"
        },
    )
    logger.info("Patlak and U-Net datasets saved; frames=%s", frame_indices.tolist())


def average_aif_terms_over_frames(frame_start_min, frame_end_min, aif_time_min, aif_cp, points_per_min):
    starts = np.asarray(frame_start_min, dtype=np.float64)
    ends = np.asarray(frame_end_min, dtype=np.float64)
    max_time = float(np.max(ends))
    uniform_time = np.linspace(0.0,max_time,int(np.ceil(max_time * points_per_min)) + 1,dtype=np.float64)
    aif_time = np.asarray(aif_time_min, dtype=np.float64)
    integration_time = np.unique(np.concatenate([uniform_time, aif_time[(aif_time >= 0.0) & (aif_time <= max_time)], starts, ends]))
    cp_continuous = np.interp(
        integration_time,
        aif_time,
        np.asarray(aif_cp, dtype=np.float64),
        left=0.0,
        right=float(aif_cp[-1])
    )
    cumulative_aif = cumulative_trapezoid(cp_continuous, integration_time)
    cp_average: list[float] = []
    cumulative_average: list[float] = []
    for start, end in zip(starts, ends):
        mask = (integration_time >= start) & (integration_time <= end)
        time = integration_time[mask]
        duration = end - start
        cp_average.append(float(trapezoid(cp_continuous[mask], time) / duration))
        cumulative_average.append(float(trapezoid(cumulative_aif[mask], time) / duration))
    return np.asarray(cp_average), np.asarray(cumulative_average)


def cumulative_trapezoid(values, time):
    output = np.zeros_like(time, dtype=np.float64)
    if len(time) > 1:
        output[1:] = np.cumsum(0.5 * (values[:-1] + values[1:]) * np.diff(time))
    return output


def make_patlak_regression_weights(cp_average, cumulative_aif_average, cp_epsilon):
    cp_average = np.asarray(cp_average, dtype=np.float64)
    if len(cp_average) < 2 or np.any(cp_average <= cp_epsilon):
        raise ValueError("Patlak fitting requires at least two frames with positive Cp")
    x = np.asarray(cumulative_aif_average, dtype=np.float64) / cp_average
    centered = x - np.mean(x)
    sum_squares = float(np.sum(centered ** 2))
    if sum_squares <= 0:
        raise ValueError("Patlak x has no variation")
    slope = centered / sum_squares
    intercept = 1.0 / len(x) - float(np.mean(x)) * slope
    return x, slope.astype(np.float32), intercept.astype(np.float32)


def accumulate_patlak_frame(ki_map, vd_map, tissue_frame_average, cp_frame_average, slope_weight, intercept_weight):
    y = np.asarray(tissue_frame_average, dtype=np.float32) / np.float32(cp_frame_average)
    np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    ki_map += np.float32(slope_weight) * y
    vd_map += np.float32(intercept_weight) * y


def kinetic_parameters_to_patlak_gt(k1, k2, k3, vb):
    """Convert kinetic parameters, including exact-zero-rate blood/background.

    Assumes zero initial tissue activity and the pipeline's Vb * Cp blood term.
    This targeted fix does not add support for K1 > 0 with k2 = k3 = 0.
    """
    k1, k2, k3, vb = (float(value) for value in (k1, k2, k3, vb))
    values = np.asarray([k1, k2, k3, vb], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Kinetic parameters must be finite: {values.tolist()}")
    if k1 < 0.0 or k2 < 0.0 or k3 < 0.0:
        raise ValueError(f"Kinetic rates must be nonnegative: {values.tolist()}")
    if not 0.0 <= vb <= 1.0:
        raise ValueError(f"Vb must be in [0, 1], got {vb}")

    # Use exact zeros; small positive rates are not silently treated as zero.
    if k1 == 0.0 and k2 == 0.0 and k3 == 0.0:
        return 0.0, vb

    denominator = float(k2) + float(k3)
    if denominator <= 0:
        raise ValueError(
            "This GT conversion does not implement the K1 > 0, k2 = k3 = 0 "
            "special case; review the intended model rather than replacing "
            "zero rates with epsilon."
        )
    ki = (1.0 - float(vb)) * float(k1) * float(k3) / denominator
    vd = (1.0 - float(vb)) * float(k1) * float(k2) / denominator ** 2 + float(vb)
    if not np.all(np.isfinite([ki, vd])):
        raise ValueError(f"Non-finite Patlak parameters: Ki={ki}, Vd={vd}")
    return float(ki), float(vd)
