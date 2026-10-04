# SPDX-License-Identifier: Apache-2.0

"""Frame timing, two-tissue-compartment TAC simulation, and emission assignment."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from scipy.integrate import solve_ivp, trapezoid

from .config import FramesConfig, KineticsConfig

PARAMETER_NAMES = ("k1", "k2", "k3", "k4", "Vb")


@dataclass
class KineticsSample:
    frame_start_min: np.ndarray
    frame_end_min: np.ndarray
    frame_mid_min: np.ndarray
    label_ids: np.ndarray
    tac_frame_by_label: np.ndarray
    kinetic_source_label_ids: np.ndarray
    kinetic_parameters_by_label: np.ndarray
    organ_tac: np.ndarray
    lesion_tac: np.ndarray
    organ_continuous_time_min: np.ndarray
    organ_continuous_activity: np.ndarray
    lesion_continuous_time_min: np.ndarray
    lesion_continuous_activity: np.ndarray
    labels_without_own_kinetics: np.ndarray
    labels_mapped_to_replacement: np.ndarray
    unfilled_nonzero_labels: np.ndarray
    replacement_label: int | None

    @property
    def n_frames(self):
        return int(self.frame_mid_min.size)

    def emission_frame_xyz(self, label_xyz, lesion_mask_xyz, frame_index):
        emission = np.zeros(label_xyz.shape, dtype=np.float32)
        for row, label in enumerate(self.label_ids):
            emission[label_xyz == int(label)] = self.tac_frame_by_label[row, frame_index]
        emission[np.asarray(lesion_mask_xyz, dtype=bool)] = self.lesion_tac[frame_index]
        return emission


def generate_kinetics(label_xyz, organ_id, regions, lesion_parameters, aif_time_min, aif_cp, frames_config,
                      kinetics_config, logger):
    frame_start_min, frame_end_min = make_frame_intervals(
        frames_config.n_frames,
        frames_config.duration_seconds,
        frames_config.start_scan_seconds,
        frames_config.end_scan_seconds
    )
    nonzero_labels = sorted(int(value) for value in np.unique(label_xyz) if int(value) != 0)
    if organ_id not in nonzero_labels or organ_id not in regions:
        raise RuntimeError(f"target organ label {organ_id} lacks anatomy or kinetic parameters")
    labels_without = [label for label in nonzero_labels if label not in regions]
    replacement_label: int | None = None
    if kinetics_config.use_missing_replacement and labels_without:
        candidates = [label for label, info in regions.items() if
                      str(info["name"]).strip().lower() == kinetics_config.replacement_name.strip().lower()]
        if not candidates:
            raise RuntimeError(f"replacement tissue {kinetics_config.replacement_name!r} is absent from lookup table")
        replacement_label = int(candidates[0])

    source_by_label: dict[int, int] = {}
    for label in nonzero_labels:
        if label in regions:
            source_by_label[label] = label
        elif replacement_label is not None:
            source_by_label[label] = replacement_label
    label_ids = np.asarray(sorted(source_by_label), dtype=np.int32)
    if label_ids.size == 0:
        raise RuntimeError("no label in the cropped FOV can produce emission")

    simulation_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    tacs: list[np.ndarray] = []
    parameters: list[list[float]] = []
    source_ids: list[int] = []
    organ_continuous: tuple[np.ndarray, np.ndarray] | None = None
    frame_mid_min: np.ndarray | None = None
    for label in label_ids:
        source_label = source_by_label[int(label)]
        if source_label not in simulation_cache:
            parameter_dict = {name: float(regions[source_label][name]) for name in PARAMETER_NAMES}
            simulation_cache[source_label] = simulate_tac_2tc(
                parameter_dict,
                aif_time_min,
                aif_cp,
                frame_start_min,
                frame_end_min,
                frames_config.points_per_min
            )
        current_mid, current_tac, current_time, current_activity = simulation_cache[source_label]
        frame_mid_min = current_mid if frame_mid_min is None else frame_mid_min
        tacs.append(current_tac.astype(np.float32))
        parameters.append([float(regions[source_label][name]) for name in PARAMETER_NAMES])
        source_ids.append(source_label)
        if int(label) == organ_id:
            organ_continuous = (current_time.copy(), current_activity.copy())

    if frame_mid_min is None or organ_continuous is None:
        raise RuntimeError("failed to generate target-organ TAC")
    lesion_mid, lesion_tac, lesion_time, lesion_activity = simulate_tac_2tc(
        lesion_parameters,
        aif_time_min,
        aif_cp,
        frame_start_min,
        frame_end_min,
        frames_config.points_per_min
    )
    np.testing.assert_allclose(lesion_mid, frame_mid_min)
    row_by_label = {int(label): row for row, label in enumerate(label_ids)}
    organ_tac = tacs[row_by_label[organ_id]]
    unfilled = sorted(set(nonzero_labels) - set(int(value) for value in label_ids))
    mapped = labels_without if replacement_label is not None else []
    if unfilled:
        logger.warning("labels without emission because replacement is disabled: %s", unfilled)
    logger.info("generated TACs for %d labels and %d frames", len(label_ids), len(frame_mid_min))
    return KineticsSample(
        frame_start_min=frame_start_min.astype(np.float32),
        frame_end_min=frame_end_min.astype(np.float32),
        frame_mid_min=frame_mid_min.astype(np.float32),
        label_ids=label_ids,
        tac_frame_by_label=np.stack(tacs).astype(np.float32),
        kinetic_source_label_ids=np.asarray(source_ids, dtype=np.int32),
        kinetic_parameters_by_label=np.asarray(parameters, dtype=np.float32),
        organ_tac=np.asarray(organ_tac, dtype=np.float32),
        lesion_tac=np.asarray(lesion_tac, dtype=np.float32),
        organ_continuous_time_min=organ_continuous[0].astype(np.float32),
        organ_continuous_activity=organ_continuous[1].astype(np.float32),
        lesion_continuous_time_min=lesion_time.astype(np.float32),
        lesion_continuous_activity=lesion_activity.astype(np.float32),
        labels_without_own_kinetics=np.asarray(labels_without, dtype=np.int32),
        labels_mapped_to_replacement=np.asarray(mapped, dtype=np.int32),
        unfilled_nonzero_labels=np.asarray(unfilled, dtype=np.int32),
        replacement_label=replacement_label
    )


def make_frame_intervals(n_frames, duration_seconds, start_scan_seconds, end_scan_seconds):
    latest_start = end_scan_seconds - duration_seconds
    starts = np.linspace(start_scan_seconds, latest_start, n_frames, dtype=np.float64)
    return starts / 60.0, (starts + duration_seconds) / 60.0


def simulate_tac_2tc(parameters, aif_time_min, aif_cp, frame_start_min, frame_end_min, points_per_min):
    k1, k2, k3, k4, vb = (float(parameters[name]) for name in PARAMETER_NAMES)

    def cp(time):
        return np.interp(time, aif_time_min, aif_cp, left=0.0, right=float(aif_cp[-1]))

    def ode(time, state):
        compartment_1, compartment_2 = state
        plasma = float(cp(time))
        return [
            k1 * plasma - (k2 + k3) * compartment_1 + k4 * compartment_2,
            k3 * compartment_1 - k4 * compartment_2
        ]

    end_min = float(frame_end_min[-1])
    time_eval = np.linspace(
        0.0,
        end_min,
        int(np.ceil(end_min * points_per_min)) + 1,
        dtype=np.float64
    )
    solution = solve_ivp(
        ode,
        (0.0, end_min),
        [0.0, 0.0],
        t_eval=time_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    tissue = (1.0 - vb) * (solution.y[0] + solution.y[1]) + vb * cp(solution.t)
    frame_mid, frame_average = average_tac_over_frames(solution.t, tissue, frame_start_min, frame_end_min)
    return frame_mid, frame_average, solution.t, np.asarray(tissue)


def average_tac_over_frames(continuous_time, continuous_tac, frame_start_min, frame_end_min):
    mids: list[float] = []
    averages: list[float] = []
    for start, end in zip(frame_start_min, frame_end_min):
        inside = (continuous_time >= start) & (continuous_time <= end)
        time_segment = continuous_time[inside]
        tac_segment = continuous_tac[inside]
        if time_segment.size == 0 or time_segment[0] > start:
            time_segment = np.insert(time_segment, 0, start)
            tac_segment = np.insert(tac_segment, 0, np.interp(start, continuous_time, continuous_tac))
        if time_segment[-1] < end:
            time_segment = np.append(time_segment, end)
            tac_segment = np.append(tac_segment, np.interp(end, continuous_time, continuous_tac))
        averages.append(float(trapezoid(tac_segment, time_segment) / (end - start)))
        mids.append(float((start + end) / 2.0))
    return np.asarray(mids, dtype=np.float64), np.asarray(averages, dtype=np.float64)
