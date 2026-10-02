# SPDX-License-Identifier: Apache-2.0

"""Sequential, resumable batch runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import traceback

from .config import SimulationConfig, sample_seed
from .inputs import load_inputs
from .io_utils import (
    append_jsonl,
    close_logger,
    configure_logging,
    is_completed_sample,
    read_json,
    utc_now,
    write_json_atomic,
)
from .pipeline import SamplePaths, run_sample


@dataclass
class BatchResult:
    completed: int = 0
    skipped: int = 0
    failed: int = 0


def run_batch(config, *, force, verbose):
    output_root = Path(config.batch.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "batch_manifest.jsonl"
    write_json_atomic(output_root / "resolved_config.json", config.to_dict())
    batch_logger = configure_logging(output_root / "batch.log", verbose=verbose)
    batch_logger.info("loading immutable inputs once for the whole batch")
    inputs = load_inputs(config.inputs)
    result = BatchResult()
    first = config.batch.start_index
    stop = first + config.batch.num_samples
    for sample_index in range(first, stop):
        seed = sample_seed(config.batch.master_seed, sample_index)
        sample_dir = output_root / f"sample_{sample_index:06d}"
        paths = SamplePaths.create(sample_dir, config.output.h5_name)
        if not force and config.batch.resume and is_completed_sample(sample_dir, config.output.h5_name):
            result.skipped += 1
            event = {
                "time": utc_now(),
                "sample_index": sample_index,
                "seed": seed,
                "state": "skipped_completed",
                "sample_dir": str(sample_dir)
            }
            append_jsonl(manifest, event)
            batch_logger.info("sample %06d already completed; skipping", sample_index)
            continue
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_logger = configure_logging(paths.log_file, verbose=verbose)
        sample_logger.info("starting sample %06d with seed %d", sample_index, seed)
        try:
            status = run_sample(inputs, config, sample_index, seed, paths, sample_logger)
        except Exception as error:  # keep the batch alive unless fail_fast is requested
            result.failed += 1
            try:
                previous_status = read_json(paths.status_json)
            except Exception:
                previous_status = {}
            failed_status = {
                "state": "failed",
                "sample_index": sample_index,
                "seed": seed,
                "stage": previous_status.get("stage", "unknown"),
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc()
            }
            write_json_atomic(paths.status_json, failed_status)
            append_jsonl(
                manifest,
                {
                    **failed_status,
                    "traceback": None,
                    "sample_dir": str(sample_dir)
                }
            )
            sample_logger.exception("sample %06d failed", sample_index)
            if not config.output.keep_failed_sample:
                shutil.rmtree(sample_dir, ignore_errors=True)
            if config.batch.fail_fast:
                raise
        else:
            result.completed += 1
            append_jsonl(
                manifest,
                {
                    "time": utc_now(),
                    "sample_index": sample_index,
                    "seed": seed,
                    "state": "completed",
                    "sample_dir": str(sample_dir),
                    "project_h5": status["project_h5"]
                }
            )
            sample_logger.info("sample %06d completed", sample_index)
        finally:
            close_logger(sample_logger)
    batch_logger.info(
        "batch finished: completed=%d skipped=%d failed=%d",
        result.completed,
        result.skipped,
        result.failed
    )
    return result
