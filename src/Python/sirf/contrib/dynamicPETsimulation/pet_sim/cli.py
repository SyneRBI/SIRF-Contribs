# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SimulationConfig, load_config, save_config


def build_parser():
    parser = argparse.ArgumentParser(description="Generate motion-corrupted dynamic PET/Patlak samples in batch.")
    parser.add_argument("--config", type=Path, default=Path("config.example.json"),
                        help="JSON configuration file (default: config.example.json)")
    parser.add_argument("--samples", type=int, help="override batch.num_samples")
    parser.add_argument("--start-index", type=int, help="override batch.start_index")
    parser.add_argument("--seed", type=int, help="override batch.master_seed")
    parser.add_argument("--output-dir", type=Path, help="override batch.output_root")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="arbitrary config override; VALUE accepts JSON syntax"
    )
    parser.add_argument("--force", action="store_true", help="rerun even completed samples")
    parser.add_argument("--no-resume", action="store_true", help="disable completed-sample skipping")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failed sample")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="load and validate configuration without running SIRF"
    )
    parser.add_argument(
        "--write-default-config",
        type=Path,
        metavar="PATH",
        help="write a complete default JSON config and exit"
    )
    return parser


def main(argv):
    args = build_parser().parse_args(argv)
    if args.write_default_config is not None:
        save_config(SimulationConfig(), args.write_default_config)
        print(f"default config written to {args.write_default_config}")
        return 0
    overrides = list(args.overrides)
    _append_override(overrides, "batch.num_samples", args.samples)
    _append_override(overrides, "batch.start_index", args.start_index)
    _append_override(overrides, "batch.master_seed", args.seed)
    _append_override(overrides, "batch.output_root", str(args.output_dir) if args.output_dir is not None else None)
    if args.no_resume:
        overrides.append("batch.resume=false")
    if args.fail_fast:
        overrides.append("batch.fail_fast=true")
    config = load_config(args.config, overrides)
    if args.validate_only:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        print("configuration is valid")
        return 0
    from .batch import run_batch

    result = run_batch(config, force=args.force, verbose=args.verbose)
    print(f"completed={result.completed} skipped={result.skipped} failed={result.failed}")
    return 1 if result.failed else 0


def _append_override(overrides: list[str], key: str, value: object | None) -> None:
    if value is not None:
        overrides.append(f"{key}={json.dumps(value)}")
