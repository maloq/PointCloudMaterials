#!/usr/bin/env python3
"""Unified experiment runner.

Launch experiment plans defined in YAML with SLURM or local execution,
multi-stage checkpoint chaining, and automated result aggregation.

Examples:
    # Submit to SLURM (default):
    python scripts/run_experiments.py --plan experiments/vicreg_encoders.yaml

    # Run locally (no SLURM):
    python scripts/run_experiments.py --plan experiments/vicreg_encoders.yaml --local

    # Dry run (preview commands without executing):
    python scripts/run_experiments.py --plan experiments/vicreg_encoders.yaml --dry-run

    # Resume a previously interrupted run:
    python scripts/run_experiments.py --plan experiments/vicreg_encoders.yaml --resume output/experiments/vicreg_encoders_20260219_120000

    # Re-collect results from an existing output directory:
    python scripts/run_experiments.py --plan experiments/vicreg_encoders.yaml --collect output/experiments/vicreg_encoders_20260219_120000
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified experiment runner: YAML plan -> SLURM/local execution -> result tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--plan", required=True, type=Path,
        help="Path to the experiment plan YAML file.",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run experiments locally as subprocesses instead of submitting to SLURM.",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of parallel local jobs (only with --local, default: 1 = sequential).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate scripts/commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue running remaining experiments when one fails.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, metavar="OUTPUT_DIR",
        help="Resume a previous run from its output directory (reads state.json).",
    )
    parser.add_argument(
        "--collect", type=Path, default=None, metavar="OUTPUT_DIR",
        help="Only collect and aggregate results from an existing output directory.",
    )
    parser.add_argument(
        "--nan-restart-max-retries", type=int, default=1,
        help=(
            "If a run fails due to NaN/non-finite loss, retry it up to this many "
            "times with reduced learning rate (default: 1, set 0 to disable)."
        ),
    )
    parser.add_argument(
        "--nan-restart-lr-factor", type=float, default=0.7,
        help=(
            "Learning-rate multiplier applied on each NaN-loss retry "
            "(must be between 0 and 1, default: 0.7)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from src.experiment_runner.plan import load_plan
    from src.experiment_runner.runner import collect_only, run_plan
    from src.experiment_runner.state import RunState

    plan_path = args.plan.resolve()
    plan = load_plan(plan_path)

    if args.nan_restart_max_retries < 0:
        print(
            "Error: --nan-restart-max-retries must be >= 0.",
            file=sys.stderr,
        )
        return 1
    if not (0.0 < args.nan_restart_lr_factor < 1.0):
        print(
            "Error: --nan-restart-lr-factor must be in (0, 1).",
            file=sys.stderr,
        )
        return 1

    print(f"Plan: {plan.name}")
    print(f"Train script: {plan.train_script}")
    print(f"Stages: {len(plan.stages)}")
    total_experiments = sum(len(s.experiments) for s in plan.stages)
    print(f"Total experiments: {total_experiments}")

    # --- Collect-only mode ---
    if args.collect is not None:
        output_dir = args.collect.resolve()
        if not output_dir.exists():
            print(f"Error: output directory does not exist: {output_dir}", file=sys.stderr)
            return 1
        collect_only(plan, output_dir)
        return 0

    # --- Determine output directory ---
    if args.resume is not None:
        output_dir = args.resume.resolve()
        if not output_dir.exists():
            print(f"Error: resume directory does not exist: {output_dir}", file=sys.stderr)
            return 1
        resume_state = RunState.load(output_dir)
        print(f"Resuming from: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (REPO_ROOT / plan.output_root / f"{plan.name}_{timestamp}").resolve()
        resume_state = None
        print(f"Output: {output_dir}")

    mode = "local" if args.local else "SLURM"
    print(f"Execution: {mode}" + (f" (parallel={args.parallel})" if args.local and args.parallel > 1 else ""))
    if args.dry_run:
        print("Mode: DRY RUN")
    print()

    try:
        run_plan(
            plan,
            output_dir=output_dir,
            repo_root=REPO_ROOT,
            local=args.local,
            parallel=args.parallel,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
            resume_state=resume_state,
            nan_restart_max_retries=args.nan_restart_max_retries,
            nan_restart_lr_factor=args.nan_restart_lr_factor,
        )
    except RuntimeError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    raise SystemExit(main())
