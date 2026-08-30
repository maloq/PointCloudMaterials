#!/usr/bin/env python3
"""Prepare, execute, and summarize 70k-atom 2NN-MEAM shooting branches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.synthetic.atomistic.lammps_shooting import (  # noqa: E402
    load_shooting_config,
    prepare_campaign,
    run_branch,
    submit_next_wave,
    summarize_campaign,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run position-conditioned fixed-cell Langevin-NVT branches from archived "
            "70,304-atom 2NN-MEAM crystallization frames."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", required=True, type=Path)
    task = subparsers.add_parser("run-task")
    task.add_argument("--campaign-root", required=True, type=Path)
    task.add_argument("--task-index", required=True, type=int)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--campaign-root", required=True, type=Path)
    submit_wave = subparsers.add_parser("submit-next-wave")
    submit_wave.add_argument("--campaign-root", required=True, type=Path)
    submit_wave.add_argument("--start-index", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.action == "prepare":
        manifest = prepare_campaign(load_shooting_config(args.config))
        print(json.dumps(manifest["counts"], indent=2))
    elif args.action == "run-task":
        run_branch(args.campaign_root, args.task_index)
    elif args.action == "summarize":
        summarize_campaign(args.campaign_root)
    elif args.action == "submit-next-wave":
        submit_next_wave(args.campaign_root, args.start_index)
    else:
        raise AssertionError(f"Unhandled action {args.action!r}.")


if __name__ == "__main__":
    main()
