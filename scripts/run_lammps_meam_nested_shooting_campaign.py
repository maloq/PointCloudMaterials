#!/usr/bin/env python3
"""Prepare and run transition-balanced nested 70,304-atom MEAM shooting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.synthetic.atomistic.lammps_nested_shooting import (  # noqa: E402
    evaluate_monitor_frame,
    load_nested_shooting_config,
    prepare_nested_campaign,
    run_nested_branch,
    submit_next_nested_wave,
    summarize_nested_campaign,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    monitor = subparsers.add_parser("monitor-frame")
    monitor.add_argument("--branch-dir", type=Path, required=True)
    task = subparsers.add_parser("run-task")
    task.add_argument("--campaign-root", type=Path, required=True)
    task.add_argument("--task-index", type=int, required=True)
    submit = subparsers.add_parser("submit-next-wave")
    submit.add_argument("--campaign-root", type=Path, required=True)
    submit.add_argument("--start-index", type=int, required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--campaign-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.action == "prepare":
        manifest = prepare_nested_campaign(load_nested_shooting_config(args.config))
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    elif args.action == "monitor-frame":
        state = evaluate_monitor_frame(args.branch_dir)
        print(
            json.dumps(
                {
                    "branch_id": state["branch_id"],
                    "last_observation": state["observations"][-1],
                    "first_passage_outcome": state["first_passage_outcome"],
                },
                sort_keys=True,
            )
        )
    elif args.action == "run-task":
        run_nested_branch(args.campaign_root, args.task_index)
    elif args.action == "submit-next-wave":
        submit_next_nested_wave(args.campaign_root, args.start_index)
    elif args.action == "summarize":
        summarize_nested_campaign(args.campaign_root)
    else:
        raise AssertionError(f"Unhandled action {args.action!r}.")


if __name__ == "__main__":
    main()
