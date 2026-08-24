"""Queued multi-GPU direct-coexistence campaign CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from .atomistic.transition_campaign import (
    run_analysis_worker,
    run_deferred_transition_analysis,
    run_md_worker,
    run_transition_campaign,
)
from .atomistic.transition_campaign_config import load_transition_campaign_config
from .atomistic.transition_campaign_queue import campaign_rows


def _devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(","))
    if not devices or any(not item for item in devices):
        raise argparse.ArgumentTypeError(
            f"--devices must be a comma-separated list such as 0 or 0,1, got {value!r}."
        )
    return devices


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable direct-coexistence temperature/replica queue with one "
            "persistent MACE model per GPU and deferred CPU PTM/RDF analysis."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run or resume GPU MD tasks.")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--devices", required=True, type=_devices)
    run_parser.add_argument("--retry-failed", action="store_true")
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run or resume deferred CPU PTM/RDF analysis."
    )
    analyze_parser.add_argument("--config", required=True, type=Path)
    analyze_parser.add_argument("--workers", required=True, type=int)
    analyze_parser.add_argument("--retry-failed", action="store_true")
    status_parser = subparsers.add_parser("status", help="Print task queue rows.")
    status_parser.add_argument("--config", required=True, type=Path)
    worker_parser = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--config", required=True, type=Path)
    worker_parser.add_argument("--worker-name", required=True)
    analyzer_parser = subparsers.add_parser("analyzer", help=argparse.SUPPRESS)
    analyzer_parser.add_argument("--config", required=True, type=Path)
    analyzer_parser.add_argument("--worker-name", required=True)
    args = parser.parse_args()
    config = load_transition_campaign_config(args.config)
    if args.command == "run":
        run_transition_campaign(
            config, devices=args.devices, retry_failed=args.retry_failed
        )
    elif args.command == "analyze":
        run_deferred_transition_analysis(
            config, workers=args.workers, retry_failed=args.retry_failed
        )
    elif args.command == "status":
        for row in campaign_rows(config):
            print(row)
    elif args.command == "worker":
        run_md_worker(config, worker_name=args.worker_name)
    elif args.command == "analyzer":
        run_analysis_worker(config, worker_name=args.worker_name)
    else:
        raise RuntimeError(f"Unsupported command {args.command!r}.")


if __name__ == "__main__":
    main()
