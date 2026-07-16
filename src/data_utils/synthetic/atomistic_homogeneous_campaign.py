"""Dynamic, resumable multi-GPU homogeneous-crystallization campaign CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from .atomistic.homogeneous_campaign import (
    run_analysis_worker,
    run_deferred_campaign_analysis,
    run_md_worker,
    run_optimized_campaign,
)
from .atomistic.homogeneous_campaign_config import (
    load_homogeneous_campaign_config,
)


def _devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(","))
    if not devices or any(not item for item in devices):
        raise argparse.ArgumentTypeError(
            f"devices must be a comma-separated non-empty list, got {value!r}."
        )
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError(f"devices must be unique, got {devices}.")
    return devices


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run homogeneous-crystallization replicas with persistent per-GPU models, a "
            "dynamic seed queue, exact MTK checkpoints, online event stopping, and "
            "asynchronous or deferred full PTM/RDF analysis."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run/resume GPU MD campaign.")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--devices", required=True, type=_devices)
    run_parser.add_argument("--retry-failed", action="store_true")

    analyze_parser = subparsers.add_parser(
        "analyze", help="Run/resume CPU-only full PTM/RDF analysis."
    )
    analyze_parser.add_argument("--config", required=True, type=Path)
    analyze_parser.add_argument("--workers", required=True, type=int)
    analyze_parser.add_argument("--retry-failed", action="store_true")

    worker_parser = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--config", required=True, type=Path)
    worker_parser.add_argument("--worker-name", required=True)

    analyzer_parser = subparsers.add_parser("analyzer", help=argparse.SUPPRESS)
    analyzer_parser.add_argument("--config", required=True, type=Path)
    analyzer_parser.add_argument("--worker-name", required=True)
    analyzer_parser.add_argument("--follow-md", action="store_true")

    args = parser.parse_args()
    config = load_homogeneous_campaign_config(args.config)
    if args.command == "run":
        run_optimized_campaign(
            config,
            devices=args.devices,
            retry_failed=args.retry_failed,
        )
    elif args.command == "analyze":
        run_deferred_campaign_analysis(
            config,
            workers=args.workers,
            retry_failed=args.retry_failed,
        )
    elif args.command == "worker":
        run_md_worker(
            config,
            worker_name=args.worker_name,
        )
    elif args.command == "analyzer":
        run_analysis_worker(
            config,
            worker_name=args.worker_name,
            follow_md=args.follow_md,
        )
    else:
        raise RuntimeError(f"Unsupported command={args.command!r}.")


if __name__ == "__main__":
    main()
