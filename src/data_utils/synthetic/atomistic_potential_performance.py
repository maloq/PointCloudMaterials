"""Benchmark end-to-end production NPT throughput for pinned MLIPs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .atomistic.potential_performance import (
    load_potential_performance_config,
    run_potential_performance_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time the exact NPT production path for pinned MACE candidates."
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_potential_performance_benchmark(
        load_potential_performance_config(args.config)
    )


if __name__ == "__main__":
    main()
