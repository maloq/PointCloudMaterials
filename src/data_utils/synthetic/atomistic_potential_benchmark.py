"""Compare pinned MACE models on the aluminium crystallization use case."""

from __future__ import annotations

import argparse
from pathlib import Path

from .atomistic.potential_benchmark import (
    load_potential_benchmark_config,
    run_potential_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark explicitly pinned MLIPs on Al phase energetics, NVE conservation, "
            "application structures, DFT references, and melting-point evidence."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_potential_benchmark_config(args.config)
    run_potential_benchmark(config)


if __name__ == "__main__":
    main()
