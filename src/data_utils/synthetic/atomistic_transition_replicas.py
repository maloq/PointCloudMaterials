"""Run the configured direct-coexistence temperature grid and replicas."""

from __future__ import annotations

import argparse
from pathlib import Path

from .atomistic.transition_config import load_transition_config
from .atomistic.transition_generator import generate_transition_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run every explicitly configured direct-coexistence temperature and independent "
            "replica, then summarize the spatial front velocities."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_transition_config(args.config)
    generate_transition_dataset(config)


if __name__ == "__main__":
    main()
