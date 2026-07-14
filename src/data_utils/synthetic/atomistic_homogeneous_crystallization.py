"""CLI for unseeded crystallization from a repository-owned bulk liquid."""

from __future__ import annotations

import argparse
from pathlib import Path

from .atomistic.homogeneous_config import load_homogeneous_crystallization_config
from .atomistic.homogeneous_generator import (
    generate_homogeneous_crystallization_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a MACE homogeneous-crystallization trajectory by quenching the "
            "repository-owned validated bulk liquid."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    generate_homogeneous_crystallization_dataset(
        load_homogeneous_crystallization_config(args.config)
    )


if __name__ == "__main__":
    main()
