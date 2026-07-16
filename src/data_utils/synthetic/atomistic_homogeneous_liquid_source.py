"""CLI for an immutable liquid-only homogeneous-crystallization source."""

from __future__ import annotations

import argparse
from pathlib import Path

from .atomistic.config import load_config
from .atomistic.homogeneous_liquid_source import generate_homogeneous_liquid_source


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare only the validated bulk liquid needed by homogeneous-crystallization "
            "replicas. The output is immutable and omits solid-liquid interface simulation."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    generate_homogeneous_liquid_source(load_config(args.config))


if __name__ == "__main__":
    main()
