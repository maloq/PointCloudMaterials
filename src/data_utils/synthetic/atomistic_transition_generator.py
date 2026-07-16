"""CLI for a replicated direct-coexistence temperature grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from .atomistic.transition_config import load_transition_config
from .atomistic.transition_generator import generate_transition_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the configured temperature grid and independent-velocity replicas from "
            "a repository-owned prepared solid-liquid interface."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    generate_transition_dataset(load_transition_config(args.config))


if __name__ == "__main__":
    main()
