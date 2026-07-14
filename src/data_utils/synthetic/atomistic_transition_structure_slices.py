"""CLI for adding PTM-colored structure slices to transition datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from .atomistic.transition_config import load_transition_config
from .atomistic.transition_slices import add_structure_slices_to_transition_datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render initial, midpoint, and final atomic slices for transition branches."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, action="append", type=Path)
    args = parser.parse_args()
    add_structure_slices_to_transition_datasets(
        load_transition_config(args.config),
        tuple(path.expanduser().resolve() for path in args.dataset),
    )


if __name__ == "__main__":
    main()
