"""Select an Al MLIP using scientific priority and absolute MD feasibility."""

from __future__ import annotations

import argparse
from pathlib import Path

from .atomistic.potential_selection import (
    load_potential_selection_config,
    select_potential,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select a pinned Al MLIP using scientific evidence and an exact "
            "full-duration NPT makespan projection."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = select_potential(load_potential_selection_config(args.config))
    print(result["selected_generator_config"])


if __name__ == "__main__":
    main()
