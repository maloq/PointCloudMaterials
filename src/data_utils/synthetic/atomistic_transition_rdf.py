"""CLI for adding per-phase RDF analysis to completed transition trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from .atomistic.transition_config import load_transition_config
from .atomistic.transition_rdf import add_phase_rdf_to_transition_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-phase radial distribution functions from every saved frame of a "
            "completed direct-coexistence temperature-grid dataset."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    add_phase_rdf_to_transition_dataset(load_transition_config(args.config))


if __name__ == "__main__":
    main()
