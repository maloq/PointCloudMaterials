"""CLI for force-driven atomistic benchmark generation.

The implementation lives in :mod:`src.data_utils.synthetic.atomistic`.  This
module remains the executable entry point because repository scripts invoke it
with ``python -m src.data_utils.synthetic.atomistic_generator``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .atomistic import GenerationResult, GeneratorConfig, generate_dataset, load_config

__all__ = ["GenerationResult", "GeneratorConfig", "generate_dataset", "load_config"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate physically simulated bulk-solid, bulk-liquid, and "
            "solid-liquid-interface atomistic environments."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages; exceptions and validation failures remain visible.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    progress = (lambda _message: None) if args.quiet else print
    generate_dataset(config, progress=progress)


if __name__ == "__main__":
    main()
