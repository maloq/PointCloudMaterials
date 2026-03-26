from __future__ import annotations

import argparse

from .api import generate_temporal_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a temporal synthetic atomistic benchmark dataset.")
    parser.add_argument("--config", required=True, help="Path to the temporal synthetic YAML config.")
    parser.add_argument("--output-dir", default=None, help="Optional override for output.output_dir.")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for the config seed.")
    parser.add_argument("--quiet", action="store_true", help="Disable generation progress prints.")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Accepted for interface stability. Validation still runs and will fail loudly on inconsistencies.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    generate_temporal_dataset(
        args.config,
        seed=args.seed,
        output_dir=args.output_dir,
        validate=not args.skip_validation,
        progress=not args.quiet,
    )
    return 0
