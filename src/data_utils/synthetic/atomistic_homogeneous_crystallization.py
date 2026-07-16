"""CLI for unseeded crystallization from a repository-owned bulk liquid."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    parser.add_argument(
        "--shard-index",
        type=int,
        help="Zero-based replica shard index. Requires --shard-count and --output-root.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        help="Number of disjoint round-robin replica shards.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Explicit output root for this shard; must not already exist.",
    )
    args = parser.parse_args()
    config = load_homogeneous_crystallization_config(args.config)
    shard_values = (args.shard_index, args.shard_count, args.output_root)
    if any(value is not None for value in shard_values):
        if any(value is None for value in shard_values):
            raise ValueError(
                "--shard-index, --shard-count, and --output-root must be supplied together."
            )
        shard_index = int(args.shard_index)
        shard_count = int(args.shard_count)
        if shard_count <= 0:
            raise ValueError(f"--shard-count must be positive, got {shard_count}.")
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError(
                f"--shard-index must be in [0, {shard_count}), got {shard_index}."
            )
        shard_seeds = tuple(config.random_seeds[shard_index::shard_count])
        if not shard_seeds:
            raise ValueError(
                f"Replica shard {shard_index}/{shard_count} is empty for configured seeds "
                f"{config.random_seeds}."
            )
        output_root = args.output_root.expanduser().resolve()
        config = replace(
            config,
            dataset_name=f"{config.dataset_name}_shard_{shard_index:02d}_of_{shard_count:02d}",
            random_seeds=shard_seeds,
            output=replace(
                config.output,
                root_dir=output_root,
                overwrite=False,
            ),
        )
    generate_homogeneous_crystallization_dataset(
        config
    )


if __name__ == "__main__":
    main()
