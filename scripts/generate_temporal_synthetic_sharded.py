from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils.synthetic.temporal.sharded import generate_temporal_dataset_shards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate multiple temporal synthetic atomistic dataset shards in parallel."
    )
    parser.add_argument("--config", required=True, help="Path to the temporal synthetic YAML config.")
    parser.add_argument("--num-shards", required=True, type=int, help="Number of independent shards to generate.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of shard processes to run in parallel. Defaults to min(num_shards, cpu_count).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root for the shard collection. Defaults to '<config output_dir>_sharded'.",
    )
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=10_000,
        help="Seed increment between successive shards.",
    )
    parser.add_argument(
        "--keep-visualization",
        action="store_true",
        help="Keep visualization enabled inside each shard. Disabled by default for throughput.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable shard progress prints.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    generate_temporal_dataset_shards(
        args.config,
        num_shards=args.num_shards,
        max_workers=args.max_workers,
        output_root=args.output_root,
        shard_seed_stride=args.seed_stride,
        disable_visualization=not args.keep_visualization,
        progress=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
