from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils.synthetic.temporal import generate_temporal_visualizations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate visual QA artifacts for a temporal synthetic dataset.")
    parser.add_argument("--dataset-dir", required=True, help="Temporal dataset directory to visualize.")
    parser.add_argument("--output-dir", default=None, help="Optional override for the visualization output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    generate_temporal_visualizations(args.dataset_dir, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
