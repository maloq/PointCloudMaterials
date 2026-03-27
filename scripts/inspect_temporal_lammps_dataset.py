from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils.temporal_lammps_dataset import (
    TemporalLAMMPSDumpDataset,
    inspect_lammps_dump_file,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a temporal LAMMPS dump file and optionally build the "
            "persistent cache used by TemporalLAMMPSDumpDataset."
        )
    )
    parser.add_argument(
        "--dump-file",
        required=True,
        help="Path to the LAMMPS dump file (for example datasets/dump_Pure_Al_500K.pos).",
    )
    parser.add_argument(
        "--build-cache",
        action="store_true",
        help="Instantiate TemporalLAMMPSDumpDataset after inspection to build/reuse the persistent cache.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory override. Default: <dump>.temporal_cache",
    )
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument(
        "--center-atom-id",
        type=int,
        action="append",
        default=None,
        help="Specific atom id to track. May be passed multiple times.",
    )
    parser.add_argument(
        "--center-atom-stride",
        type=int,
        default=None,
        help="Track every k-th atom id after sorting by atom id.",
    )
    parser.add_argument(
        "--max-center-atoms",
        type=int,
        default=None,
        help="Track a random subset of this many atoms.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index to materialize after cache build.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force a rebuild of the cache, ignoring any existing valid cache.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = inspect_lammps_dump_file(args.dump_file)
    print(json.dumps(summary, indent=2))

    if not args.build_cache:
        return

    if (
        args.center_atom_id is None
        and args.center_atom_stride is None
        and args.max_center_atoms is None
    ):
        raise ValueError(
            "Cache build mode requires one of --center-atom-id, "
            "--center-atom-stride, or --max-center-atoms."
        )

    dataset = TemporalLAMMPSDumpDataset(
        dump_file=args.dump_file,
        cache_dir=args.cache_dir,
        sequence_length=int(args.sequence_length),
        num_points=int(args.num_points),
        radius=float(args.radius),
        frame_stride=int(args.frame_stride),
        window_stride=int(args.window_stride),
        center_atom_ids=args.center_atom_id,
        center_atom_stride=args.center_atom_stride,
        max_center_atoms=args.max_center_atoms,
        rebuild_cache=bool(args.rebuild_cache),
    )
    sample = dataset[int(args.sample_index)]
    sample_summary = {
        "dataset_len": len(dataset),
        "sample_index": int(args.sample_index),
        "points_shape": list(sample["points"].shape),
        "timesteps": sample["timesteps"].tolist(),
        "frame_indices": sample["frame_indices"].tolist(),
        "center_atom_id": int(sample["center_atom_id"].item()),
        "cache_dir": str(dataset.cache_dir),
    }
    print(json.dumps(sample_summary, indent=2))


if __name__ == "__main__":
    main()
