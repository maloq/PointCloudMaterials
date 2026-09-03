#!/usr/bin/env python3
"""Measure end-to-end binary mmap plus periodic-neighborhood input throughput."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data_utils.shooting_binary_dataset import (
    ShootingBinaryEnvironmentDataset,
    make_shooting_environment_loader,
)
from src.data_utils.shooting_dataset import (
    load_shooting_campaign_snapshot,
    load_shooting_campaigns_snapshot,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", action="append", required=True)
    parser.add_argument("--temperatures-K", nargs="+", type=float, required=True)
    parser.add_argument("--minimum-branches-per-parent", type=int, default=1)
    parser.add_argument("--timesteps", nargs="+", type=int, required=True)
    parser.add_argument("--branch-count", type=int, default=16)
    parser.add_argument("--center-atom-count", type=int, default=64)
    parser.add_argument("--center-selection-seed", type=int, default=123)
    parser.add_argument("--num-points", type=int, default=160)
    parser.add_argument("--radius", type=float, default=9.192189)
    parser.add_argument("--context-center-count", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 4])
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    roots = tuple(Path(value).expanduser().resolve() for value in args.campaign_root)
    if len(roots) == 1:
        snapshot = load_shooting_campaign_snapshot(
            roots[0],
            temperatures_K=args.temperatures_K,
            minimum_complete_branches_per_parent=args.minimum_branches_per_parent,
        )
    else:
        snapshot = load_shooting_campaigns_snapshot(
            roots,
            temperatures_K=args.temperatures_K,
            minimum_complete_branches_per_parent=args.minimum_branches_per_parent,
        )
    selected_branches = snapshot.branches[: int(args.branch_count)]
    if len(selected_branches) != int(args.branch_count):
        raise RuntimeError(
            f"Requested {args.branch_count} branches, snapshot has {len(snapshot.branches)}."
        )
    atom_count = int(snapshot.manifest["atom_count"])
    rng = np.random.default_rng(int(args.center_selection_seed))
    atom_ids = np.sort(
        rng.choice(
            np.arange(1, atom_count + 1, dtype=np.int64),
            size=int(args.center_atom_count),
            replace=False,
        )
    )
    dataset = ShootingBinaryEnvironmentDataset(
        snapshot,
        branches=selected_branches,
        timesteps=args.timesteps,
        center_atom_ids=atom_ids,
        num_points=int(args.num_points),
        radius=float(args.radius),
        spatial_context_center_count=int(args.context_center_count),
    )
    results = []
    for worker_count in args.workers:
        loader = make_shooting_environment_loader(
            dataset,
            batch_size=int(args.batch_size),
            num_workers=int(worker_count),
            pin_memory=False,
        )
        checksum = 0.0
        started = time.perf_counter()
        observed_branches = 0
        for batch in loader:
            observed_branches += int(batch["points"].shape[0])
            checksum += float(batch["points"][:, :, :, 1, :].sum().item())
        elapsed = time.perf_counter() - started
        if observed_branches != len(dataset):
            raise RuntimeError(
                f"Loader returned {observed_branches} branches, expected {len(dataset)}."
            )
        results.append(
            {
                "num_workers": int(worker_count),
                "elapsed_seconds": elapsed,
                "branches_per_second": observed_branches / elapsed,
                "frames_per_second": observed_branches * len(args.timesteps) / elapsed,
                "local_clouds_per_second": (
                    observed_branches
                    * len(args.timesteps)
                    * int(args.center_atom_count)
                    / elapsed
                ),
                "checksum": checksum,
            }
        )
    if len({value["checksum"] for value in results}) != 1:
        raise RuntimeError(f"Worker configurations returned different data: {results}.")
    print(
        json.dumps(
            {
                "campaign_roots": [str(value) for value in roots],
                "branch_count": len(dataset),
                "timesteps": list(args.timesteps),
                "center_atom_count": int(args.center_atom_count),
                "num_points": int(args.num_points),
                "context_center_count": int(args.context_center_count),
                "batch_size": int(args.batch_size),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
