#!/usr/bin/env python3
"""Prepare balanced Al/Mg/Ta views from the completed 24 ps trajectories."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from numpy.lib.format import open_memmap

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from src.data_utils.spatiotemporal_views import periodic_tree, local_views
from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory


def prepare_branch(task):
    source, output, branch_index, seed = task
    trajectory = TemporalLAMMPSBinaryTrajectory.load(source["path"])
    if trajectory.frame_count != 241:
        raise ValueError(f"Expected 241 saved frames at 0.1 ps spacing: {source['path']}")
    expected_step = 50 if source["material"] == "Ta" else 100
    if not np.array_equal(trajectory.timesteps, np.arange(241) * expected_step):
        raise ValueError(f"Unexpected timestep grid: {source['path']}")
    rng = np.random.default_rng(seed + branch_index)
    multiplier = 6 if source["material"] == "Ta" else 1
    shards = []
    for split, anchors, centers in (("train", range(20, 176, 5), 1024 * multiplier),
                                     ("val", range(200, 236, 5), 256 * multiplier)):
        count = len(anchors) * centers
        stem = f"{source['material']}_{source['snapshot']}_{split}"
        views_file, pairs_file = f"{stem}.views.npy", f"{stem}.pairs.npy"
        views = open_memmap(output / views_file, mode="w+", dtype=np.float16, shape=(count, 3, 80, 3))
        pairs = open_memmap(output / pairs_file, mode="w+", dtype=np.int64, shape=(count, 4))
        rows = np.arange(trajectory.atom_count)
        pool = rows[rows % 5 != 0] if split == "train" else rows[rows % 5 == 0]
        for j, anchor in enumerate(anchors):
            selected = rng.choice(pool, size=centers, replace=False)
            lengths = trajectory.box_high[anchor] - trajectory.box_low[anchor]
            points, tree = periodic_tree(trajectory.positions[anchor], lengths)
            _, nearest = tree.query(points[selected], k=9, workers=1)
            candidates = nearest[nearest != selected[:, None]].reshape(centers, 8)
            spatial = candidates[np.arange(centers), rng.integers(0, 8, size=centers)]
            batch = slice(j * centers, (j + 1) * centers)
            views[batch, 0] = local_views(points, tree, lengths, selected, num_points=80, radius=source["radius"])
            views[batch, 1] = local_views(points, tree, lengths, spatial, num_points=80, radius=source["radius"])
            lags = np.tile([1, 5], centers // 2)
            rng.shuffle(lags)
            for lag in (1, 5):
                frame = anchor + lag
                lengths_t = trajectory.box_high[frame] - trajectory.box_low[frame]
                points_t, tree_t = periodic_tree(trajectory.positions[frame], lengths_t)
                mask = lags == lag
                views[j * centers + np.flatnonzero(mask), 2] = local_views(
                    points_t, tree_t, lengths_t, selected[mask], num_points=80, radius=source["radius"],
                )
            pairs[batch] = np.column_stack((trajectory.atom_ids[selected], trajectory.atom_ids[spatial], np.full(centers, anchor), lags))
        views.flush()
        pairs.flush()
        shards.append(dict(material=source["material"], snapshot=source["snapshot"], split=split,
                           samples=count, views=views_file, pairs=pairs_file))
        print(f"Prepared {stem}: {count} triplets", flush=True)
    return shards


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = []
    root = REPOSITORY / "datasets/zr_al_mg_initial_6x24ps/branches"
    for material, snapshots, radius in (
        ("Al", ("166ps", "170ps", "174ps", "175ps", "177ps", "240ps"), 9.192189),
        ("Mg", ("940ps", "960ps", "980ps", "990ps", "1000ps", "1500ps"), 10.169428),
        ("Ta", ("model_1m",), 9.388275),
    ):
        for snapshot in snapshots:
            path = (REPOSITORY / "datasets/ta_initial_1x24ps/branches/Ta/model_1m/trajectory_binary_float32"
                    if material == "Ta" else root / material / snapshot / "trajectory_binary_float16")
            manifest = (path / "manifest.json").read_bytes()
            sources.append(dict(material=material, snapshot=snapshot, radius=radius, path=str(path.resolve()),
                                manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                                storage_dtype=json.loads(manifest)["storage_dtype"]))
    report = dict(state="building", sources=sources, seed=20260905, num_points=80,
                  views=["anchor", "spatial_neighbor", "same_atom_future"],
                  temporal_lags_ps=[0.1, 0.5], spatial_neighbor_k=8,
                  train_anchor_ps=[2.0, 17.5], val_anchor_ps=[20.0, 23.5],
                  pair_columns=["center_atom_id", "spatial_center_atom_id", "anchor_frame", "lag_frames"],
                  split_note="Disjoint time blocks and center-ID pools; same source trajectories, not independent-trajectory validation. Ta has only one trajectory.",
                  cache_storage_dtype="float16",
                  quantization_note="Local views use float16 storage and float32 decoding. Al/Mg also retain global-coordinate float16 quantization from supplied files.")
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    started = time.monotonic()
    try:
        shards = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(prepare_branch, (source, args.output, index, report["seed"])) for index, source in enumerate(sources)]
            for future in as_completed(futures):
                shards.extend(future.result())
        report.update(state="complete", shards=sorted(shards, key=lambda s: s["views"]), elapsed_seconds=time.monotonic() - started)
        for split in ("train", "val"):
            counts = {m: sum(s["samples"] for s in shards if s["material"] == m and s["split"] == split) for m in ("Al", "Mg", "Ta")}
            if len(set(counts.values())) != 1:
                raise RuntimeError(f"Material balance failed: {split}, {counts}")
            report[f"{split}_samples_per_material"] = counts
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        manifest_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
