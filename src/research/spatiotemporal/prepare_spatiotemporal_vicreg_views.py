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

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from src.data.spatiotemporal import prepare_branch




def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, help="Explicit expanded-cache configuration; reuses this view producer.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.config is not None:
        from src.data.spatiotemporal import prepare_expanded
        from src.project_runtime.paths import load_json
        prepare_expanded(load_json(args.config))
        return
    if args.output is None:
        parser.error("--output is required without --config")
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
        for source in sources:
            multiplier = 6 if source["material"] == "Ta" else 1
            source.update(frame_count=241, timestep_stride=50 if source["material"] == "Ta" else 100,
                lags=[1,5], splits=[dict(split="train", anchors=list(range(20,176,5)), centers=1024*multiplier),
                                  dict(split="val", anchors=list(range(200,236,5)), centers=256*multiplier)])
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
