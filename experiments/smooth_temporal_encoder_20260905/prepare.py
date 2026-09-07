"""Prepare the explicit Al/Mg/Ta radius-neighborhood pilot and PTM assays."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from numpy.lib.format import open_memmap
from src.data_utils.spatiotemporal_views import periodic_tree
from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def ptm_labels(offsets, rmsd_cutoff):
    from ovito.data import DataCollection, Particles
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    count, neighbors, _ = offsets.shape
    points = np.concatenate((np.zeros((count, 1, 3)), offsets.astype(np.float64)), axis=1)
    points[:, :, 0] += 4 * np.arange(count)[:, None]
    particles = Particles(count=count*(neighbors+1))
    particles.create_property("Position", data=points.reshape(-1, 3))
    data = DataCollection()
    data.objects.append(particles)
    pipeline = Pipeline(source=StaticSource(data=data))
    # only_selected also excludes unselected atoms from neighbor search in OVITO.
    # Analyze the full local clouds, then read only their tracked centers.
    pipeline.modifiers.append(PolyhedralTemplateMatchingModifier(rmsd_cutoff=rmsd_cutoff))
    result = pipeline.compute()
    return np.asarray(result.particles["Structure Type"])[::neighbors+1].copy()


def local_offsets(points, lengths, rows, indices, radius):
    offsets = points[indices].astype(np.float64) - points[rows, None]
    offsets -= lengths * np.round(offsets / lengths)
    return (offsets / radius).astype(np.float32)


def prepare_branch(source, config, index):
    # OVITO parallelism is bounded in each spawned worker before its first import.
    os.environ["OVITO_THREAD_COUNT"] = "1"
    trajectory = TemporalLAMMPSBinaryTrajectory.load(source["path"])
    manifest_hash = hashlib.sha256((trajectory.root / "manifest.json").read_bytes()).hexdigest()
    if manifest_hash != source["manifest_sha256"]:
        raise RuntimeError(f"Source manifest changed: {trajectory.root}")
    steps = 50 if source["material"] == "Ta" else 100
    if not np.array_equal(trajectory.timesteps, np.arange(241) * steps):
        raise ValueError(f"Expected complete 0.1 ps timeline: {trajectory.root}")
    output = ROOT / config["output"] / "data"
    stem = f"{source['material']}_{source['snapshot']}"
    rng = np.random.default_rng(config["seed"] + index)
    multiplier = 6 if source["material"] == "Ta" else 1
    counts = [split["centers_per_Al_Mg_branch"] * multiplier for split in config["splits"].values()]
    selected = rng.choice(trajectory.atom_count, sum(counts), replace=False)
    radius = source["radius"] * config["cutoff_factor"]
    neighbors = config["candidate_neighbors"]
    shards = []
    offset = 0
    for (split_name, split), count in zip(config["splits"].items(), counts):
        started = time.monotonic()
        rows = selected[offset:offset+count]
        offset += count
        frames = np.arange(split["first_frame"], split["last_frame"]+1)
        shape = (count, len(frames))
        prefix = output / f"{stem}_{split_name}"
        clouds = open_memmap(str(prefix)+".clouds.npy", mode="w+", dtype=np.float32, shape=(*shape, 2, neighbors, 3))
        cross = open_memmap(str(prefix)+".cross.npy", mode="w+", dtype=np.float32, shape=(*shape, 3, 3))
        labels = np.empty(shape, dtype=np.int32)
        nonaffine = np.empty(shape, dtype=np.float32)
        support = np.empty(shape, dtype=np.int32)
        spatial_ids = np.empty(shape, dtype=np.int64)
        minimum_margin = float("inf")
        previous = None
        for t, frame in enumerate(frames):
            lengths = trajectory.box_high[frame] - trajectory.box_low[frame]
            points, tree = periodic_tree(trajectory.positions[frame], lengths)
            distances, ids = tree.query(points[rows], k=neighbors+2, workers=1)
            if not np.array_equal(ids[:, 0], rows):
                raise RuntimeError(f"Central atom query mismatch: {stem}/{frame}")
            margin = float(distances[:, -1].min() - radius)
            minimum_margin = min(minimum_margin, margin)
            if margin <= np.sqrt(3)*config["jitter_clip_A"]:
                raise RuntimeError(f"Neighbor capacity truncates possible contributions: {stem}/{frame}, margin={margin} A")
            current_ids = ids[:, 1:neighbors+1]
            anchor = local_offsets(points, lengths, rows, current_ids, radius)
            spatial = ids[np.arange(count), rng.integers(1, 9, count)]
            sd, si = tree.query(points[spatial], k=neighbors+2, workers=1)
            if sd[:, -1].min() <= radius:
                raise RuntimeError(f"Spatial radius is truncated: {stem}/{frame}")
            clouds[:, t, 0] = anchor
            clouds[:, t, 1] = local_offsets(points, lengths, spatial, si[:, 1:neighbors+1], radius)
            spatial_ids[:, t] = trajectory.atom_ids[spatial]
            support[:, t] = (distances[:, 1:neighbors+1] < radius).sum(1)
            labels[:, t] = ptm_labels(anchor, config["ptm_rmsd_cutoff"])
            if previous is None:
                past = anchor
            else:
                past = local_offsets(previous, previous_lengths, rows, current_ids, radius)
            r0, r1 = np.linalg.norm(past, axis=-1), np.linalg.norm(anchor, axis=-1)
            u0, u1 = np.clip((r0-.8)/.2, 0, 1), np.clip((r1-.8)/.2, 0, 1)
            weights = (1-u0**3*(10-u0*(15-6*u0))) * (1-u1**3*(10-u1*(15-6*u1)))
            x, y, w = past.astype(np.float64), anchor.astype(np.float64), weights.astype(np.float64)
            c = np.einsum("bni,bnj,bn->bij", y, x, w)
            xx = np.einsum("bni,bnj,bn->bij", x, x, w)
            affine = np.linalg.solve(xx, c.transpose(0, 2, 1)).transpose(0, 2, 1)
            residual = y - x @ affine.transpose(0, 2, 1)
            nonaffine[:, t] = (np.einsum("bni,bni,bn->b", residual, residual, w)/w.sum(1))*radius**2
            cross[:, t] = c / config["density"]["density_scale"]
            previous, previous_lengths = points, lengths
            if t % 20 == 0 or t == len(frames)-1:
                print(f"{stem}/{split_name}: frame {frame}, {t+1}/{len(frames)}", flush=True)
        clouds.flush()
        cross.flush()
        np.savez(str(prefix)+".metadata.npz", center_ids=trajectory.atom_ids[rows], frames=frames,
                 labels=labels, nonaffine_A2=nonaffine, support_counts=support, spatial_ids=spatial_ids)
        shard = dict(stem=f"{stem}_{split_name}", material=source["material"], snapshot=source["snapshot"],
                     split=split_name, centers=count, frames=len(frames), radius_A=radius,
                     minimum_candidate_margin_A=minimum_margin, maximum_support=int(support.max()),
                     source=source, elapsed_seconds=time.monotonic()-started)
        write_json(Path(str(prefix)+".json"), shard)
        shards.append(shard)
    return shards


def repair_labels(shard, config):
    os.environ["OVITO_THREAD_COUNT"] = "1"
    prefix = ROOT / config["output"] / "data" / shard["stem"]
    clouds = np.load(str(prefix)+".clouds.npy", mmap_mode="r")
    shape = clouds.shape[:2]
    flat = clouds.reshape(-1, 2, config["candidate_neighbors"], 3)
    labels = np.empty(len(flat), dtype=np.int32)
    for start in range(0, len(flat), 128):
        labels[start:start+128] = ptm_labels(flat[start:start+128, 0], config["ptm_rmsd_cutoff"])
    with np.load(str(prefix)+".metadata.npz") as saved:
        arrays = {key:saved[key] for key in saved.files}
    arrays["labels"] = labels.reshape(shape)
    np.savez(str(prefix)+".metadata.npz", **arrays)
    counts = {str(k):int(v) for k,v in zip(*np.unique(labels, return_counts=True))}
    print(f"PTM corrected: {shard['stem']}: {counts}", flush=True)
    return dict(stem=shard["stem"], counts=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--labels-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = ROOT / config["output"]
    if args.labels_only:
        import multiprocessing
        manifest = json.loads((output / "data/manifest.json").read_text())
        write_json(output / "labels_status.json", dict(state="running", pid=os.getpid()))
        try:
            with ProcessPoolExecutor(max_workers=config["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
                results = list(pool.map(repair_labels, manifest["shards"], [config]*len(manifest["shards"])))
            write_json(output / "labels_status.json", dict(state="complete", results=results,
                       correction="Removed only_selected=True, which excluded neighbors. Labels are evaluation-only and were not used for training."))
        except BaseException as error:
            write_json(output / "labels_status.json", dict(state="failed", error=repr(error), traceback=traceback.format_exc()))
            raise
        return
    (output / "data").mkdir(parents=True, exist_ok=False)
    sources = json.loads(Path(config["views_manifest"]).read_text())["sources"]
    status = dict(state="running", stage="prepare", started_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid())
    write_json(output / "prepare_status.json", status)
    started = time.monotonic()
    try:
        shards = []
        import multiprocessing
        with ProcessPoolExecutor(max_workers=config["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(prepare_branch, source, config, index) for index, source in enumerate(sources)]
            for future in as_completed(futures):
                shards.extend(future.result())
        manifest = dict(state="complete", shards=sorted(shards, key=lambda s:s["stem"]), config=config,
                        elapsed_seconds=time.monotonic()-started,
                        split_note="Disjoint center IDs and time blocks, shared source trajectories; no independent Ta validation.")
        write_json(output / "data/manifest.json", manifest)
        status.update(state="complete", elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        status.update(state="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(output / "prepare_status.json", status)


if __name__ == "__main__":
    main()
