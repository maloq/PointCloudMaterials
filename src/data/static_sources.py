"""Source access and cutoff operations shared by static data consumers."""

import bisect
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree


class ShardValueSequence(Sequence):
    """Per-sample metadata stored as constants for each source shard."""

    def __init__(self, values: Sequence[Any], counts: Sequence[int]) -> None:
        self._values = list(values)
        self._counts = [int(v) for v in counts]
        self._cumulative = np.cumsum(self._counts, dtype=np.int64).tolist()
        self._length = int(self._cumulative[-1]) if self._cumulative else 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            return [self[i] for i in range(start, stop, step)]
        idx = int(index)
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(
                f"Index {index} out of range for sequence "
                f"length {self._length}."
            )
        shard_idx = bisect.bisect_right(self._cumulative, idx)
        return self._values[shard_idx]


def read_off_file(filename: str, verbose=True, cache=True) -> np.ndarray:
    """Read points from OFF file and return as numpy array.

    Optionally caches the file on disk in a faster .npy format.

    Args:
        filename: Path to the OFF file.
        verbose: If True, prints additional information.
        cache: If True, will attempt to load a cached npy file if available,
               and will save to cache after parsing.

    Returns:
        A numpy array of point coordinates (shape: [N, 3]).
    """
    if cache:
        base, _ = os.path.splitext(filename)
        cache_filename = base + '.npy'
        if os.path.exists(cache_filename):
            if verbose:
                print(f"Loading cached file from {cache_filename}")
            points = np.load(cache_filename)
            return points

    # Read the OFF file
    with open(filename, 'r') as f:
        # Read and verify OFF header
        header = f.readline().strip()
        if header != 'OFF':
            raise ValueError("Invalid OFF file format")
        n_vertices, n_faces, n_edges = map(int, f.readline().split())
        points = []
        for _ in range(n_vertices):
            x, y, z = map(float, f.readline().split())
            points.append([x, y, z])

    points = np.array(points)

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    space_size = max_coords - min_coords
    if verbose:
        logger.print(f"Read {len(points)} points")
        logger.print(f"Size of space: {space_size}")
        logger.print(f"Min coords: {min_coords}")
        logger.print(f"Max coords: {max_coords}")

    # Cache the data to disk for faster future loading
    if cache:
        if verbose:
            print(f"Caching file to disk at {cache_filename}")
        np.save(cache_filename, points)
    return points


def load_points(filepath: str) -> np.ndarray:
    """Read an NPY or OFF point cloud as a float32 (N, 3) array."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.npy':
        points = np.load(filepath)
    elif ext == '.off':
        points = read_off_file(filepath, verbose=False)
    else:
        raise ValueError(
            f"Unsupported file extension {ext!r} for {filepath}. "
            "Use .npy or .off"
        )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected (N, 3) array from {filepath}, got shape {points.shape}"
        )
    return points.astype(np.float32, copy=False)


def resolve_sources(
    root: str,
    data_files: list[str] | None,
    data_sources: list[dict] | None,
) -> list[dict[str, Any]]:
    """Return list of source descriptors."""
    if data_sources:
        sources = []
        used_names: set[str] = set()
        for source_index, src in enumerate(data_sources):
            src_path = src["data_path"]
            src_files = src["data_files"]
            source_name_raw = src.get("name", None)
            source_name = (
                str(source_name_raw)
                if source_name_raw is not None
                else (Path(str(src_path)).name or f"source_{source_index}")
            )
            if source_name in used_names:
                source_name = f"{source_name}_{source_index}"
            candidate = source_name
            suffix = 1
            while source_name in used_names:
                source_name = f"{candidate}_{suffix}"
                suffix += 1
            used_names.add(source_name)
            source_max_samples = src.get("max_samples", None)
            if source_max_samples is not None:
                source_max_samples = int(source_max_samples)
                if source_max_samples <= 0:
                    raise ValueError(
                        f"data_sources[{source_index}].max_samples "
                        "must be > 0 when set, "
                        f"got {src.get('max_samples')!r}."
                    )
            sources.append(
                {
                    "index": int(source_index),
                    "name": source_name,
                    "root": str(src_path),
                    "files": list(src_files),
                    "radius_override": src.get("radius", None),
                    "max_samples": source_max_samples,
                }
            )
        return sources

    source_name = Path(str(root)).name if str(root) else "single_source"
    return [
        {
            "index": 0,
            "name": source_name or "single_source",
            "root": str(root),
            "files": list(data_files),
            "radius_override": None,
            "max_samples": None,
        }
    ]


def resolve_auto_cutoff_config(
    auto_cutoff_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if auto_cutoff_config is None or not auto_cutoff_config["enabled"]:
        return None
    resolved = {
        "target_points": int(auto_cutoff_config["target_points"]),
        "quantile": float(auto_cutoff_config["quantile"]),
        "estimation_samples_per_file": int(
            auto_cutoff_config["estimation_samples_per_file"]
        ),
        "seed": int(auto_cutoff_config["seed"]),
        "safety_factor": float(auto_cutoff_config["safety_factor"]),
        "boundary_margin": auto_cutoff_config["boundary_margin"],
        "reference_frame_index": (
            int(auto_cutoff_config["reference_frame_index"])
            if "reference_frame_index" in auto_cutoff_config
            else None
        ),
    }
    return resolved


def estimate_source_cutoff_radius(
    *,
    source_root: str,
    source_files: list[str],
    target_points: int,
    quantile: float,
    estimation_samples_per_file: int,
    seed: int,
    safety_factor: float,
    boundary_margin: float | None,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    kth_distances_all: list[np.ndarray] = []

    for file_name in source_files:
        filepath = os.path.join(source_root, file_name)
        points = load_points(filepath)
        num_atoms = len(points)
        candidate_indices = np.arange(num_atoms, dtype=np.int64)
        if boundary_margin is not None and boundary_margin > 0.0:
            boundary_margin = float(boundary_margin)
            min_coords = points.min(axis=0)
            max_coords = points.max(axis=0)
            interior_mask = np.all(
                (points >= (min_coords + boundary_margin))
                & (points <= (max_coords - boundary_margin)),
                axis=1,
            )
            interior_indices = np.flatnonzero(interior_mask)
            if interior_indices.size > 0:
                candidate_indices = interior_indices.astype(
                    np.int64, copy=False
                )

        centers_to_sample = min(
            estimation_samples_per_file, int(candidate_indices.size)
        )
        center_indices = rng.choice(
            candidate_indices, size=centers_to_sample, replace=False
        )

        tree = cKDTree(points)
        k = min(int(target_points), num_atoms)
        dists, _ = tree.query(points[center_indices], k=k)
        dists = np.asarray(dists, dtype=np.float64)
        if k == 1:
            kth_dist = dists.reshape(-1)
        else:
            kth_dist = dists[:, k - 1]
        kth_distances_all.append(kth_dist)

    kth_all = np.concatenate(kth_distances_all).astype(np.float64, copy=False)
    estimated_radius = float(np.quantile(kth_all, quantile)) * float(
        safety_factor
    )

    coverage = float(np.mean(kth_all <= estimated_radius))
    return estimated_radius, coverage
