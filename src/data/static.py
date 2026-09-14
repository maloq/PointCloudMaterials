"""Static point-cloud datasets and their verified sample caches."""

import bisect
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.sampling import (
    get_random_samples,
    get_regular_samples,
    pc_normalize,
)
from src.data.static_sources import (
    ShardValueSequence,
    estimate_source_cutoff_radius,
    load_points,
    resolve_auto_cutoff_config,
    resolve_sources,
)
from src.utils.logging_config import setup_logging

logger = setup_logging()
_STATIC_SAMPLE_CACHE_VERSION = 1


def _split_source_sample_limit(source_max_samples: int | None, n_files: int) -> list[int | None]:
    if source_max_samples is None:
        return [None] * int(n_files)
    limit = int(source_max_samples)
    if limit <= 0:
        raise ValueError(f"source max_samples must be > 0 when set, got {source_max_samples!r}.")
    if n_files <= 0:
        raise ValueError(f"Cannot split source max_samples={limit} across n_files={n_files}.")
    base = limit // int(n_files)
    remainder = limit % int(n_files)
    return [base + (1 if file_idx < remainder else 0) for file_idx in range(int(n_files))]


def _safe_cache_stem(*parts: Any) -> str:
    raw = "__".join(str(part) for part in parts)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw).strip("_")
    return safe or "shard"


def _sample_single_file(
    filepath,
    sample_type,
    radius,
    n_points,
    overlap_fraction,
    n_samples,
    return_coords,
    sampling_method,
    drop_edge_samples=True,
    edge_drop_layers: int | None = None,
    max_samples: int | None = None,
):
    """Load one file and generate samples. Standalone function for multiprocessing."""
    points = load_points(filepath)
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            return []
    if sample_type == 'regular':
        regular_max_samples = max_samples if max_samples is not None else int(2e32)
        return get_regular_samples(
            points, size=radius, n_points=n_points,
            overlap_fraction=overlap_fraction,
            return_coords=return_coords,
            max_samples=regular_max_samples,
            drop_edge_samples=bool(drop_edge_samples),
            edge_drop_layers=edge_drop_layers,
            sampling_method=sampling_method,
        )
    elif sample_type == 'random':
        random_n_samples = max_samples if max_samples is not None else int(n_samples)
        return get_random_samples(
            points, n_samples=random_n_samples, size=radius,
            n_points=n_points, return_coords=return_coords,
            sampling_method=sampling_method,
        )
    else:
        raise ValueError(f"Invalid sample type: {sample_type!r}")


def read_and_sample_files(
    root,
    data_files,
    radius,
    n_points,
    overlap_fraction,
    sample_type,
    n_samples,
    return_coords,
    sampling_method="drop_farthest",
    drop_edge_samples=True,
    edge_drop_layers: int | None = None,
    source_max_samples: int | None = None,
):
    """Read point cloud files and sample sub-clouds. Processes files in parallel when possible."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    filepaths = [os.path.join(root, f) for f in data_files]
    n_files = len(filepaths)
    max_workers = min(n_files, max(1, multiprocessing.cpu_count() // 2))
    file_limits = _split_source_sample_limit(source_max_samples, n_files)
    args = (
        sample_type,
        radius,
        n_points,
        overlap_fraction,
        n_samples,
        return_coords,
        sampling_method,
        drop_edge_samples,
        edge_drop_layers,
    )

    all_samples: list = []

    if n_files > 1 and max_workers > 1:
        logger.info(f"Processing {n_files} files in parallel with {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_sample_single_file, fp, *args, max_samples=file_limit)
                for fp, file_limit in zip(filepaths, file_limits)
            ]
            for future in futures:
                file_samples = future.result()
                all_samples.extend(file_samples)
    else:
        for fp, file_limit in zip(filepaths, file_limits):
            file_samples = _sample_single_file(fp, *args, max_samples=file_limit)
            all_samples.extend(file_samples)

    if not all_samples:
        raise ValueError(f"No samples found for {data_files} in {root}")
    if source_max_samples is not None and len(all_samples) > int(source_max_samples):
        all_samples = all_samples[: int(source_max_samples)]
    return all_samples


class PointCloudDataset(Dataset):
    def __init__(self,
                 root: str = "",
                 data_files: list[str] | None = None,
                 data_sources: list[dict] | None = None,
                 return_coords=False,
                 sample_type='regular',
                 radius=8,
                 overlap_fraction=0.0,
                 n_samples=1000,
                 num_points=100,
                 drop_edge_samples=True,
                 edge_drop_layers: int | None = None,
                 pre_normalize=True,
                 normalize=True,
                 sampling_method="drop_farthest",
                 auto_cutoff_config: dict[str, Any] | None = None,
                 sample_cache_config: dict[str, Any] | None = None,
                 atomic_context: dict[str, Any] | None = None):
        """Initialize the dataset with samples from point cloud files.

        Supports the two repository configuration modes:

        1. **Single source**: provide ``root`` + ``data_files``.
        2. **Multi source**: provide ``data_sources``, a list of dicts each with
           ``data_path`` and ``data_files`` keys.  Samples from all sources are
           concatenated into a single dataset.

        Args:
            root: Directory containing data files (single-source mode).
            data_files: List of filenames relative to *root* (single-source mode).
            data_sources: List of ``{"data_path": str, "data_files": [str, ...]}``
                dicts (multi-source mode).  When provided, *root* and *data_files*
                are ignored.
        """
        self.pre_normalize = pre_normalize
        self.normalize = normalize
        self.return_coords = return_coords
        self.sample_type = sample_type
        self.sampling_method = sampling_method
        self.radius = float(radius)
        self.num_points = int(num_points)
        self.drop_edge_samples = bool(drop_edge_samples)
        self.edge_drop_layers = edge_drop_layers

        auto_cfg = resolve_auto_cutoff_config(
            auto_cutoff_config,
        )

        sources = resolve_sources(root, data_files, data_sources)
        self.source_radii: dict[str, float] = {}
        self.sample_source_names: list[str] = []
        self.sample_radii: list[float] | ShardValueSequence = []
        self._cache_sample_arrays: list[np.ndarray] | None = None
        self._cache_coord_arrays: list[np.ndarray | None] | None = None
        self._cache_cumulative_counts: list[int] = []
        self._cache_total_samples = 0

        for source in sources:
            src_radius = self._resolve_source_cutoff_radius(
                source=source,
                default_radius=self.radius,
                auto_cutoff_config=auto_cfg,
                num_points=self.num_points,
            )
            source["resolved_radius"] = float(src_radius)
            self.source_radii[str(source["name"])] = float(src_radius)

        cache_cfg = self._resolve_sample_cache_config(sample_cache_config)
        if cache_cfg["enabled"]:
            cache_request = self._build_sample_cache_request(
                sources=sources,
                overlap_fraction=overlap_fraction,
                sample_type=sample_type,
                n_samples=n_samples,
                return_coords=return_coords,
                normalize=normalize,
            )
            self._load_or_build_sample_cache(
                cache_cfg=cache_cfg,
                cache_request=cache_request,
                sources=sources,
                overlap_fraction=overlap_fraction,
                n_samples=n_samples,
            )
            if atomic_context is not None:
                from src.data.atomic_context import attach_atomic_context
                attach_atomic_context(self, atomic_context, cache_cfg["cache_dir"])
            return

        if atomic_context is not None:
            raise ValueError("atomic_context requires an existing static sample cache with coordinates")
        all_sample_radii: list[float] = []
        all_samples: list = []

        for source in sources:
            src_name = source["name"]
            src_root = source["root"]
            src_files = source["files"]
            src_radius = float(source["resolved_radius"])

            samples = read_and_sample_files(
                src_root, src_files, src_radius, self.num_points,
                overlap_fraction, sample_type, n_samples,
                return_coords, sampling_method,
                self.drop_edge_samples, self.edge_drop_layers,
                source_max_samples=source["max_samples"],
            )
            all_samples.extend(samples)
            all_sample_radii.extend([src_radius] * len(samples))
            self.sample_source_names.extend([src_name] * len(samples))

        if not all_samples:
            raise ValueError("No samples generated from any data source")

        self.samples = all_samples
        self.sample_radii = all_sample_radii

        if self.return_coords:
            self.samples, self.coords = zip(*self.samples)
            self.samples = list(self.samples)
            self.coords = list(self.coords)
        else:
            self.coords = None

        if self.pre_normalize and self.normalize:
            self.samples = [
                pc_normalize(sample, sample_radius).astype(np.float32)
                for sample, sample_radius in zip(self.samples, self.sample_radii)
            ]
        elif not self.normalize:
            print("Point Cloud normalization skipped")
        logger.info(f"Point set shape: {self.samples[0].shape}")
        if len(self.source_radii) > 1:
            formatted = ", ".join(
                f"{name}: {radius_val:.4f}"
                for name, radius_val in sorted(self.source_radii.items(), key=lambda kv: kv[0])
            )
            logger.print(f"Per-source cutoff radii: {formatted}")

    @staticmethod
    def _resolve_sample_cache_config(
        sample_cache_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if sample_cache_config is None or not sample_cache_config["enabled"]:
            return {"enabled": False}
        cache_dir = sample_cache_config["cache_dir"]
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError(
                "data.sample_cache.cache_dir is required when data.sample_cache.enabled=true."
            )
        return {
            "enabled": True,
            "cache_dir": str(cache_dir),
            "rebuild": bool(sample_cache_config["rebuild"]),
            "local_cache_dir": sample_cache_config["local_cache_dir"],
        }

    def _build_sample_cache_request(
        self,
        *,
        sources: list[dict[str, Any]],
        overlap_fraction: float,
        sample_type: str,
        n_samples: int,
        return_coords: bool,
        normalize: bool,
    ) -> dict[str, Any]:
        source_entries: list[dict[str, Any]] = []
        for source in sources:
            file_entries = []
            for file_name in source["files"]:
                file_path = Path(str(source["root"])) / str(file_name)
                if not file_path.exists():
                    raise FileNotFoundError(
                        "Cannot build static sample-cache request because a source file is missing: "
                        f"source={source['name']!r}, path={file_path}."
                    )
                stat = file_path.stat()
                file_entries.append(
                    {
                        "name": str(file_name),
                        "path": str(file_path),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )
            source_entries.append(
                {
                    "name": str(source["name"]),
                    "root": str(source["root"]),
                    "radius": float(source["resolved_radius"]),
                    "max_samples": source["max_samples"],
                    "files": file_entries,
                }
            )

        return {
            "schema_version": _STATIC_SAMPLE_CACHE_VERSION,
            "sample_type": str(sample_type),
            "num_points": int(self.num_points),
            "overlap_fraction": float(overlap_fraction),
            "n_samples": int(n_samples) if str(sample_type) == "random" else None,
            "return_coords": bool(return_coords),
            "normalize": bool(normalize),
            "drop_edge_samples": bool(self.drop_edge_samples),
            "edge_drop_layers": self.edge_drop_layers,
            "sampling_method": str(self.sampling_method),
            "sources": source_entries,
        }

    @staticmethod
    def _sample_cache_fingerprint(cache_request: dict[str, Any]) -> str:
        payload = json.dumps(cache_request, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_or_build_sample_cache(
        self,
        *,
        cache_cfg: dict[str, Any],
        cache_request: dict[str, Any],
        sources: list[dict[str, Any]],
        overlap_fraction: float,
        n_samples: int,
    ) -> None:
        cache_dir = Path(cache_cfg["cache_dir"])
        metadata_path = cache_dir / "metadata.json"
        building_marker = cache_dir / "BUILDING"
        expected_fingerprint = self._sample_cache_fingerprint(cache_request)

        if building_marker.exists():
            if not cache_cfg["rebuild"]:
                raise RuntimeError(
                    "Static sample cache appears incomplete from an interrupted build. "
                    f"cache_dir={cache_dir}. Set data.sample_cache.rebuild=true to delete "
                    "the incomplete cache and rebuild it."
                )
            self._remove_cache_dir_for_rebuild(cache_dir)

        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            observed_fingerprint = metadata["fingerprint"]
            if observed_fingerprint == expected_fingerprint:
                load_dir, load_metadata = self._stage_sample_cache_if_configured(
                    source_cache_dir=cache_dir,
                    metadata=metadata,
                    cache_cfg=cache_cfg,
                )
                self._load_sample_cache_from_metadata(
                    cache_dir=load_dir,
                    metadata=load_metadata,
                )
                return
            if not cache_cfg["rebuild"]:
                raise RuntimeError(
                    "Static sample cache metadata does not match the requested dataset. "
                    f"cache_dir={cache_dir}, expected_fingerprint={expected_fingerprint}, "
                    f"observed_fingerprint={observed_fingerprint}. Set "
                    "data.sample_cache.rebuild=true to overwrite this cache in place."
                )
            if cache_dir.exists():
                self._remove_cache_dir_for_rebuild(cache_dir)
        elif cache_dir.exists() and any(cache_dir.iterdir()):
            if not cache_cfg["rebuild"]:
                raise RuntimeError(
                    "Static sample cache directory exists but metadata.json is missing. "
                    f"cache_dir={cache_dir}. Set data.sample_cache.rebuild=true to delete "
                    "the incomplete cache and rebuild it."
                )
            self._remove_cache_dir_for_rebuild(cache_dir)

        self._build_sample_cache(
            cache_dir=cache_dir,
            cache_request=cache_request,
            fingerprint=expected_fingerprint,
            sources=sources,
            overlap_fraction=overlap_fraction,
            n_samples=n_samples,
        )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        load_dir, load_metadata = self._stage_sample_cache_if_configured(
            source_cache_dir=cache_dir,
            metadata=metadata,
            cache_cfg=cache_cfg,
        )
        self._load_sample_cache_from_metadata(cache_dir=load_dir, metadata=load_metadata)

    @staticmethod
    def _cache_copy_matches(
        *,
        source_cache_dir: Path,
        staged_cache_dir: Path,
        metadata: dict[str, Any],
    ) -> tuple[bool, str]:
        staged_metadata_path = staged_cache_dir / "metadata.json"
        if not staged_metadata_path.exists():
            return False, f"metadata is missing: {staged_metadata_path}"
        try:
            with staged_metadata_path.open("r", encoding="utf-8") as handle:
                staged_metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"metadata is unreadable: {staged_metadata_path}: {exc}"
        if staged_metadata.get("fingerprint") != metadata.get("fingerprint"):
            return False, "metadata fingerprint differs from the source cache"

        relative_paths = ["metadata.json"]
        for shard in metadata.get("shards", []):
            relative_paths.append(str(shard["samples_path"]))
            coords_path = shard.get("coords_path")
            if coords_path is not None:
                relative_paths.append(str(coords_path))
        for relative_path in relative_paths:
            source_path = source_cache_dir / relative_path
            staged_path = staged_cache_dir / relative_path
            if not source_path.is_file():
                return False, f"source cache file is missing: {source_path}"
            if not staged_path.is_file():
                return False, f"staged cache file is missing: {staged_path}"
            if source_path.stat().st_size != staged_path.stat().st_size:
                return False, (
                    f"file size differs for {relative_path}: "
                    f"source={source_path.stat().st_size}, staged={staged_path.stat().st_size}"
                )
        return True, "cache fingerprint and file sizes match"

    def _stage_sample_cache_if_configured(
        self,
        *,
        source_cache_dir: Path,
        metadata: dict[str, Any],
        cache_cfg: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        local_cache_dir_raw = cache_cfg.get("local_cache_dir", None)
        if local_cache_dir_raw is None or not str(local_cache_dir_raw).strip():
            return source_cache_dir, metadata

        staged_cache_dir = Path(str(local_cache_dir_raw)).expanduser()
        if not staged_cache_dir.is_absolute():
            raise ValueError(
                "data.sample_cache.local_cache_dir must be an absolute path, "
                f"got {local_cache_dir_raw!r}."
            )
        if staged_cache_dir.resolve() == source_cache_dir.resolve():
            raise ValueError(
                "data.sample_cache.local_cache_dir must differ from cache_dir, "
                f"both resolve to {staged_cache_dir.resolve()}."
            )

        stage_parent = staged_cache_dir.parent
        stage_parent.mkdir(parents=True, exist_ok=True)
        lock_path = stage_parent / f".{staged_cache_dir.name}.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

            if staged_cache_dir.exists():
                matches, reason = self._cache_copy_matches(
                    source_cache_dir=source_cache_dir,
                    staged_cache_dir=staged_cache_dir,
                    metadata=metadata,
                )
                if matches:
                    logger.print(
                        f"[sample_cache] using node-local staged cache at {staged_cache_dir}."
                    )
                    with (staged_cache_dir / "metadata.json").open("r", encoding="utf-8") as handle:
                        return staged_cache_dir, json.load(handle)
                logger.print(
                    "[sample_cache] replacing invalid node-local staged cache: "
                    f"path={staged_cache_dir}, reason={reason}."
                )
                self._remove_cache_dir_for_rebuild(staged_cache_dir)

            source_bytes = sum(
                path.stat().st_size
                for path in source_cache_dir.rglob("*")
                if path.is_file()
            )
            available_bytes = shutil.disk_usage(stage_parent).free
            if available_bytes < source_bytes:
                raise RuntimeError(
                    "Insufficient space for node-local static sample cache. "
                    f"source={source_cache_dir}, destination={staged_cache_dir}, "
                    f"required_bytes={source_bytes}, available_bytes={available_bytes}."
                )

            staging_dir = stage_parent / f".{staged_cache_dir.name}.staging-{os.getpid()}"
            if staging_dir.exists():
                self._remove_cache_dir_for_rebuild(staging_dir)
            logger.print(
                "[sample_cache] staging cache to node-local storage: "
                f"source={source_cache_dir}, destination={staged_cache_dir}, "
                f"bytes={source_bytes}."
            )
            try:
                shutil.copytree(source_cache_dir, staging_dir, copy_function=shutil.copy2)
                matches, reason = self._cache_copy_matches(
                    source_cache_dir=source_cache_dir,
                    staged_cache_dir=staging_dir,
                    metadata=metadata,
                )
                if not matches:
                    raise RuntimeError(
                        "Node-local static sample cache validation failed after copy. "
                        f"source={source_cache_dir}, staging_dir={staging_dir}, reason={reason}."
                    )
                staging_dir.rename(staged_cache_dir)
            except Exception:
                if staging_dir.exists():
                    self._remove_cache_dir_for_rebuild(staging_dir)
                raise

            logger.print(f"[sample_cache] staged node-local cache at {staged_cache_dir}.")
            with (staged_cache_dir / "metadata.json").open("r", encoding="utf-8") as handle:
                return staged_cache_dir, json.load(handle)

    @staticmethod
    def _remove_cache_dir_for_rebuild(cache_dir: Path) -> None:
        resolved = cache_dir.resolve()
        forbidden = {Path("/").resolve(), Path.cwd().resolve(), Path.home().resolve()}
        if resolved in forbidden:
            raise RuntimeError(
                "Refusing to delete unsafe sample cache directory during rebuild: "
                f"{resolved}."
            )
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    def _build_sample_cache(
        self,
        *,
        cache_dir: Path,
        cache_request: dict[str, Any],
        fingerprint: str,
        sources: list[dict[str, Any]],
        overlap_fraction: float,
        n_samples: int,
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        shards_dir = cache_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        building_marker = cache_dir / "BUILDING"
        building_marker.write_text(
            "Static sample cache build in progress. Delete by running with "
            "data.sample_cache.rebuild=true.\n",
            encoding="utf-8",
        )

        metadata: dict[str, Any] = {
            "schema_version": _STATIC_SAMPLE_CACHE_VERSION,
            "fingerprint": fingerprint,
            "request": cache_request,
            "shards": [],
            "source_counts": {},
            "total_samples": 0,
        }

        total_samples = 0
        for source in sources:
            source_name = str(source["name"])
            source_root = str(source["root"])
            source_radius = float(source["resolved_radius"])
            source_files = list(source["files"])
            file_limits = _split_source_sample_limit(source["max_samples"], len(source_files))
            remaining = source["max_samples"]
            remaining = None if remaining is None else int(remaining)
            source_count = 0

            for file_index, (file_name, file_limit) in enumerate(zip(source_files, file_limits)):
                if remaining is not None and remaining <= 0:
                    break
                effective_limit = file_limit
                if remaining is not None:
                    effective_limit = min(int(effective_limit), remaining)
                filepath = os.path.join(source_root, str(file_name))
                logger.print(
                    "[sample_cache] building shard "
                    f"source={source_name!r}, file={file_name!r}, "
                    f"radius={source_radius:.6f}, max_samples={effective_limit}."
                )
                samples = _sample_single_file(
                    filepath,
                    self.sample_type,
                    source_radius,
                    self.num_points,
                    overlap_fraction,
                    n_samples,
                    self.return_coords,
                    self.sampling_method,
                    self.drop_edge_samples,
                    self.edge_drop_layers,
                    max_samples=effective_limit,
                )
                if remaining is not None and len(samples) > remaining:
                    samples = samples[:remaining]
                if not samples:
                    raise RuntimeError(
                        "Static sample cache shard produced zero samples. "
                        f"source={source_name!r}, file={file_name!r}, path={filepath}, "
                        f"radius={source_radius}, max_samples={effective_limit}."
                    )

                shard_stem = _safe_cache_stem(source_name, file_index, Path(str(file_name)).stem)
                sample_relpath = f"shards/{shard_stem}.samples.npy"
                coords_relpath = f"shards/{shard_stem}.coords.npy" if self.return_coords else None
                sample_path = cache_dir / sample_relpath
                coords_path = cache_dir / coords_relpath if coords_relpath is not None else None
                shard_count = self._write_sample_cache_shard(
                    samples=samples,
                    sample_path=sample_path,
                    coords_path=coords_path,
                    radius=source_radius,
                    source_name=source_name,
                    file_name=str(file_name),
                )
                del samples

                metadata["shards"].append(
                    {
                        "source": source_name,
                        "file": str(file_name),
                        "samples_path": sample_relpath,
                        "coords_path": coords_relpath,
                        "count": int(shard_count),
                        "radius": source_radius,
                    }
                )
                source_count += int(shard_count)
                total_samples += int(shard_count)
                if remaining is not None:
                    remaining -= int(shard_count)

            if source_count <= 0:
                raise RuntimeError(
                    "Static sample cache source produced zero samples. "
                    f"source={source_name!r}, root={source_root}, files={source_files}."
                )
            metadata["source_counts"][source_name] = int(source_count)

        if total_samples <= 0:
            raise RuntimeError(f"Static sample cache build produced zero samples in {cache_dir}.")

        metadata["total_samples"] = int(total_samples)
        metadata_path = cache_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        building_marker.unlink()
        logger.print(
            f"[sample_cache] built {total_samples} ready-to-train samples at {cache_dir}."
        )

    def _write_sample_cache_shard(
        self,
        *,
        samples: list,
        sample_path: Path,
        coords_path: Path | None,
        radius: float,
        source_name: str,
        file_name: str,
    ) -> int:
        sample_count = len(samples)
        sample_array = np.lib.format.open_memmap(
            sample_path,
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, self.num_points, 3),
        )
        coords_array = None
        if self.return_coords:
            if coords_path is None:
                raise RuntimeError(
                    "coords_path must be provided when return_coords=True while writing sample cache."
                )
            coords_array = np.lib.format.open_memmap(
                coords_path,
                mode="w+",
                dtype=np.float32,
                shape=(sample_count, 3),
            )

        for sample_index, item in enumerate(samples):
            if self.return_coords:
                sample_points, coords = item
            else:
                sample_points, coords = item, None
            sample_np = np.asarray(sample_points, dtype=np.float32)
            if sample_np.shape != (self.num_points, 3):
                raise ValueError(
                    "Static sample cache expected each sample to have shape "
                    f"({self.num_points}, 3), got {sample_np.shape}. "
                    f"source={source_name!r}, file={file_name!r}, sample_index={sample_index}."
                )
            if self.normalize:
                sample_np = pc_normalize(sample_np, radius).astype(np.float32, copy=False)
            sample_array[sample_index] = sample_np
            if coords_array is not None:
                coords_np = np.asarray(coords, dtype=np.float32)
                if coords_np.shape != (3,):
                    raise ValueError(
                        "Static sample cache expected coords to have shape (3,), "
                        f"got {coords_np.shape}. source={source_name!r}, "
                        f"file={file_name!r}, sample_index={sample_index}."
                    )
                coords_array[sample_index] = coords_np

        sample_array.flush()
        del sample_array
        if coords_array is not None:
            coords_array.flush()
            del coords_array
        return int(sample_count)

    def _load_sample_cache_from_metadata(
        self,
        *,
        cache_dir: Path,
        metadata: dict[str, Any],
    ) -> None:
        schema_version = int(metadata["schema_version"])
        if schema_version != _STATIC_SAMPLE_CACHE_VERSION:
            raise RuntimeError(
                "Unsupported static sample cache schema version. "
                f"cache_dir={cache_dir}, schema_version={schema_version}, "
                f"expected={_STATIC_SAMPLE_CACHE_VERSION}."
            )

        request = metadata["request"]
        cached_num_points = int(request["num_points"])
        if cached_num_points != int(self.num_points):
            raise RuntimeError(
                "Static sample cache num_points mismatch after metadata validation. "
                f"cache_dir={cache_dir}, cached={cached_num_points}, requested={self.num_points}."
            )
        cached_return_coords = bool(request["return_coords"])
        if cached_return_coords != bool(self.return_coords):
            raise RuntimeError(
                "Static sample cache return_coords mismatch after metadata validation. "
                f"cache_dir={cache_dir}, cached={cached_return_coords}, "
                f"requested={self.return_coords}."
            )

        sample_arrays: list[np.ndarray] = []
        coord_arrays: list[np.ndarray | None] = []
        counts: list[int] = []
        source_names: list[str] = []
        radii: list[float] = []

        for shard in metadata["shards"]:
            count = int(shard["count"])
            sample_path = cache_dir / shard["samples_path"]
            if not sample_path.exists():
                raise FileNotFoundError(
                    "Static sample cache metadata references a missing samples shard: "
                    f"{sample_path}."
                )
            sample_array = np.load(sample_path, mmap_mode="r")
            expected_shape = (count, int(self.num_points), 3)
            if tuple(sample_array.shape) != expected_shape:
                raise RuntimeError(
                    "Static sample cache samples shard has unexpected shape. "
                    f"path={sample_path}, expected={expected_shape}, got={sample_array.shape}."
                )
            if sample_array.dtype != np.float32:
                raise RuntimeError(
                    "Static sample cache samples shard must be float32. "
                    f"path={sample_path}, dtype={sample_array.dtype}."
                )

            coord_array = None
            coords_relpath = shard["coords_path"]
            if self.return_coords:
                if coords_relpath is None:
                    raise RuntimeError(
                        "Static sample cache is missing coords_path for a return_coords=True request. "
                        f"sample_path={sample_path}."
                    )
                coords_path = cache_dir / coords_relpath
                if not coords_path.exists():
                    raise FileNotFoundError(
                        "Static sample cache metadata references a missing coords shard: "
                        f"{coords_path}."
                    )
                coord_array = np.load(coords_path, mmap_mode="r")
                expected_coords_shape = (count, 3)
                if tuple(coord_array.shape) != expected_coords_shape:
                    raise RuntimeError(
                        "Static sample cache coords shard has unexpected shape. "
                        f"path={coords_path}, expected={expected_coords_shape}, "
                        f"got={coord_array.shape}."
                    )
                if coord_array.dtype != np.float32:
                    raise RuntimeError(
                        "Static sample cache coords shard must be float32. "
                        f"path={coords_path}, dtype={coord_array.dtype}."
                    )

            sample_arrays.append(sample_array)
            coord_arrays.append(coord_array)
            counts.append(count)
            source_names.append(str(shard["source"]))
            radii.append(float(shard["radius"]))

        total_samples = int(metadata["total_samples"])
        if total_samples <= 0 or total_samples != int(sum(counts)):
            raise RuntimeError(
                "Static sample cache total_samples is invalid. "
                f"cache_dir={cache_dir}, total_samples={total_samples}, "
                f"sum_shards={int(sum(counts))}."
            )

        self._cache_sample_arrays = sample_arrays
        self._cache_coord_arrays = coord_arrays
        self._cache_cumulative_counts = np.cumsum(counts, dtype=np.int64).tolist()
        self._cache_total_samples = total_samples
        self.samples = None
        self.coords = None
        self.sample_source_names = ShardValueSequence(source_names, counts)
        self.sample_radii = ShardValueSequence(radii, counts)
        logger.info(f"Point set shape: {sample_arrays[0].shape[1:]}")
        logger.print(
            f"[sample_cache] loaded {total_samples} ready-to-train samples from {cache_dir}."
        )
        if len(self.source_radii) > 1:
            formatted = ", ".join(
                f"{name}: {radius_val:.4f}"
                for name, radius_val in sorted(self.source_radii.items(), key=lambda kv: kv[0])
            )
            logger.print(f"Per-source cutoff radii: {formatted}")

    def _resolve_source_cutoff_radius(
        self,
        *,
        source: dict[str, Any],
        default_radius: float,
        auto_cutoff_config: dict[str, Any] | None,
        num_points: int,
    ) -> float:
        source_name = str(source["name"])
        source_root = str(source["root"])
        source_files = source["files"]
        radius_override = source["radius_override"]

        if radius_override is not None:
            return float(radius_override)

        if auto_cutoff_config is None:
            return float(default_radius)

        target_points = max(int(auto_cutoff_config["target_points"]), int(num_points))

        seed = int(auto_cutoff_config["seed"]) + int(source["index"])
        estimated_radius, coverage = estimate_source_cutoff_radius(
            source_root=source_root,
            source_files=source_files,
            target_points=target_points,
            quantile=float(auto_cutoff_config["quantile"]),
            estimation_samples_per_file=int(auto_cutoff_config["estimation_samples_per_file"]),
            seed=seed,
            safety_factor=float(auto_cutoff_config["safety_factor"]),
            boundary_margin=auto_cutoff_config["boundary_margin"],
        )
        logger.print(
            "[auto_cutoff] "
            f"source={source_name!r}, target_points={target_points}, "
            f"quantile={float(auto_cutoff_config['quantile']):.4f}, "
            f"coverage~{coverage * 100.0:.2f}%, "
            f"radius={estimated_radius:.4f} (default={default_radius:.4f})."
        )
        return estimated_radius

    def __len__(self):
        if self._cache_sample_arrays is not None:
            return int(self._cache_total_samples)
        return len(self.samples)

    def _resolve_cache_index(self, index: int) -> tuple[int, int]:
        idx = int(index)
        total = int(self._cache_total_samples)
        if idx < 0:
            idx += total
        if idx < 0 or idx >= total:
            raise IndexError(f"PointCloudDataset cache index {index} out of range for length {total}.")
        shard_idx = bisect.bisect_right(self._cache_cumulative_counts, idx)
        previous = 0 if shard_idx == 0 else int(self._cache_cumulative_counts[shard_idx - 1])
        return shard_idx, idx - previous

    def __getitem__(self, index):
        if self._cache_sample_arrays is not None:
            shard_idx, local_idx = self._resolve_cache_index(index)
            point_set = self._cache_sample_arrays[shard_idx][local_idx]
            sample = {"points": torch.tensor(point_set, dtype=torch.float32)}
            if self.return_coords:
                coord_array = self._cache_coord_arrays[shard_idx]
                if coord_array is None:
                    raise RuntimeError(
                        "Cached PointCloudDataset was requested with return_coords=True, "
                        f"but shard {shard_idx} has no coords array."
                    )
                sample["coords"] = torch.tensor(coord_array[local_idx], dtype=torch.float32)
            return sample

        point_set = self.samples[index]
        if not self.pre_normalize and self.normalize:
            point_set = pc_normalize(point_set, float(self.sample_radii[index])).astype(np.float32)
        point_set_tensor = torch.tensor(point_set, dtype=torch.float32)
        sample = {"points": point_set_tensor}
        if self.return_coords:
            sample["coords"] = torch.tensor(self.coords[index], dtype=torch.float32)
        return sample

    def __getitems__(self, indices):
        if self._cache_sample_arrays is None:
            return [self[index] for index in indices]

        global_indices = np.asarray(indices, dtype=np.int64)
        if global_indices.ndim != 1:
            raise ValueError(
                "PointCloudDataset batched indices must be one-dimensional, "
                f"got shape={global_indices.shape}."
            )
        global_indices = global_indices.copy()
        global_indices[global_indices < 0] += int(self._cache_total_samples)
        invalid = (global_indices < 0) | (global_indices >= int(self._cache_total_samples))
        if np.any(invalid):
            bad_indices = global_indices[invalid][:10].tolist()
            raise IndexError(
                "PointCloudDataset batched cache indices are out of range. "
                f"dataset_length={self._cache_total_samples}, invalid_indices={bad_indices}."
            )

        batch_size = int(global_indices.size)
        point_batch = np.empty((batch_size, self.num_points, 3), dtype=np.float32)
        coord_batch = np.empty((batch_size, 3), dtype=np.float32) if self.return_coords else None
        cumulative = np.asarray(self._cache_cumulative_counts, dtype=np.int64)
        shard_indices = np.searchsorted(cumulative, global_indices, side="right")

        for shard_idx in np.unique(shard_indices):
            output_positions = np.flatnonzero(shard_indices == shard_idx)
            previous = 0 if shard_idx == 0 else int(cumulative[shard_idx - 1])
            local_indices = global_indices[output_positions] - previous
            local_order = np.argsort(local_indices, kind="stable")
            sorted_output_positions = output_positions[local_order]
            sorted_local_indices = local_indices[local_order]
            point_batch[sorted_output_positions] = self._cache_sample_arrays[shard_idx][
                sorted_local_indices
            ]
            if coord_batch is not None:
                coord_array = self._cache_coord_arrays[shard_idx]
                if coord_array is None:
                    raise RuntimeError(
                        "Cached PointCloudDataset was requested with return_coords=True, "
                        f"but shard {shard_idx} has no coords array."
                    )
                coord_batch[sorted_output_positions] = coord_array[sorted_local_indices]

        point_tensor = torch.from_numpy(point_batch)
        coord_tensor = torch.from_numpy(coord_batch) if coord_batch is not None else None
        samples = []
        for batch_index in range(batch_size):
            sample = {"points": point_tensor[batch_index]}
            if coord_tensor is not None:
                sample["coords"] = coord_tensor[batch_index]
            samples.append(sample)
        return samples
