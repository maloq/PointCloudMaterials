"""Buffered file I/O and materialized float16 mmap batches; no research inputs."""
import hashlib
import mmap
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import summarize


@dataclass
class StorageSettings:
    size_gib: float = 1.0
    block_mib: int = 8
    points: int = 80
    batch_size: int = 512
    batches: int = 128
    repeats: int = 3
    seed: int = 1729


def validate(settings):
    if not np.isfinite(settings.size_gib) or settings.size_gib <= 0:
        raise ValueError("storage.size_gib must be finite and positive")
    for name in ("block_mib", "points", "batch_size", "batches", "repeats"):
        if type(getattr(settings, name)) is not int or getattr(settings, name) < 1:
            raise ValueError(f"storage.{name} must be a positive integer")
    if int(settings.size_gib * 2**30) < settings.points * 3 * 2:
        raise ValueError("storage.size_gib must hold at least one float16 point cloud")


def _stream_read(path, block_bytes):
    buffer = bytearray(block_bytes)
    with path.open("rb", buffering=0) as stream:
        started = time.perf_counter()
        count = 0
        while size := stream.readinto(buffer):
            count += size
        elapsed = time.perf_counter() - started
    if count != path.stat().st_size:
        raise RuntimeError(f"Short sequential read of {path}: {count} bytes")
    return elapsed


def _advise_eviction(path):
    # Best-effort file-local eviction; never drop the machine's global caches.
    with path.open("rb", buffering=0) as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def _write_and_verify(path, settings, records, chunk_records):
    rng = np.random.default_rng(settings.seed)
    expected = hashlib.sha256()
    started = time.perf_counter()
    io_seconds = 0.0
    with path.open("wb", buffering=0) as stream:
        for start in range(0, records, chunk_records):
            count = min(chunk_records, records - start)
            values = rng.uniform(-9.2, 9.2, (count, settings.points, 3)).astype("<f2")
            payload = memoryview(values).cast("B")
            expected.update(payload)
            io_started = time.perf_counter()
            while payload:
                written = stream.write(payload)
                if written is None or written <= 0:
                    raise OSError(f"Short write to synthetic cache {path}")
                payload = payload[written:]
            io_seconds += time.perf_counter() - io_started
        io_started = time.perf_counter()
        os.fsync(stream.fileno())
        io_seconds += time.perf_counter() - io_started
    wall_seconds = time.perf_counter() - started
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected.hexdigest():
        raise RuntimeError(f"Synthetic cache checksum mismatch at {path}: {actual}")
    return wall_seconds, io_seconds, actual


def run(settings, target):
    validate(settings)
    target = Path(target).expanduser().resolve(strict=True)
    if not target.is_dir():
        raise ValueError(f"Storage target must be an existing directory: {target}")
    record_bytes = settings.points * 3 * 2
    records = int(settings.size_gib * 2**30) // record_bytes
    size_bytes = records * record_bytes
    if shutil.disk_usage(target).free < size_bytes + 16 * 2**20:
        raise RuntimeError(f"Need {size_bytes + 16 * 2**20} free bytes under {target}")
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("The storage protocol requires Linux/POSIX posix_fadvise")
    block_bytes = settings.block_mib * 2**20
    chunk_records = max(1, block_bytes // record_bytes)
    write_wall, write_io = [], []
    timings = {f"{kind}_{cache}": [] for kind in ("sequential", "mmap")
               for cache in ("eviction_advised", "warm")}
    rng = np.random.default_rng(settings.seed + 1)
    indices = rng.integers(records, size=(settings.batches, settings.batch_size))
    reference_checksum = None
    with tempfile.TemporaryDirectory(prefix="pcm-hardware-", dir=target) as scratch:
        path = Path(scratch) / "clouds.float16.bin"
        for repeat in range(settings.repeats):
            print(f"storage: trial {repeat + 1}/{settings.repeats}, {size_bytes / 2**30:.3f} GiB", flush=True)
            wall, io, digest = _write_and_verify(path, settings, records, chunk_records)
            write_wall.append(wall)
            write_io.append(io)
            for cache in ("eviction_advised", "warm"):
                if cache == "eviction_advised":
                    _advise_eviction(path)
                else:
                    _stream_read(path, block_bytes)
                timings[f"sequential_{cache}"].append(_stream_read(path, block_bytes))
            # Close each mapping before the next eviction request. Copy/convert
            # every requested element, rather than just timing a lazy mmap view.
            for cache in ("eviction_advised", "warm"):
                _advise_eviction(path)
                with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                    array = np.ndarray((records, settings.points, 3), dtype="<f2", buffer=mapped)
                    if cache == "warm":
                        for batch_indices in indices:
                            array[batch_indices].astype(np.float32).sum(dtype=np.float64)
                    started = time.perf_counter()
                    checksum = 0.0
                    for batch_indices in indices:
                        batch = array[batch_indices].astype(np.float32)
                        checksum += float(batch.sum(dtype=np.float64))
                    elapsed = time.perf_counter() - started
                    del array
                timings[f"mmap_{cache}"].append(elapsed)
                if reference_checksum is None:
                    reference_checksum = checksum
                if checksum != reference_checksum:
                    raise RuntimeError(f"mmap batch checksum changed: {checksum} != {reference_checksum}")
        metrics = dict(write_generate_hash_fsync=summarize(write_wall, size_bytes / 2**20, "MiB"),
                       write_syscalls_fsync=summarize(write_io, size_bytes / 2**20, "MiB"))
        for name, samples in timings.items():
            if name.startswith("sequential"):
                metrics[name] = summarize(samples, size_bytes / 2**20, "MiB")
            else:
                metrics[name] = summarize(samples, settings.batches * settings.batch_size, "clouds")
                metrics[name]["logical_MiB_per_second"] = (
                    metrics[name]["clouds_per_second"] * record_bytes / 2**20)
    return dict(metrics=metrics, target=str(target), file_bytes=size_bytes,
                shape=[records, settings.points, 3], dtype="<f2", sha256=digest,
                sampled_checksum=reference_checksum, temporary_files_removed=True,
                cache_policy="POSIX_FADV_DONTNEED is advisory; warm mmap replays the same indices. "
                             "Neither result is a guaranteed physical-device bandwidth measurement.")
