#!/usr/bin/env python3
"""Benchmark a static GeoFrame VICReg fused two-view training step.

The ``run`` command measures compute only. It deterministically preloads a real
``(B, data.num_points, 3)`` batch from the repository sample cache onto one GPU,
then times:

1. stochastic construction of the two VICReg views;
2. the production fused ``cat([view_a, view_b])`` encoder call, projector, and loss;
3. backward;
4. configured gradient clipping and AdamW update.

CUDA events provide phase timings, while a synchronized wall clock provides the
end-to-end step timing. Compilation occurs lazily in the warmup interval and is
reported separately. Data loading, host-to-device transfer, Lightning logging,
validation probes, and checkpoint I/O are intentionally outside the measured
interval.

Use ``compare`` on reports produced in separate processes/worktrees. This avoids
allocator and compiler-cache contamination between a baseline and candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import socket
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.data_kinds import normalize_data_kind  # noqa: E402
from src.training_methods.contrastive_learning.vicreg_module import (  # noqa: E402
    VICRegModule,
)


SCHEMA_VERSION = 1
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "output/detached/vicreg_geoframe_corrected_h100_20260829_220250/.hydra/config.yaml"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: Path, *, must_exist: bool) -> Path:
    expanded = value.expanduser()
    path = expanded if expanded.is_absolute() else REPOSITORY_ROOT / expanded
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Required path does not exist: {path}")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8")


def _run_metadata_command(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return {"status": "unavailable", "error": str(exc)}
    if completed.returncode != 0:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
        }
    return {"status": "ok", "stdout": completed.stdout.strip()}


def _git_metadata() -> dict[str, Any]:
    commit = _run_metadata_command(["git", "rev-parse", "HEAD"])
    status = _run_metadata_command(["git", "status", "--porcelain=v1"])
    return {
        "commit": commit,
        "status": status,
        "dirty": status.get("status") == "ok" and bool(status.get("stdout")),
    }


def _source_fingerprint() -> dict[str, Any]:
    roots = [
        REPOSITORY_ROOT / "src/models",
        REPOSITORY_ROOT / "src/training_methods/contrastive_learning",
    ]
    explicit = [
        REPOSITORY_ROOT / "src/training_methods/base_ssl_module.py",
        REPOSITORY_ROOT / "src/utils/pointcloud_ops.py",
        Path(__file__).resolve(),
    ]
    paths: set[Path] = set(explicit)
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Benchmark source root is missing: {root}")
        paths.update(root.rglob("*.py"))

    files = {}
    overall = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        file_digest = _sha256_file(path)
        files[relative] = file_digest
        overall.update(relative.encode("utf-8"))
        overall.update(b"\0")
        overall.update(file_digest.encode("ascii"))
        overall.update(b"\n")
    return {"sha256": overall.hexdigest(), "files": files}


def _load_config(
    path: Path,
    *,
    batch_size: int | None,
    compile_override: str,
    compile_mode: str | None,
) -> tuple[DictConfig, Path]:
    config_path = _resolve_path(path, must_exist=True)
    config_root = REPOSITORY_ROOT / "configs"
    if config_path.is_relative_to(config_root):
        config_name = config_path.relative_to(config_root).with_suffix("").as_posix()
        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_root),
        ):
            cfg = compose(config_name=config_name)
    else:
        cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected a mapping config at {config_path}, got {type(cfg)!r}.")
    OmegaConf.resolve(cfg)

    if batch_size is not None:
        if batch_size < 1:
            raise ValueError(f"--batch-size must be positive, got {batch_size}.")
        cfg.batch_size = int(batch_size)
    if compile_override == "enabled":
        cfg.compile_encoder = True
    elif compile_override == "disabled":
        cfg.compile_encoder = False
    elif compile_override != "config":
        raise ValueError(f"Unexpected compile override: {compile_override!r}.")
    if compile_mode is not None:
        cfg.encoder_compile_mode = str(compile_mode)

    if normalize_data_kind(str(cfg.data.kind)) != "static":
        raise ValueError(
            "This benchmark only supports the repository static dataset; "
            f"got data.kind={cfg.data.kind!r} in {config_path}."
        )
    if str(cfg.model_type).lower() != "vicreg":
        raise ValueError(
            f"Expected model_type='vicreg', got {cfg.model_type!r} in {config_path}."
        )
    encoder_name = str(cfg.encoder.name)
    if encoder_name not in {"GeoFrameTransformer", "GeoFrameTransformerV2"}:
        raise ValueError(
            "This benchmark is scoped to GeoFrameTransformer variants; "
            f"got encoder.name={encoder_name!r}."
        )
    if int(cfg.data.num_points) < 1 or int(cfg.data.model_points) < 1:
        raise ValueError(
            "Static benchmark requires positive data.num_points and data.model_points, "
            f"got {cfg.data.num_points!r} and {cfg.data.model_points!r}."
        )
    if not bool(cfg.vicreg_enabled) or float(cfg.vicreg_weight) <= 0.0:
        raise ValueError(
            "The resolved config must have an active VICReg objective; "
            f"got vicreg_enabled={cfg.vicreg_enabled!r}, vicreg_weight={cfg.vicreg_weight!r}."
        )
    if bool(getattr(cfg, "swav_enabled", False)):
        raise ValueError("Static fused VICReg benchmark requires swav_enabled=false.")
    if bool(getattr(cfg, "factor_vae_enabled", False)):
        raise ValueError("Static fused VICReg benchmark requires factor_vae_enabled=false.")
    return cfg, config_path


def _resolve_sample_cache(cfg: DictConfig, override: Path | None) -> Path:
    if override is not None:
        cache_dir = _resolve_path(override, must_exist=True)
    else:
        cache_cfg = getattr(cfg.data, "sample_cache", None)
        if cache_cfg is None or not bool(cache_cfg.enabled):
            raise ValueError(
                "The static config does not enable a sample cache. Pass --sample-cache "
                "with a ready-to-train cache directory."
            )
        cache_dir = _resolve_path(Path(str(cache_cfg.cache_dir)), must_exist=True)
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Static sample cache has no metadata.json: cache_dir={cache_dir}."
        )
    return cache_dir


def _load_real_batch(
    *,
    cache_dir: Path,
    batch_size: int,
    expected_points: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    metadata_path = cache_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict) or int(metadata.get("schema_version", -1)) != 1:
        raise ValueError(
            f"Unsupported static sample cache metadata at {metadata_path}: "
            f"schema_version={metadata.get('schema_version')!r}."
        )
    shards = metadata.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"Static sample cache has no shards: {metadata_path}.")
    request = metadata.get("request")
    if not isinstance(request, dict):
        raise TypeError(f"Static sample cache request must be a mapping: {metadata_path}.")
    cached_points = int(request.get("num_points", -1))
    if cached_points != expected_points:
        raise ValueError(
            "Static cache/config point-count mismatch: "
            f"cache num_points={cached_points}, config data.num_points={expected_points}, "
            f"cache_dir={cache_dir}."
        )

    counts = np.asarray([int(shard["count"]) for shard in shards], dtype=np.int64)
    if np.any(counts <= 0):
        raise ValueError(f"Static sample cache contains non-positive shard counts: {counts.tolist()}.")
    cumulative = np.cumsum(counts)
    total_samples = int(cumulative[-1])
    if batch_size > total_samples:
        raise ValueError(
            f"Benchmark batch_size={batch_size} exceeds cache sample count={total_samples}."
        )

    rng = np.random.default_rng(seed)
    global_indices = rng.choice(total_samples, size=batch_size, replace=False, shuffle=True)
    shard_indices = np.searchsorted(cumulative, global_indices, side="right")
    previous = np.concatenate([np.zeros(1, dtype=np.int64), cumulative[:-1]])
    local_indices = global_indices - previous[shard_indices]
    points = np.empty((batch_size, expected_points, 3), dtype=np.float32)
    selected_sources: dict[str, int] = {}

    for shard_index in np.unique(shard_indices):
        positions = np.flatnonzero(shard_indices == shard_index)
        shard = shards[int(shard_index)]
        samples_path = cache_dir / str(shard["samples_path"])
        if not samples_path.is_file():
            raise FileNotFoundError(
                f"Static sample-cache shard is missing: {samples_path}."
            )
        samples = np.load(samples_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (int(shard["count"]), expected_points, 3)
        if tuple(samples.shape) != expected_shape or samples.dtype != np.float32:
            raise ValueError(
                "Static sample-cache shard has an unexpected array contract: "
                f"path={samples_path}, expected_shape={expected_shape}, "
                f"actual_shape={tuple(samples.shape)}, expected_dtype=float32, "
                f"actual_dtype={samples.dtype}."
            )
        points[positions] = samples[local_indices[positions]]
        source = str(shard["source"])
        selected_sources[source] = selected_sources.get(source, 0) + int(positions.size)

    tensor = torch.from_numpy(points)
    batch_sha256 = _sha256_bytes(memoryview(points).cast("B"))
    return tensor, {
        "cache_dir": str(cache_dir),
        "metadata_sha256": _sha256_file(metadata_path),
        "cache_fingerprint": metadata.get("fingerprint"),
        "total_cache_samples": total_samples,
        "selection_seed": seed,
        "selected_source_counts": selected_sources,
        "global_indices_sha256": _sha256_bytes(global_indices.tobytes()),
        "batch_sha256": batch_sha256,
    }


def _precision_context(precision: str):
    normalized = precision.strip().lower()
    if normalized in {"bf16-mixed", "bf16"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if normalized in {"32-true", "32", "32-full"}:
        return nullcontext()
    raise ValueError(
        "This benchmark intentionally supports only the production bf16-mixed path "
        f"and an FP32 control; got precision={precision!r}."
    )


def _extract_optimizer(module: VICRegModule) -> torch.optim.Optimizer:
    configured = module.configure_optimizers()
    if not isinstance(configured, tuple) or len(configured) != 2:
        raise TypeError(
            "Expected VICRegModule.configure_optimizers() to return "
            f"(optimizers, schedulers), got {type(configured)!r}: {configured!r}."
        )
    optimizers = configured[0]
    if not isinstance(optimizers, list) or len(optimizers) != 1:
        raise ValueError(
            "Static VICReg benchmark expects exactly one model optimizer, "
            f"got {optimizers!r}."
        )
    optimizer = optimizers[0]
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(f"Configured optimizer is not a torch optimizer: {type(optimizer)!r}.")
    return optimizer


def _forward_vicreg_loss(
    module: VICRegModule,
    raw_points: torch.Tensor,
    *,
    precision: str,
) -> torch.Tensor:
    with _precision_context(precision):
        views = module._build_contrastive_view_pair(
            raw_points,
            view_points=module.vicreg.view_points,
        )
        z_a, z_b = module._encode_contrastive_view_pair(views)
        vicreg_loss, _ = module.vicreg.compute_loss_from_features(
            z_a_feat=z_a,
            z_b_feat=z_b,
            current_epoch=0,
        )
        if vicreg_loss is None:
            raise RuntimeError(
                "Configured VICReg objective returned no loss at epoch 0. "
                "Check vicreg_enabled, vicreg_weight, vicreg_start_epoch, and projector settings."
            )
        loss = module.vicreg.weight * vicreg_loss
    if loss.ndim != 0:
        raise ValueError(f"Expected scalar VICReg loss, got shape={tuple(loss.shape)}.")
    return loss


def _new_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _one_step(
    *,
    module: VICRegModule,
    optimizer: torch.optim.Optimizer,
    raw_points: torch.Tensor,
    precision: str,
    gradient_clip: float,
    optimizer_step: bool,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(raw_points.device)

    step_start = _new_event()
    forward_end = _new_event()
    backward_end = _new_event()
    update_end = _new_event()

    wall_start = time.perf_counter()
    step_start.record()
    loss = _forward_vicreg_loss(module, raw_points, precision=precision)
    forward_end.record()
    loss.backward()
    backward_end.record()

    if gradient_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=gradient_clip)
    if optimizer_step:
        optimizer.step()
    update_end.record()
    update_end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    loss_value = float(loss.detach().float().item())
    if not np.isfinite(loss_value):
        raise FloatingPointError(f"Static VICReg benchmark produced non-finite loss={loss_value}.")
    return {
        "forward_ms": float(step_start.elapsed_time(forward_end)),
        "backward_ms": float(forward_end.elapsed_time(backward_end)),
        "update_ms": float(backward_end.elapsed_time(update_end)),
        "step_cuda_ms": float(step_start.elapsed_time(update_end)),
        "step_wall_ms": float(wall_ms),
        "loss": loss_value,
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty benchmark sample sequence.")
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _nvidia_smi_metadata(device_index: int) -> dict[str, Any]:
    query = (
        "name,uuid,driver_version,memory.total,pstate,clocks.current.sm,"
        "clocks.current.memory,power.limit"
    )
    return _run_metadata_command(
        [
            "nvidia-smi",
            f"--id={device_index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    )


def _environment_metadata(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "gpu": {
            "index": int(device.index),
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": int(properties.multi_processor_count),
            "nvidia_smi": _nvidia_smi_metadata(int(device.index)),
        },
    }


def _run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Static VICReg benchmark requires CUDA; torch.cuda.is_available() is false.")
    if args.warmup_steps < 1:
        raise ValueError(f"--warmup-steps must be >= 1, got {args.warmup_steps}.")
    if args.measure_steps < 1:
        raise ValueError(f"--measure-steps must be >= 1, got {args.measure_steps}.")

    cfg, config_path = _load_config(
        args.config,
        batch_size=args.batch_size,
        compile_override=args.compile,
        compile_mode=args.compile_mode,
    )
    batch_size = int(cfg.batch_size)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda", int(args.device))
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    cache_dir = _resolve_sample_cache(cfg, args.sample_cache)
    host_points, input_metadata = _load_real_batch(
        cache_dir=cache_dir,
        batch_size=batch_size,
        expected_points=int(cfg.data.num_points),
        seed=seed,
    )
    raw_points = host_points.to(device=device, dtype=torch.float32, non_blocking=False)
    del host_points

    module = VICRegModule(cfg).to(device).train()
    if not module.vicreg.should_run(current_epoch=0):
        raise RuntimeError("Resolved VICReg objective is inactive at current_epoch=0.")
    optimizer = _extract_optimizer(module)
    gradient_clip = float(getattr(cfg, "gradient_clip_val", 0.0))
    optimizer_step = not bool(args.no_optimizer_step)

    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    initial_allocated = int(torch.cuda.memory_allocated(device))
    initial_reserved = int(torch.cuda.memory_reserved(device))

    print(
        "[static-vicreg-benchmark] "
        f"label={args.label!r} batch={batch_size} raw_points={cfg.data.num_points} "
        f"model_points={cfg.data.model_points} precision={cfg.precision} "
        f"compile={cfg.compile_encoder} warmup={args.warmup_steps} "
        f"measure={args.measure_steps}",
        flush=True,
    )

    warmup_records = []
    warmup_wall_start = time.perf_counter()
    for step_index in range(args.warmup_steps):
        record = _one_step(
            module=module,
            optimizer=optimizer,
            raw_points=raw_points,
            precision=str(cfg.precision),
            gradient_clip=gradient_clip,
            optimizer_step=optimizer_step,
        )
        warmup_records.append(record)
        print(
            "[static-vicreg-benchmark] "
            f"warmup={step_index + 1}/{args.warmup_steps} "
            f"wall_ms={record['step_wall_ms']:.3f}",
            flush=True,
        )
    warmup_wall_seconds = time.perf_counter() - warmup_wall_start

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    measured_initial_allocated = int(torch.cuda.memory_allocated(device))
    measured_initial_reserved = int(torch.cuda.memory_reserved(device))

    records = []
    for step_index in range(args.measure_steps):
        record = _one_step(
            module=module,
            optimizer=optimizer,
            raw_points=raw_points,
            precision=str(cfg.precision),
            gradient_clip=gradient_clip,
            optimizer_step=optimizer_step,
        )
        records.append(record)
        print(
            "[static-vicreg-benchmark] "
            f"measure={step_index + 1}/{args.measure_steps} "
            f"forward_ms={record['forward_ms']:.3f} "
            f"backward_ms={record['backward_ms']:.3f} "
            f"update_ms={record['update_ms']:.3f} "
            f"step_ms={record['step_cuda_ms']:.3f}",
            flush=True,
        )

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    timing = {
        name: _distribution([record[name] for record in records])
        for name in (
            "forward_ms",
            "backward_ms",
            "update_ms",
            "step_cuda_ms",
            "step_wall_ms",
            "loss",
        )
    }
    samples_per_second = batch_size / (timing["step_cuda_ms"]["median"] / 1000.0)

    report = {
        "schema_version": SCHEMA_VERSION,
        "report_type": "static_vicreg_fused_two_view_step",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": str(args.label),
        "scope": {
            "included": [
                "two stochastic static VICReg views",
                f"one fused 2B {cfg.encoder.name} encoder forward",
                "VICReg projector and configured loss",
                "backward",
                "configured gradient clipping",
                "AdamW update" if optimizer_step else "no optimizer update",
            ],
            "excluded": [
                "data loading and host-to-device transfer",
                "Lightning logging and callbacks",
                "validation/probe metrics",
                "checkpoint load/save",
            ],
        },
        "command": [sys.executable, *sys.argv],
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "resolved_yaml_sha256": _sha256_bytes(
                OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True).encode("utf-8")
            ),
            "batch_size": batch_size,
            "encoder_name": str(cfg.encoder.name),
            "raw_points": int(cfg.data.num_points),
            "model_points": int(cfg.data.model_points),
            "precision": str(cfg.precision),
            "compile_encoder": bool(cfg.compile_encoder),
            "compile_mode": str(cfg.encoder_compile_mode),
            "compile_fullgraph": bool(cfg.encoder_compile_fullgraph),
            "compile_dynamic": bool(cfg.encoder_compile_dynamic),
            "gradient_clip_val": gradient_clip,
            "optimizer_step": optimizer_step,
        },
        "input": input_metadata,
        "warmup": {
            "steps": int(args.warmup_steps),
            "wall_seconds": float(warmup_wall_seconds),
            "first_step_wall_ms": float(warmup_records[0]["step_wall_ms"]),
            "last_step_wall_ms": float(warmup_records[-1]["step_wall_ms"]),
        },
        "measurement": {
            "steps": int(args.measure_steps),
            "timing": timing,
            "median_samples_per_second": float(samples_per_second),
            "raw_records": records,
        },
        "memory": {
            "after_setup_allocated_bytes": initial_allocated,
            "after_setup_reserved_bytes": initial_reserved,
            "measurement_start_allocated_bytes": measured_initial_allocated,
            "measurement_start_reserved_bytes": measured_initial_reserved,
            "measurement_peak_allocated_bytes": peak_allocated,
            "measurement_peak_reserved_bytes": peak_reserved,
            "measurement_peak_incremental_allocated_bytes": max(
                0, peak_allocated - measured_initial_allocated
            ),
        },
        "model": {
            "parameter_count": int(parameter_count),
            "trainable_parameter_count": int(trainable_parameter_count),
        },
        "environment": _environment_metadata(device),
        "provenance": {
            "git": _git_metadata(),
            "source": _source_fingerprint(),
        },
    }
    return report


def _load_report(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = _resolve_path(path, must_exist=True)
    with resolved.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise TypeError(f"Benchmark report must be a JSON mapping: {resolved}.")
    if int(report.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported benchmark report schema at {resolved}: "
            f"{report.get('schema_version')!r}."
        )
    if report.get("report_type") != "static_vicreg_fused_two_view_step":
        raise ValueError(
            f"Unexpected report_type in {resolved}: {report.get('report_type')!r}."
        )
    return resolved, report


def _nested(report: dict[str, Any], *keys: str) -> Any:
    value: Any = report
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Benchmark report is missing {'.'.join(keys)!r}.")
        value = value[key]
    return value


def _compare_reports(args: argparse.Namespace) -> dict[str, Any]:
    baseline_path, baseline = _load_report(args.baseline)
    candidate_path, candidate = _load_report(args.candidate)

    comparability_fields = [
        ("config", "batch_size"),
        ("config", "raw_points"),
        ("config", "model_points"),
        ("config", "precision"),
        ("config", "gradient_clip_val"),
        ("config", "optimizer_step"),
        ("input", "batch_sha256"),
        ("environment", "gpu", "name"),
        ("environment", "gpu", "compute_capability"),
    ]
    mismatches = []
    for field in comparability_fields:
        baseline_value = _nested(baseline, *field)
        candidate_value = _nested(candidate, *field)
        if baseline_value != candidate_value:
            mismatches.append(
                {
                    "field": ".".join(field),
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                }
            )
    if mismatches and not args.allow_mismatch:
        raise ValueError(
            "Baseline and candidate reports are not directly comparable. "
            f"Mismatches={mismatches}. Re-run with identical inputs/settings, or pass "
            "--allow-mismatch to produce an explicitly non-comparable report."
        )

    timing_comparison = {}
    for metric in (
        "forward_ms",
        "backward_ms",
        "update_ms",
        "step_cuda_ms",
        "step_wall_ms",
    ):
        baseline_median = float(
            _nested(baseline, "measurement", "timing", metric, "median")
        )
        candidate_median = float(
            _nested(candidate, "measurement", "timing", metric, "median")
        )
        if baseline_median <= 0.0 or candidate_median <= 0.0:
            raise ValueError(
                f"Cannot compare non-positive median {metric}: "
                f"baseline={baseline_median}, candidate={candidate_median}."
            )
        timing_comparison[metric] = {
            "baseline_median": baseline_median,
            "candidate_median": candidate_median,
            "speedup_x": baseline_median / candidate_median,
            "candidate_change_percent": 100.0
            * (candidate_median - baseline_median)
            / baseline_median,
        }

    baseline_peak = int(_nested(baseline, "memory", "measurement_peak_allocated_bytes"))
    candidate_peak = int(_nested(candidate, "memory", "measurement_peak_allocated_bytes"))
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "report_type": "static_vicreg_fused_two_view_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "path": str(baseline_path),
            "sha256": _sha256_file(baseline_path),
            "label": baseline.get("label"),
            "source_sha256": _nested(baseline, "provenance", "source", "sha256"),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256_file(candidate_path),
            "label": candidate.get("label"),
            "source_sha256": _nested(candidate, "provenance", "source", "sha256"),
        },
        "directly_comparable": not mismatches,
        "mismatches": mismatches,
        "timing": timing_comparison,
        "memory": {
            "baseline_peak_allocated_bytes": baseline_peak,
            "candidate_peak_allocated_bytes": candidate_peak,
            "candidate_delta_bytes": candidate_peak - baseline_peak,
            "candidate_change_percent": (
                100.0 * (candidate_peak - baseline_peak) / baseline_peak
                if baseline_peak > 0
                else None
            ),
        },
    }
    return comparison


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark or compare static GeoFrame VICReg fused two-view steps."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one benchmark in this process.")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser.add_argument(
        "--sample-cache",
        type=Path,
        default=None,
        help="Ready-to-train static cache; defaults to data.sample_cache.cache_dir.",
    )
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--label", default="unnamed")
    run_parser.add_argument("--device", type=int, default=0)
    run_parser.add_argument("--batch-size", type=int, default=None)
    run_parser.add_argument("--warmup-steps", type=int, default=5)
    run_parser.add_argument("--measure-steps", type=int, default=20)
    run_parser.add_argument("--seed", type=int, default=123)
    run_parser.add_argument(
        "--compile",
        choices=("config", "enabled", "disabled"),
        default="config",
        help="Inherit compile_encoder from config, or override it.",
    )
    run_parser.add_argument(
        "--compile-mode",
        default=None,
        help="Optional encoder_compile_mode override; fullgraph/dynamic remain from config.",
    )
    run_parser.add_argument(
        "--no-optimizer-step",
        action="store_true",
        help="Measure forward+backward+clip without AdamW update.",
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Compare reports produced by two independent run commands."
    )
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Emit a report marked non-comparable instead of rejecting setting/input mismatches.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "run":
        report = _run_benchmark(args)
    elif args.command == "compare":
        report = _compare_reports(args)
    else:
        raise AssertionError(f"Unhandled command: {args.command!r}.")

    output_path = _resolve_path(args.output, must_exist=False)
    _write_json(output_path, report)
    if args.command == "run":
        timing = report["measurement"]["timing"]
        memory_gib = report["memory"]["measurement_peak_allocated_bytes"] / (1024**3)
        print(
            "[static-vicreg-benchmark] complete "
            f"forward={timing['forward_ms']['median']:.3f} ms "
            f"backward={timing['backward_ms']['median']:.3f} ms "
            f"step={timing['step_cuda_ms']['median']:.3f} ms "
            f"peak_allocated={memory_gib:.3f} GiB "
            f"report={output_path}",
            flush=True,
        )
    else:
        speedup = report["timing"]["step_cuda_ms"]["speedup_x"]
        print(
            "[static-vicreg-benchmark] comparison complete "
            f"step_speedup={speedup:.4f}x report={output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
