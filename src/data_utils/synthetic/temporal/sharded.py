from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from os import cpu_count
from pathlib import Path

from .api import generate_temporal_dataset
from .config import TemporalBenchmarkConfig, dump_temporal_config, load_temporal_config


@dataclass(frozen=True)
class ShardedTemporalDatasetResult:
    output_root: Path
    manifest_path: Path
    shard_output_dirs: list[Path]


def generate_temporal_dataset_shards(
    config_or_path: TemporalBenchmarkConfig | str | Path,
    *,
    num_shards: int,
    max_workers: int | None = None,
    output_root: str | Path | None = None,
    shard_seed_stride: int = 10_000,
    disable_visualization: bool = True,
    progress: bool = True,
) -> ShardedTemporalDatasetResult:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}.")

    base_config = (
        config_or_path
        if isinstance(config_or_path, TemporalBenchmarkConfig)
        else load_temporal_config(config_or_path)
    )
    root_dir = Path(output_root) if output_root is not None else _default_output_root(base_config)
    if root_dir.exists():
        if not base_config.output.overwrite:
            raise FileExistsError(
                f"Sharded temporal output root already exists: {root_dir}. "
                "Set output.overwrite=true in the base config or choose a different output_root."
            )
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True, exist_ok=False)

    total_cpus = max(1, cpu_count() or 1)
    shard_workers = int(max_workers) if max_workers is not None else min(num_shards, total_cpus)
    if shard_workers <= 0:
        raise ValueError(f"max_workers must be positive when provided, got {max_workers}.")
    shard_workers = min(shard_workers, num_shards)
    per_shard_workers = max(1, total_cpus // shard_workers)
    prepared_config = _prepare_shard_config(
        base_config=base_config,
        root_dir=root_dir,
        per_shard_workers=per_shard_workers,
        disable_visualization=disable_visualization,
    )
    resolved_config_path = root_dir / ".resolved_base_config.yaml"
    dump_temporal_config(prepared_config, resolved_config_path)

    shard_specs = [
        {
            "shard_index": shard_idx,
            "seed": int(base_config.seed) + shard_idx * int(shard_seed_stride),
            "output_dir": root_dir / f"shard_{shard_idx:03d}",
        }
        for shard_idx in range(num_shards)
    ]

    if progress:
        print(
            (
                f"[temporal-shards] Launching {num_shards} shards in parallel with "
                f"max_workers={shard_workers}, per_shard_render_workers={prepared_config.rendering.parallel_workers}, "
                f"output_root={root_dir}"
            ),
            flush=True,
        )

    results: list[dict[str, object]] = []
    started_at = time.perf_counter()
    mp_context = _resolve_mp_context()
    with ProcessPoolExecutor(max_workers=shard_workers, mp_context=mp_context) as executor:
        future_to_spec = {
            executor.submit(
                _generate_single_shard,
                str(resolved_config_path),
                int(spec["seed"]),
                str(spec["output_dir"]),
            ): spec
            for spec in shard_specs
        }
        completed = 0
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            result = future.result()
            results.append(result)
            completed += 1
            if progress:
                elapsed = time.perf_counter() - started_at
                print(
                    (
                        f"[temporal-shards] Completed shard {spec['shard_index']}/{num_shards - 1} "
                        f"({completed}/{num_shards}) elapsed={_format_seconds(elapsed)} "
                        f"shard_time={_format_seconds(float(result['elapsed_seconds']))}"
                    ),
                    flush=True,
                )

    results.sort(key=lambda item: int(item["shard_index"]))
    manifest = {
        "output_root": str(root_dir),
        "num_shards": int(num_shards),
        "max_workers": int(shard_workers),
        "per_shard_render_workers": int(prepared_config.rendering.parallel_workers),
        "per_shard_visualization_workers": int(prepared_config.visualization.parallel_workers),
        "disable_visualization": bool(disable_visualization),
        "resolved_config_path": str(resolved_config_path),
        "shards": results,
        "total_elapsed_seconds": float(time.perf_counter() - started_at),
    }
    manifest_path = root_dir / "collection_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return ShardedTemporalDatasetResult(
        output_root=root_dir,
        manifest_path=manifest_path,
        shard_output_dirs=[Path(str(item["output_dir"])) for item in results],
    )


def _prepare_shard_config(
    *,
    base_config: TemporalBenchmarkConfig,
    root_dir: Path,
    per_shard_workers: int,
    disable_visualization: bool,
) -> TemporalBenchmarkConfig:
    visualization = base_config.visualization
    if disable_visualization:
        visualization = replace(visualization, enabled=False, parallel_workers=1)
    else:
        visualization = replace(
            visualization,
            parallel_workers=min(int(visualization.parallel_workers), per_shard_workers),
        )
    rendering = replace(
        base_config.rendering,
        parallel_workers=min(int(base_config.rendering.parallel_workers), per_shard_workers),
    )
    return replace(
        base_config,
        rendering=rendering,
        visualization=visualization,
        output=replace(base_config.output, output_dir=root_dir / "unused_base_output"),
    )


def _default_output_root(config: TemporalBenchmarkConfig) -> Path:
    base_output = Path(config.output.output_dir)
    return base_output.parent / f"{base_output.name}_sharded"


def _resolve_mp_context() -> mp.context.BaseContext:
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_file and main_file != "<stdin>":
        return mp.get_context("spawn")
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _generate_single_shard(
    resolved_config_path: str,
    seed: int,
    output_dir: str,
) -> dict[str, object]:
    started_at = time.perf_counter()
    result = generate_temporal_dataset(
        resolved_config_path,
        seed=seed,
        output_dir=output_dir,
        progress=False,
    )
    elapsed = time.perf_counter() - started_at
    return {
        "shard_index": int(Path(output_dir).name.split("_")[-1]),
        "seed": int(seed),
        "output_dir": str(result.output_dir),
        "manifest_path": str(result.manifest_path),
        "validation_summary_path": str(result.validation_summary_path),
        "frame_chunk_path": str(result.frame_chunk_path) if result.frame_chunk_path is not None else None,
        "elapsed_seconds": float(elapsed),
    }


def _format_seconds(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    if total_seconds < 60.0:
        return f"{total_seconds:.1f}s"
    minutes, sec = divmod(int(round(total_seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"
