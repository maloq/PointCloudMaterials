#!/usr/bin/env python3
"""Extract a compute-scaled ordinary-MD cache for temporal encoder training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data_utils.temporal_binary_context_dataset import TemporalBinaryContextDataset
from src.temporal_vamp.embeddings import load_frozen_encoder
from src.temporal_vamp.ordinary_pretraining import extract_ordinary_context_embedding_cache
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.simulation_catalog import discover_simulation_catalog


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Ordinary temporal cache configuration requires {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg: DictConfig = OmegaConf.load(_resolve_path(args.config))
    OmegaConf.resolve(cfg)
    output_dir = _resolve_path(_required(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_cache_config.yaml")
    device = str(_required(cfg, "device"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requests CUDA, but CUDA is unavailable.")

    entries = discover_simulation_catalog(
        _resolve_path(_required(cfg, "ordinary.catalog.root")),
        campaign_globs=[
            str(value) for value in _required(cfg, "ordinary.catalog.campaign_globs")
        ],
        cache_root=_resolve_path(_required(cfg, "ordinary.catalog.cache_root")),
        required_atom_count=int(_required(cfg, "ordinary.catalog.required_atom_count")),
        required_potential_parameter_sha256=str(
            _required(cfg, "ordinary.catalog.required_potential_parameter_sha256")
        ),
        required_crystal_seed=OmegaConf.select(
            cfg, "ordinary.catalog.required_crystal_seed", default=None
        ),
        require_periodic=bool(_required(cfg, "ordinary.catalog.require_periodic")),
    )
    included_seeds = {
        int(value) for value in _required(cfg, "ordinary.included_velocity_seeds")
    }
    selected_entries = tuple(
        entry for entry in entries if entry.metadata.velocity_seed in included_seeds
    )
    expected_runs = int(_required(cfg, "ordinary.expected_run_count"))
    if len(selected_entries) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} selected ordinary runs, got {len(selected_entries)}."
        )
    reference = ShootingEmbeddingCache.load(
        _resolve_path(_required(cfg, "ordinary.center_atom_reference_cache"))
    )
    dataset = TemporalBinaryContextDataset(
        selected_entries,
        center_atom_ids=np.asarray(reference.atom_ids, dtype=np.int64),
        horizons_ps=[float(value) for value in _required(cfg, "ordinary.horizons_ps")],
        anchor_stride_frames=int(_required(cfg, "ordinary.anchor_stride_frames")),
        num_points=int(_required(cfg, "data.num_points")),
        radius=float(_required(cfg, "data.radius")),
        context_center_count=0,
        steinhardt_shell_min_neighbors=int(
            _required(cfg, "data.steinhardt_shell_min_neighbors")
        ),
        steinhardt_shell_max_neighbors=int(
            _required(cfg, "data.steinhardt_shell_max_neighbors")
        ),
        trajectory_cache_size=int(_required(cfg, "ordinary.trajectory_cache_size")),
    )
    encoder = load_frozen_encoder(
        _resolve_path(_required(cfg, "encoder.checkpoint")),
        device=device,
        repeats=int(_required(cfg, "encoder.repeats")),
        seed=int(_required(cfg, "encoder.seed")),
        representation_source=str(_required(cfg, "encoder.representation_source")),
    )
    cache = extract_ordinary_context_embedding_cache(
        dataset,
        encoder=encoder,
        cache_path=output_dir / "ordinary_context_embeddings",
        point_cloud_batch_size=int(_required(cfg, "encoder.point_cloud_batch_size")),
        environment_batch_size=int(_required(cfg, "encoder.environment_batch_size")),
        environment_num_workers=int(_required(cfg, "encoder.environment_num_workers")),
        force_recompute=bool(_required(cfg, "cache.force_recompute")),
    )
    summary = {
        "runs": len(selected_entries),
        "anchors": len(dataset),
        "centers_per_anchor": int(dataset.center_atom_ids.size),
        "rows": int(cache.manifest["row_count"]),
        "horizons_ps": dataset.horizons_ps.tolist(),
        "anchor_stride_frames": dataset.anchor_stride_frames,
        "source_run_ids": [entry.run_id for entry in selected_entries],
    }
    with (output_dir / "cache_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[ordinary-temporal-cache] complete {summary}", flush=True)


if __name__ == "__main__":
    main()
