#!/usr/bin/env python3
"""Encode every stored 24 ps frame and analyze GeoFrame temporal variability."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import json
import shutil
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data_utils.shooting_dataset import load_predictive_shooting_snapshot
from src.temporal_vamp.embeddings import load_frozen_encoder
from src.temporal_vamp.geoframe_temporal_variability import (
    analyze_temporal_embeddings,
    write_experiment_readme,
)
from src.temporal_vamp.shooting_embeddings import (
    ShootingEmbeddingCache,
    extract_shooting_embedding_cache,
)
from src.temporal_vamp.commands.common import (
    required,
    resolve_path,
)


def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Configuration requests device={requested!r}, but CUDA is unavailable."
        )
    return requested


def _all_stored_horizons_ps(snapshot: Any) -> list[float]:
    protocol = snapshot.manifest["protocol"]
    timestep_ps = float(protocol["timestep_fs"]) / 1000.0
    interval_steps = int(protocol["sample_interval_steps"])
    run_steps = int(protocol["run_steps"])
    timesteps = np.arange(interval_steps, run_steps + 1, interval_steps, dtype=np.int64)
    return (timesteps.astype(np.float64) * timestep_ps).tolist()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage", choices=("all", "extract", "analyze"), default="all"
    )
    args = parser.parse_args(argv)

    config_path = resolve_path(args.config)
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    output_dir = resolve_path(required(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    campaign_root = resolve_path(required(cfg, "data.campaign_root"))
    snapshot = load_predictive_shooting_snapshot(
        [campaign_root],
        temperatures_K=[float(value) for value in required(cfg, "data.temperatures_K")],
        minimum_complete_branches_per_parent=int(
            required(cfg, "data.minimum_complete_branches_per_parent")
        ),
        basin_roles=[str(value) for value in required(cfg, "data.basin_roles")],
    )
    expected_parents = int(required(cfg, "data.expected_parent_count"))
    expected_branches = int(required(cfg, "data.expected_branch_count"))
    if len(snapshot.parents) != expected_parents or len(snapshot.branches) != expected_branches:
        raise RuntimeError(
            "Temporal-variability dataset contract changed: "
            f"expected parents/branches={expected_parents}/{expected_branches}, "
            f"observed={len(snapshot.parents)}/{len(snapshot.branches)}."
        )
    with (output_dir / "dataset_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(snapshot.to_dict(), handle, indent=2, sort_keys=True)

    cache_path = output_dir / "embeddings"
    checkpoint = resolve_path(required(cfg, "encoder.checkpoint"))
    if args.stage in {"all", "extract"}:
        device = _resolve_device(str(required(cfg, "device")))
        encoder = load_frozen_encoder(
            checkpoint,
            device=device,
            repeats=int(required(cfg, "encoder.repeats")),
            seed=int(required(cfg, "encoder.seed")),
            representation_source=str(required(cfg, "encoder.representation_source")),
        )
        cache = extract_shooting_embedding_cache(
            snapshot,
            encoder=encoder,
            cache_path=cache_path,
            horizons_ps=_all_stored_horizons_ps(snapshot),
            center_atom_count=int(required(cfg, "data.center_atom_count")),
            center_selection_seed=int(required(cfg, "data.center_selection_seed")),
            num_points=int(required(cfg, "data.num_points")),
            radius=float(required(cfg, "data.radius")),
            spatial_context_center_count=0,
            spatial_context_aggregation="mean_std",
            point_cloud_batch_size=int(required(cfg, "encoder.point_cloud_batch_size")),
            environment_batch_size=int(
                required(cfg, "encoder.environment_batch_size")
            ),
            environment_num_workers=int(
                required(cfg, "encoder.environment_num_workers")
            ),
            force_recompute=bool(required(cfg, "cache.force_recompute")),
        )
        if bool(required(cfg, "cache.remove_extraction_shards")):
            shard_root = cache_path.parent / f"{cache_path.name}_shards"
            if not (cache_path / "manifest.json").is_file():
                raise RuntimeError(
                    f"Refusing to remove extraction shards before final cache exists: {cache_path}."
                )
            if shard_root.exists():
                shutil.rmtree(shard_root)
                print(f"[temporal-variability] removed completed extraction shards: {shard_root}")
    else:
        cache = ShootingEmbeddingCache.load(cache_path)

    if args.stage in {"all", "analyze"}:
        metrics = analyze_temporal_embeddings(
            cache,
            output_dir=output_dir,
            lag_frames=[int(value) for value in required(cfg, "analysis.lag_frames")],
            smoothing_windows_frames=[
                int(value) for value in required(cfg, "analysis.smoothing_windows_frames")
            ],
            pca_dimension=int(required(cfg, "analysis.pca_dimension")),
            scatter_maximum_points=int(
                required(cfg, "analysis.scatter_maximum_points")
            ),
            seed=int(required(cfg, "analysis.seed")),
        )
        write_experiment_readme(
            output_dir,
            metrics=metrics,
            checkpoint=checkpoint,
            campaign_root=campaign_root,
            config_path=output_dir / "resolved_config.yaml",
        )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
