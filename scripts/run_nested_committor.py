#!/usr/bin/env python3
"""Extract GeoFrameV2 nested-shooting features and fit committor baselines."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.temporal_vamp.embeddings import load_frozen_encoder  # noqa: E402
from src.temporal_vamp.nested_committor import (  # noqa: E402
    extract_nested_feature_cache,
    fit_nested_committor_models,
    load_nested_shooting_snapshot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit first-passage structure and phase-space models on the nested "
            "Al shooting campaign."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    if os.environ.get("CONDA_DEFAULT_ENV") != "pointnet":
        raise RuntimeError(
            "Nested committor training must run in conda environment 'pointnet'; "
            f"observed CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV')!r}."
        )
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = load_nested_shooting_snapshot(
        str(cfg.data.campaign_root),
        require_complete_campaign=bool(cfg.data.require_complete_campaign),
    )
    if len(snapshot.parents) != int(cfg.data.expected_parent_count):
        raise RuntimeError(
            f"Expected {cfg.data.expected_parent_count} parents, got {len(snapshot.parents)}."
        )
    if len(snapshot.outcomes) != int(cfg.data.expected_branch_count):
        raise RuntimeError(
            f"Expected {cfg.data.expected_branch_count} complete branches, "
            f"got {len(snapshot.outcomes)}."
        )
    encoder = load_frozen_encoder(
        str(cfg.encoder.checkpoint),
        device=str(cfg.device),
        repeats=int(cfg.encoder.repeats),
        seed=int(cfg.encoder.seed),
        representation_source=str(cfg.encoder.representation_source),
    )
    print(
        f"[nested-committor] device={encoder.device}, checkpoint={encoder.checkpoint_path}, "
        f"representation={encoder.representation_source}, output_dim={encoder.output_dim}",
        flush=True,
    )
    cache = extract_nested_feature_cache(
        snapshot,
        encoder=encoder,
        cache_path=str(cfg.cache.path),
        num_points=int(cfg.data.num_points),
        radius=float(cfg.data.radius),
        nucleus_center_count=int(cfg.regions.nucleus_center_count),
        interface_center_count=int(cfg.regions.interface_center_count),
        background_center_count=int(cfg.regions.background_center_count),
        point_cloud_batch_size=int(cfg.encoder.point_cloud_batch_size),
        selection_seed=int(cfg.regions.selection_seed),
        force_recompute=bool(cfg.cache.force_recompute),
    )
    metrics = fit_nested_committor_models(
        snapshot,
        cache,
        output_dir=output_dir,
        geoframe_dimensions=[int(value) for value in cfg.model.geoframe_pca_dimensions],
        velocity_dimensions=[int(value) for value in cfg.model.velocity_pca_dimensions],
        logistic_C_values=[float(value) for value in cfg.model.logistic_C_values],
        group_folds=int(cfg.model.grouped_source_run_folds),
    )
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    summary = {
        name: values["final_validation_transition"]
        for name, values in metrics["models"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

