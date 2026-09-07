#!/usr/bin/env python3
"""Ablation 7: structural history and momentum-conditioned upper bounds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.shooting_dataset import load_shooting_campaigns_snapshot
from src.temporal_vamp.embeddings import load_frozen_encoder
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_distribution import prepare_distributional_target_data
from src.temporal_vamp.shooting_dynamics import (
    ShootingDynamicalFeatureCache,
    evaluate_dynamical_ablation,
    extract_shooting_dynamical_feature_cache,
    plot_dynamical_retrieval,
)
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_multiscale import write_json
from src.temporal_vamp.shooting_spatial import build_spatial_token_data
from src.temporal_vamp.commands.common import (
    required,
    resolve_path,
    prepare_run,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "extract", "evaluate"), default="all")
    args = parser.parse_args(argv)
    cfg, output_dir = prepare_run(resolve_path(args.config))
    device = str(required(cfg, "device"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requests CUDA, but CUDA is unavailable.")
    snapshot = load_shooting_campaigns_snapshot(
        [resolve_path(value) for value in required(cfg, "data.campaign_roots")],
        temperatures_K=[float(value) for value in required(cfg, "data.temperatures_K")],
        minimum_complete_branches_per_parent=int(
            required(cfg, "data.minimum_complete_branches_per_parent")
        ),
    )
    if len(snapshot.parents) != int(required(cfg, "data.expected_parent_count")) or len(
        snapshot.branches
    ) != int(required(cfg, "data.expected_branch_count")):
        raise RuntimeError(
            f"Ablation-7 snapshot changed: parents={len(snapshot.parents)}, "
            f"branches={len(snapshot.branches)}."
        )
    write_json(output_dir / "dataset_snapshot.json", snapshot.to_dict())
    base_cache = ShootingEmbeddingCache.load(
        resolve_path(required(cfg, "shooting.base_embedding_cache"))
    )
    context_cache = ShootingContextTokenCache.load(
        resolve_path(required(cfg, "shooting.context_token_cache"))
    )
    tokens = build_spatial_token_data(base_cache, context_cache)
    targets = prepare_distributional_target_data(
        base_cache,
        horizons_ps=[float(value) for value in required(cfg, "target.horizons_ps")],
        change_pca_dim=int(required(cfg, "target.change_pca_dim")),
        rff_features_per_bandwidth=int(
            required(cfg, "target.rff_features_per_bandwidth")
        ),
        bandwidth_multipliers=[
            float(value) for value in required(cfg, "target.bandwidth_multipliers")
        ],
        selection_source_velocity_seeds=[
            int(value) for value in required(cfg, "split.selection_source_velocity_seeds")
        ],
        seed=int(required(cfg, "target.seed")),
    )
    cache_path = output_dir / "dynamical_features"
    if args.stage in {"all", "extract"}:
        encoder = load_frozen_encoder(
            resolve_path(required(cfg, "encoder.checkpoint")),
            device=device,
            repeats=int(required(cfg, "encoder.repeats")),
            seed=int(required(cfg, "encoder.seed")),
            representation_source=str(required(cfg, "encoder.representation_source")),
        )
        dynamics = extract_shooting_dynamical_feature_cache(
            snapshot,
            base_cache,
            context_cache,
            encoder=encoder,
            source_trajectory_root=resolve_path(required(cfg, "history.source_trajectory_root")),
            cache_path=cache_path,
            history_lag_frames=int(required(cfg, "history.lag_frames")),
            history_lag_ps=float(required(cfg, "history.lag_ps")),
            source_sample_interval_ps=float(
                required(cfg, "history.source_sample_interval_ps")
            ),
            num_points=int(required(cfg, "data.num_points")),
            radius=float(required(cfg, "data.radius")),
            context_center_count=int(required(cfg, "context.center_count")),
            point_cloud_batch_size=int(required(cfg, "encoder.point_cloud_batch_size")),
            force_recompute=bool(required(cfg, "cache.force_recompute")),
        )
    else:
        dynamics = ShootingDynamicalFeatureCache.load(cache_path)
    if args.stage == "extract":
        write_json(
            output_dir / "extraction_summary.json",
            {
                "previous_token_z_shape": list(dynamics.previous_token_z.shape),
                "velocity_features_shape": list(dynamics.velocity_features.shape),
                "current_embedding_max_abs_error": dynamics.manifest[
                    "current_embedding_max_abs_error"
                ],
            },
        )
        return
    result = evaluate_dynamical_ablation(
        base_cache,
        targets,
        tokens,
        dynamics,
        ablation5_arrays_path=resolve_path(
            required(cfg, "initialization.ablation5_coordinates_and_predictions")
        ),
        history_pca_dimensions=[
            int(value) for value in required(cfg, "ridge.history_pca_dimensions")
        ],
        velocity_pca_dimensions=[
            int(value) for value in required(cfg, "ridge.velocity_pca_dimensions")
        ],
        ridge_alphas=[float(value) for value in required(cfg, "ridge.alphas")],
        neighbors=int(required(cfg, "evaluation.neighbors")),
        seed=int(required(cfg, "evaluation.seed")),
    )
    metrics = result.metrics
    metrics["scientific_contract"] = {
        "ablation": 7,
        "history_experiment": (
            "3 ps previous positions for identical central and satellite atom IDs; "
            "parent-level residual on the unchanged sibling-distribution target"
        ),
        "velocity_experiment": (
            "rotation-invariant local t=0 velocity/structure descriptors; branch-level "
            "residual on each individual realized future"
        ),
        "combined_experiment": "history parent prediction plus branch velocity residual",
        "encoder": "same frozen GeoFrameTransformerV2 checkpoint",
        "split": "same source-run optimization/selection/validation isolation",
        "important_noncomparability": (
            "branch-level velocity retrieval predicts individual stochastic outcomes and is "
            "reported separately from parent-level structural propensity"
        ),
        "langevin_thermostat_time_ps": 0.3,
    }
    metrics["cache_validation"] = {
        "current_embedding_max_abs_error": dynamics.manifest[
            "current_embedding_max_abs_error"
        ],
        "history_lag_ps": float(required(cfg, "history.lag_ps")),
        "velocity_feature_dim": int(dynamics.velocity_features.shape[-1]),
    }
    write_json(output_dir / "metrics.json", metrics)
    np.savez(output_dir / "coordinates_and_predictions.npz", **result.arrays)
    np.savez(output_dir / "ridge_models.npz", **result.model_arrays)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_dynamical_retrieval(metrics, plot_dir / "dynamical_retrieval.png")
    print(f"[ablation-7] complete output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
