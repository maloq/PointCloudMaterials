#!/usr/bin/env python3
"""Ablation 7: structural history and momentum-conditioned upper bounds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
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


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Dynamical ablation configuration requires {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "extract", "evaluate"), default="all")
    args = parser.parse_args()
    cfg: DictConfig = OmegaConf.load(_resolve_path(args.config))
    OmegaConf.resolve(cfg)
    output_dir = _resolve_path(_required(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    device = str(_required(cfg, "device"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requests CUDA, but CUDA is unavailable.")
    snapshot = load_shooting_campaigns_snapshot(
        [_resolve_path(value) for value in _required(cfg, "data.campaign_roots")],
        temperatures_K=[float(value) for value in _required(cfg, "data.temperatures_K")],
        minimum_complete_branches_per_parent=int(
            _required(cfg, "data.minimum_complete_branches_per_parent")
        ),
    )
    if len(snapshot.parents) != int(_required(cfg, "data.expected_parent_count")) or len(
        snapshot.branches
    ) != int(_required(cfg, "data.expected_branch_count")):
        raise RuntimeError(
            f"Ablation-7 snapshot changed: parents={len(snapshot.parents)}, "
            f"branches={len(snapshot.branches)}."
        )
    write_json(output_dir / "dataset_snapshot.json", snapshot.to_dict())
    base_cache = ShootingEmbeddingCache.load(
        _resolve_path(_required(cfg, "shooting.base_embedding_cache"))
    )
    context_cache = ShootingContextTokenCache.load(
        _resolve_path(_required(cfg, "shooting.context_token_cache"))
    )
    tokens = build_spatial_token_data(base_cache, context_cache)
    targets = prepare_distributional_target_data(
        base_cache,
        horizons_ps=[float(value) for value in _required(cfg, "target.horizons_ps")],
        change_pca_dim=int(_required(cfg, "target.change_pca_dim")),
        rff_features_per_bandwidth=int(
            _required(cfg, "target.rff_features_per_bandwidth")
        ),
        bandwidth_multipliers=[
            float(value) for value in _required(cfg, "target.bandwidth_multipliers")
        ],
        selection_source_velocity_seeds=[
            int(value) for value in _required(cfg, "split.selection_source_velocity_seeds")
        ],
        seed=int(_required(cfg, "target.seed")),
    )
    cache_path = output_dir / "dynamical_features"
    if args.stage in {"all", "extract"}:
        encoder = load_frozen_encoder(
            _resolve_path(_required(cfg, "encoder.checkpoint")),
            device=device,
            repeats=int(_required(cfg, "encoder.repeats")),
            seed=int(_required(cfg, "encoder.seed")),
            representation_source=str(_required(cfg, "encoder.representation_source")),
        )
        dynamics = extract_shooting_dynamical_feature_cache(
            snapshot,
            base_cache,
            context_cache,
            encoder=encoder,
            source_trajectory_root=_resolve_path(_required(cfg, "history.source_trajectory_root")),
            cache_path=cache_path,
            history_lag_frames=int(_required(cfg, "history.lag_frames")),
            history_lag_ps=float(_required(cfg, "history.lag_ps")),
            source_sample_interval_ps=float(
                _required(cfg, "history.source_sample_interval_ps")
            ),
            num_points=int(_required(cfg, "data.num_points")),
            radius=float(_required(cfg, "data.radius")),
            context_center_count=int(_required(cfg, "context.center_count")),
            point_cloud_batch_size=int(_required(cfg, "encoder.point_cloud_batch_size")),
            force_recompute=bool(_required(cfg, "cache.force_recompute")),
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
        ablation5_arrays_path=_resolve_path(
            _required(cfg, "initialization.ablation5_coordinates_and_predictions")
        ),
        history_pca_dimensions=[
            int(value) for value in _required(cfg, "ridge.history_pca_dimensions")
        ],
        velocity_pca_dimensions=[
            int(value) for value in _required(cfg, "ridge.velocity_pca_dimensions")
        ],
        ridge_alphas=[float(value) for value in _required(cfg, "ridge.alphas")],
        neighbors=int(_required(cfg, "evaluation.neighbors")),
        seed=int(_required(cfg, "evaluation.seed")),
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
        "history_lag_ps": float(_required(cfg, "history.lag_ps")),
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
