#!/usr/bin/env python3
"""Ablation 5: ordinary-trajectory temporal pretraining before shooting fitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.temporal_binary_context_dataset import TemporalBinaryContextDataset
from src.temporal_vamp.embeddings import load_frozen_encoder
from src.temporal_vamp.ordinary_pretraining import (
    OrdinaryContextEmbeddingCache,
    PretrainedSpatialBackbones,
    extract_ordinary_context_embedding_cache,
    prepare_ordinary_pretraining_targets,
    pretrain_spatial_context_backbones,
    save_pretrained_spatial_backbones,
)
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_distribution import (
    evaluate_distributional_predictor,
    prepare_distributional_target_data,
    save_distributional_preprocessing,
)
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_multiscale import (
    build_multiscale_feature_variants,
    plot_multiscale_retrieval,
    plot_multiscale_training,
    write_json,
)
from src.temporal_vamp.shooting_spatial import (
    build_spatial_token_data,
    fit_spatial_context_transformer,
    fit_spatial_token_standardization,
)
from src.temporal_vamp.simulation_catalog import discover_simulation_catalog


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Temporal-pretraining ablation requires configuration key {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def _load_pretrained(path: Path) -> PretrainedSpatialBackbones:
    if not path.is_file():
        raise FileNotFoundError(f"Ordinary pretrained-backbone checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return PretrainedSpatialBackbones(
        states={int(key): value for key, value in payload["backbone_states"].items()},
        histories={int(key): value for key, value in payload["histories"].items()},
        metrics={int(key): value for key, value in payload["metrics"].items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("all", "extract", "pretrain", "train"), default="all"
    )
    args = parser.parse_args()
    cfg: DictConfig = OmegaConf.load(_resolve_path(args.config))
    OmegaConf.resolve(cfg)
    output_dir = _resolve_path(_required(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    device = str(_required(cfg, "device"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requests CUDA, but CUDA is unavailable.")

    base_cache = ShootingEmbeddingCache.load(
        _resolve_path(_required(cfg, "shooting.base_embedding_cache"))
    )
    context_cache = ShootingContextTokenCache.load(
        _resolve_path(_required(cfg, "shooting.context_token_cache"))
    )
    shooting_tokens = build_spatial_token_data(base_cache, context_cache)
    shooting_targets = prepare_distributional_target_data(
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
            int(value) for value in _required(cfg, "split.shooting_selection_velocity_seeds")
        ],
        seed=int(_required(cfg, "target.seed")),
    )
    (
        embedding_mean,
        embedding_scale,
        descriptor_mean,
        descriptor_scale,
    ) = fit_spatial_token_standardization(
        shooting_tokens, shooting_targets.split_rows["optimization"]
    )

    entries = discover_simulation_catalog(
        _resolve_path(_required(cfg, "ordinary.catalog.root")),
        campaign_globs=[str(value) for value in _required(cfg, "ordinary.catalog.campaign_globs")],
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
    optimization_seeds = [
        int(value) for value in _required(cfg, "split.ordinary_optimization_velocity_seeds")
    ]
    selection_seeds = [
        int(value) for value in _required(cfg, "split.ordinary_selection_velocity_seeds")
    ]
    excluded_seeds = [
        int(value) for value in _required(cfg, "split.ordinary_excluded_velocity_seeds")
    ]
    included_seed_set = set(optimization_seeds + selection_seeds)
    selected_entries = tuple(
        entry for entry in entries if entry.metadata.velocity_seed in included_seed_set
    )
    observed_excluded = {
        entry.metadata.velocity_seed for entry in entries if entry not in selected_entries
    }
    if observed_excluded != set(excluded_seeds):
        raise RuntimeError(
            "Ordinary run split does not match the explicit leakage contract: "
            f"expected_excluded={sorted(excluded_seeds)}, "
            f"observed_excluded={sorted(observed_excluded)}."
        )
    expected_run_count = int(_required(cfg, "ordinary.expected_included_run_count"))
    if len(selected_entries) != expected_run_count:
        raise RuntimeError(
            f"Expected {expected_run_count} ordinary runs, discovered {len(selected_entries)}."
        )
    if any(not entry.trajectory_path.is_dir() for entry in selected_entries):
        raise RuntimeError("Every selected ordinary trajectory must be a float32 binary directory.")

    dataset = TemporalBinaryContextDataset(
        selected_entries,
        center_atom_ids=np.asarray(base_cache.atom_ids, dtype=np.int64),
        horizons_ps=[float(value) for value in _required(cfg, "target.horizons_ps")],
        anchor_stride_frames=int(_required(cfg, "ordinary.anchor_stride_frames")),
        num_points=int(_required(cfg, "data.num_points")),
        radius=float(_required(cfg, "data.radius")),
        context_center_count=int(_required(cfg, "context.center_count")),
        steinhardt_shell_min_neighbors=int(
            _required(cfg, "context.steinhardt_shell_min_neighbors")
        ),
        steinhardt_shell_max_neighbors=int(
            _required(cfg, "context.steinhardt_shell_max_neighbors")
        ),
        trajectory_cache_size=int(_required(cfg, "ordinary.trajectory_cache_size")),
    )
    snapshot = {
        "included_runs": [entry.to_dict() for entry in selected_entries],
        "excluded_runs": [entry.to_dict() for entry in entries if entry not in selected_entries],
        "optimization_velocity_seeds": optimization_seeds,
        "selection_velocity_seeds": selection_seeds,
        "excluded_velocity_seeds": excluded_seeds,
        "anchor_count": len(dataset),
        "row_count": len(dataset) * int(base_cache.atom_ids.size),
    }
    write_json(output_dir / "ordinary_dataset_snapshot.json", snapshot)

    cache_path = output_dir / "ordinary_context_embeddings"
    if args.stage in {"all", "extract"}:
        encoder = load_frozen_encoder(
            _resolve_path(_required(cfg, "encoder.checkpoint")),
            device=device,
            repeats=int(_required(cfg, "encoder.repeats")),
            seed=int(_required(cfg, "encoder.seed")),
            representation_source=str(_required(cfg, "encoder.representation_source")),
        )
        ordinary_cache = extract_ordinary_context_embedding_cache(
            dataset,
            encoder=encoder,
            cache_path=cache_path,
            point_cloud_batch_size=int(_required(cfg, "encoder.point_cloud_batch_size")),
            environment_batch_size=int(_required(cfg, "encoder.environment_batch_size")),
            environment_num_workers=int(_required(cfg, "encoder.environment_num_workers")),
            force_recompute=bool(_required(cfg, "cache.force_recompute")),
        )
    else:
        ordinary_cache = OrdinaryContextEmbeddingCache.load(cache_path)
    if args.stage == "extract":
        print(f"[ablation-5] extraction complete: {cache_path}", flush=True)
        return

    ordinary_targets = prepare_ordinary_pretraining_targets(
        ordinary_cache,
        optimization_velocity_seeds=optimization_seeds,
        selection_velocity_seeds=selection_seeds,
        pca_dim_per_horizon=int(_required(cfg, "pretraining.pca_dim_per_horizon")),
    )
    pretraining_path = output_dir / "ordinary_pretrained_backbones.pt"
    if args.stage in {"all", "pretrain"}:
        pretrained = pretrain_spatial_context_backbones(
            ordinary_cache.tokens,
            ordinary_targets,
            embedding_mean=embedding_mean,
            embedding_scale=embedding_scale,
            descriptor_mean=descriptor_mean,
            descriptor_scale=descriptor_scale,
            device=device,
            hidden_dim=int(_required(cfg, "model.hidden_dim")),
            heads=int(_required(cfg, "model.heads")),
            blocks=int(_required(cfg, "model.blocks")),
            rbf_dim=int(_required(cfg, "model.rbf_dim")),
            maximum_radius=float(_required(cfg, "data.radius")),
            representation_dim=int(_required(cfg, "model.representation_dim")),
            dropout=float(_required(cfg, "model.dropout")),
            learning_rate=float(_required(cfg, "pretraining.learning_rate")),
            weight_decay=float(_required(cfg, "pretraining.weight_decay")),
            batch_size=int(_required(cfg, "pretraining.batch_size")),
            maximum_epochs=int(_required(cfg, "pretraining.maximum_epochs")),
            patience=int(_required(cfg, "pretraining.patience")),
            seeds=[int(value) for value in _required(cfg, "training.seeds")],
        )
        save_pretrained_spatial_backbones(pretrained, ordinary_targets, pretraining_path)
    else:
        pretrained = _load_pretrained(pretraining_path)
    write_json(
        output_dir / "ordinary_pretraining_metrics.json",
        {str(seed): values for seed, values in pretrained.metrics.items()},
    )
    if args.stage == "pretrain":
        print(f"[ablation-5] pretraining complete: {pretraining_path}", flush=True)
        return

    fitted = fit_spatial_context_transformer(
        shooting_tokens,
        shooting_targets,
        device=device,
        hidden_dim=int(_required(cfg, "model.hidden_dim")),
        heads=int(_required(cfg, "model.heads")),
        blocks=int(_required(cfg, "model.blocks")),
        rbf_dim=int(_required(cfg, "model.rbf_dim")),
        maximum_radius=float(_required(cfg, "data.radius")),
        representation_dim=int(_required(cfg, "model.representation_dim")),
        dropout=float(_required(cfg, "model.dropout")),
        learning_rate=float(_required(cfg, "training.learning_rate")),
        weight_decay=float(_required(cfg, "training.weight_decay")),
        batch_size=int(_required(cfg, "training.batch_size")),
        maximum_epochs=int(_required(cfg, "training.maximum_epochs")),
        patience=int(_required(cfg, "training.patience")),
        seeds=[int(value) for value in _required(cfg, "training.seeds")],
        initial_backbone_states=pretrained.states,
    )
    feature_variants = build_multiscale_feature_variants(
        base_cache,
        context_cache,
        radial_scales_angstrom=[
            float(value) for value in _required(cfg, "context.radial_scales_angstrom")
        ],
    )
    metrics, arrays = evaluate_distributional_predictor(
        base_cache,
        feature_variants,
        shooting_targets,
        fitted,
        static_pca_dim=int(_required(cfg, "evaluation.static_pca_dim")),
        neighbors=int(_required(cfg, "evaluation.neighbors")),
        seed=int(_required(cfg, "evaluation.seed")),
    )
    metrics["scientific_contract"] = {
        "ablation": 5,
        "only_change_from_accepted_ablation_3": (
            "spatial-transformer backbone initialization from deterministic temporal "
            "pretraining on leakage-safe ordinary trajectories"
        ),
        "ordinary_target": "same-trajectory future embedding changes at 6/12/24 ps",
        "shooting_target": "unchanged empirical sibling distribution RFF kernel mean",
        "encoder": "same frozen GeoFrameTransformerV2 checkpoint",
        "shooting_prediction_head": "newly initialized",
        "ordinary_validation_and_shooting_validation_excluded_from_pretraining_optimization": True,
    }
    metrics["ordinary_pretraining"] = {
        "run_count": len(selected_entries),
        "anchor_count": len(dataset),
        "row_count": int(ordinary_cache.token_z.shape[0]),
        "seed_metrics": {str(seed): value for seed, value in pretrained.metrics.items()},
    }
    baseline_metrics_path = _resolve_path(_required(cfg, "comparison.ablation3_metrics"))
    with baseline_metrics_path.open("r", encoding="utf-8") as handle:
        baseline_metrics = json.load(handle)
    controlled_retrieval_change: dict[str, dict[str, dict[str, float]]] = {}
    for horizon, current_values in metrics["future_neighbor_consistency"].items():
        baseline_values = baseline_metrics["future_neighbor_consistency"][horizon]
        controlled_retrieval_change[horizon] = {}
        for space in (
            "distributional_transformer_representation",
            "predicted_kernel_mean",
        ):
            baseline_gain = float(
                baseline_values["gain_over_local_pca_percent"][space]
            )
            current_gain = float(current_values["gain_over_local_pca_percent"][space])
            baseline_distance = float(
                baseline_values[space]["mean_ensemble_future_distance"]
            )
            current_distance = float(
                current_values[space]["mean_ensemble_future_distance"]
            )
            controlled_retrieval_change[horizon][space] = {
                "ablation3_gain_over_local_pca_percent": baseline_gain,
                "ablation5_gain_over_local_pca_percent": current_gain,
                "gain_change_percentage_points": current_gain - baseline_gain,
                "future_distance_change_percent": 100.0
                * (current_distance / baseline_distance - 1.0),
            }
    metrics["comparison_ablation3"] = {
        "metrics_path": str(baseline_metrics_path),
        "baseline_selected_seed": baseline_metrics["selected_seed"],
        "controlled_retrieval_change": controlled_retrieval_change,
    }
    model = fitted.model
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": fitted.seed,
            "embedding_mean": fitted.embedding_mean,
            "embedding_scale": fitted.embedding_scale,
            "descriptor_mean": fitted.descriptor_mean,
            "descriptor_scale": fitted.descriptor_scale,
            "ordinary_pretraining_checkpoint": str(pretraining_path),
        },
        output_dir / "model.pt",
    )
    save_distributional_preprocessing(
        shooting_targets, output_dir / "target_preprocessing.npz"
    )
    np.savez(output_dir / "coordinates_and_predictions.npz", **arrays)
    write_json(output_dir / "metrics.json", metrics)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_multiscale_training(fitted, plot_dir / "shooting_training.png")
    plot_multiscale_retrieval(
        metrics["future_neighbor_consistency"], plot_dir / "future_distribution_retrieval.png"
    )
    print(
        f"[ablation-5] complete output={output_dir} selected_seed={fitted.seed}", flush=True
    )


if __name__ == "__main__":
    main()
