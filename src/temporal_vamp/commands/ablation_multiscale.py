#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data_utils.shooting_dataset import (
    load_shooting_campaign_snapshot,
    load_shooting_campaigns_snapshot,
)
from src.temporal_vamp.embeddings import load_frozen_encoder
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_context import (
    ShootingContextTokenCache,
    extract_shooting_context_token_cache,
)
from src.temporal_vamp.shooting_multiscale import (
    build_multiscale_feature_variants,
    evaluate_multiscale_ablation,
    fit_multiscale_mlp,
    fit_ridge_feature_variants,
    plot_multiscale_retrieval,
    plot_multiscale_training,
    prepare_dynamic_target_data,
    save_multiscale_predictor,
    write_json,
)
from src.temporal_vamp.commands.common import (
    required,
    resolve_path,
    prepare_run,
)


def _resolve_device(raw: str) -> str:
    requested = str(raw).strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={raw!r} requests CUDA, but torch.cuda.is_available() is false."
        )
    return str(raw)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ablation 1: q4/q6 and radial multiscale context on frozen "
            "GeoFrameV2 parent embeddings."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("all", "extract", "train"), default="all"
    )
    args = parser.parse_args(argv)

    config_path = resolve_path(args.config)
    cfg, output_dir = prepare_run(config_path)
    configured_roots = OmegaConf.select(cfg, "data.campaign_roots", default=None)
    temperatures = [float(value) for value in required(cfg, "data.temperatures_K")]
    minimum = int(required(cfg, "data.minimum_complete_branches_per_parent"))
    if configured_roots is None:
        snapshot = load_shooting_campaign_snapshot(
            resolve_path(required(cfg, "data.campaign_root")),
            temperatures_K=temperatures,
            minimum_complete_branches_per_parent=minimum,
        )
    else:
        snapshot = load_shooting_campaigns_snapshot(
            [resolve_path(value) for value in configured_roots],
            temperatures_K=temperatures,
            minimum_complete_branches_per_parent=minimum,
        )
    expected_parents = int(required(cfg, "data.expected_parent_count"))
    expected_branches = int(required(cfg, "data.expected_branch_count"))
    if len(snapshot.parents) != expected_parents or len(snapshot.branches) != expected_branches:
        raise RuntimeError(
            "The multiscale ablation snapshot violates the configured data contract: "
            f"expected parents/branches={expected_parents}/{expected_branches}, "
            f"observed={len(snapshot.parents)}/{len(snapshot.branches)}."
        )
    write_json(output_dir / "dataset_snapshot.json", snapshot.to_dict())

    base_cache = ShootingEmbeddingCache.load(
        resolve_path(required(cfg, "base_embedding_cache"))
    )
    context_cache_path = output_dir / "context_tokens"
    device = _resolve_device(str(required(cfg, "device")))
    if args.stage in {"all", "extract"}:
        encoder = load_frozen_encoder(
            resolve_path(required(cfg, "encoder.checkpoint")),
            device=device,
            repeats=int(required(cfg, "encoder.repeats")),
            seed=int(required(cfg, "encoder.seed")),
            representation_source=str(required(cfg, "encoder.representation_source")),
        )
        print(
            f"[shooting-multiscale] extracting on device={device}, "
            f"checkpoint={encoder.checkpoint_path}",
            flush=True,
        )
        context_cache = extract_shooting_context_token_cache(
            snapshot,
            base_cache,
            encoder=encoder,
            cache_path=context_cache_path,
            num_points=int(required(cfg, "data.num_points")),
            radius=float(required(cfg, "data.radius")),
            context_center_count=int(required(cfg, "context.center_count")),
            point_cloud_batch_size=int(
                required(cfg, "encoder.point_cloud_batch_size")
            ),
            environment_batch_size=int(
                required(cfg, "encoder.environment_batch_size")
            ),
            environment_num_workers=int(
                required(cfg, "encoder.environment_num_workers")
            ),
            steinhardt_shell_min_neighbors=int(
                required(cfg, "context.steinhardt_shell_min_neighbors")
            ),
            steinhardt_shell_max_neighbors=int(
                required(cfg, "context.steinhardt_shell_max_neighbors")
            ),
            force_recompute=bool(required(cfg, "cache.force_recompute")),
        )
    else:
        context_cache = ShootingContextTokenCache.load(context_cache_path)
    if args.stage == "extract":
        write_json(
            output_dir / "extraction_summary.json",
            {
                "device": device,
                "parents": int(context_cache.satellite_z.shape[0]),
                "center_atoms": int(context_cache.satellite_z.shape[1]),
                "satellites_per_center": int(context_cache.satellite_z.shape[2]),
                "embedding_dim": int(context_cache.satellite_z.shape[3]),
            },
        )
        return

    feature_variants = build_multiscale_feature_variants(
        base_cache,
        context_cache,
        radial_scales_angstrom=[
            float(value) for value in required(cfg, "context.radial_scales_angstrom")
        ],
    )
    targets = prepare_dynamic_target_data(
        base_cache,
        horizons_ps=[float(value) for value in required(cfg, "target.horizons_ps")],
        target_pca_dim=int(required(cfg, "target.pca_dim_per_horizon")),
        selection_source_velocity_seeds=[
            int(value) for value in required(cfg, "split.selection_source_velocity_seeds")
        ],
        residual_ridge_alphas=[
            float(value) for value in required(cfg, "target.residual_ridge_alphas")
        ],
    )
    ridge_predictions, ridge_metrics, ridge_parameters = fit_ridge_feature_variants(
        feature_variants,
        targets,
        ridge_alphas=[float(value) for value in required(cfg, "ridge.alphas")],
    )
    fitted = fit_multiscale_mlp(
        feature_variants["multiscale_context"],
        targets,
        device=device,
        hidden_dim=int(required(cfg, "model.hidden_dim")),
        representation_dim=int(required(cfg, "model.representation_dim")),
        dropout=float(required(cfg, "model.dropout")),
        learning_rate=float(required(cfg, "training.learning_rate")),
        weight_decay=float(required(cfg, "training.weight_decay")),
        batch_size=int(required(cfg, "training.batch_size")),
        maximum_epochs=int(required(cfg, "training.maximum_epochs")),
        patience=int(required(cfg, "training.patience")),
        seeds=[int(value) for value in required(cfg, "training.seeds")],
    )
    metrics, arrays = evaluate_multiscale_ablation(
        base_cache,
        feature_variants,
        targets,
        ridge_predictions,
        fitted,
        static_pca_dim=int(required(cfg, "evaluation.static_pca_dim")),
        neighbors=int(required(cfg, "evaluation.neighbors")),
        seed=int(required(cfg, "evaluation.seed")),
    )
    metrics["ridge_prediction"] = ridge_metrics
    metrics["feature_dimensions"] = {
        name: int(values.shape[1]) for name, values in feature_variants.items()
    }
    metrics["data_counts"] = {
        "parents": int(base_cache.parent_z.shape[0]),
        "branches": int(base_cache.future_z.shape[0]),
        "center_atoms_per_parent": int(base_cache.parent_z.shape[1]),
        "satellites_per_center": int(context_cache.satellite_z.shape[2]),
    }
    save_multiscale_predictor(fitted, targets, output_dir / "model.pt")
    ridge_payload = {
        f"{variant}__{parameter}": values
        for variant, parameters in ridge_parameters.items()
        for parameter, values in parameters.items()
    }
    np.savez(output_dir / "ridge_models.npz", **ridge_payload)
    np.savez(output_dir / "coordinates_and_predictions.npz", **arrays)
    write_json(output_dir / "metrics.json", metrics)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_multiscale_training(fitted, plot_dir / "training.png")
    plot_multiscale_retrieval(
        metrics["future_neighbor_consistency"], plot_dir / "future_change_retrieval.png"
    )
    print(
        f"[shooting-multiscale] complete output={output_dir} "
        f"selected_seed={fitted.seed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
