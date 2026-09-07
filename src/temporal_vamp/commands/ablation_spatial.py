#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from src.temporal_vamp.shooting_multiscale import (
    plot_multiscale_retrieval,
    plot_multiscale_training,
    prepare_dynamic_target_data,
    write_json,
)
from src.temporal_vamp.shooting_spatial import (
    build_spatial_token_data,
    evaluate_spatial_context_transformer,
    fit_spatial_context_transformer,
    save_spatial_context_transformer,
)
from src.temporal_vamp.commands.common import (
    required,
    resolve_path,
    prepare_run,
    load_context_features,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ablation 2: invariant spatial context transformer with the same "
            "mean future-change target as ablation 1."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = resolve_path(args.config)
    cfg, output_dir = prepare_run(config_path)
    requested_device = str(required(cfg, "device"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={requested_device!r} requests CUDA, but CUDA is unavailable."
        )

    base_cache, context_cache, feature_variants = load_context_features(cfg)
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
    tokens = build_spatial_token_data(base_cache, context_cache)
    fitted = fit_spatial_context_transformer(
        tokens,
        targets,
        device=requested_device,
        hidden_dim=int(required(cfg, "model.hidden_dim")),
        heads=int(required(cfg, "model.heads")),
        blocks=int(required(cfg, "model.blocks")),
        rbf_dim=int(required(cfg, "model.rbf_dim")),
        maximum_radius=float(required(cfg, "data.radius")),
        representation_dim=int(required(cfg, "model.representation_dim")),
        dropout=float(required(cfg, "model.dropout")),
        learning_rate=float(required(cfg, "training.learning_rate")),
        weight_decay=float(required(cfg, "training.weight_decay")),
        batch_size=int(required(cfg, "training.batch_size")),
        maximum_epochs=int(required(cfg, "training.maximum_epochs")),
        patience=int(required(cfg, "training.patience")),
        seeds=[int(value) for value in required(cfg, "training.seeds")],
    )
    metrics, arrays = evaluate_spatial_context_transformer(
        base_cache,
        feature_variants,
        targets,
        fitted,
        static_pca_dim=int(required(cfg, "evaluation.static_pca_dim")),
        neighbors=int(required(cfg, "evaluation.neighbors")),
        seed=int(required(cfg, "evaluation.seed")),
    )
    metrics["data_counts"] = {
        "parents": int(base_cache.parent_z.shape[0]),
        "branches": int(base_cache.future_z.shape[0]),
        "center_atoms_per_parent": int(base_cache.parent_z.shape[1]),
        "tokens_per_center": int(tokens.embeddings.shape[1]),
    }
    metrics["parameter_count"] = int(
        sum(parameter.numel() for parameter in fitted.model.parameters())
    )
    save_spatial_context_transformer(fitted, targets, output_dir / "model.pt")
    np.savez(output_dir / "coordinates_and_predictions.npz", **arrays)
    write_json(output_dir / "metrics.json", metrics)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_multiscale_training(fitted, plot_dir / "training.png")
    plot_multiscale_retrieval(
        metrics["future_neighbor_consistency"],
        plot_dir / "future_change_retrieval.png",
    )
    print(
        f"[shooting-spatial] complete output={output_dir} "
        f"selected_seed={fitted.seed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
