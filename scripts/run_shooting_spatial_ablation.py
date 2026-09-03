#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_multiscale import (
    build_multiscale_feature_variants,
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


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Spatial shooting configuration requires {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ablation 2: invariant spatial context transformer with the same "
            "mean future-change target as ablation 1."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = _resolve_path(args.config)
    cfg: DictConfig = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    output_dir = _resolve_path(_required(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    requested_device = str(_required(cfg, "device"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={requested_device!r} requests CUDA, but CUDA is unavailable."
        )

    base_cache = ShootingEmbeddingCache.load(
        _resolve_path(_required(cfg, "base_embedding_cache"))
    )
    context_cache = ShootingContextTokenCache.load(
        _resolve_path(_required(cfg, "context_token_cache"))
    )
    feature_variants = build_multiscale_feature_variants(
        base_cache,
        context_cache,
        radial_scales_angstrom=[
            float(value) for value in _required(cfg, "context.radial_scales_angstrom")
        ],
    )
    targets = prepare_dynamic_target_data(
        base_cache,
        horizons_ps=[float(value) for value in _required(cfg, "target.horizons_ps")],
        target_pca_dim=int(_required(cfg, "target.pca_dim_per_horizon")),
        selection_source_velocity_seeds=[
            int(value) for value in _required(cfg, "split.selection_source_velocity_seeds")
        ],
        residual_ridge_alphas=[
            float(value) for value in _required(cfg, "target.residual_ridge_alphas")
        ],
    )
    tokens = build_spatial_token_data(base_cache, context_cache)
    fitted = fit_spatial_context_transformer(
        tokens,
        targets,
        device=requested_device,
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
    )
    metrics, arrays = evaluate_spatial_context_transformer(
        base_cache,
        feature_variants,
        targets,
        fitted,
        static_pca_dim=int(_required(cfg, "evaluation.static_pca_dim")),
        neighbors=int(_required(cfg, "evaluation.neighbors")),
        seed=int(_required(cfg, "evaluation.seed")),
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
