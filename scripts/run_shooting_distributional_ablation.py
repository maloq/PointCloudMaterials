#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

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
)


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Distributional shooting configuration requires {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ablation 3: predict multi-bandwidth RFF kernel means of sibling "
            "future-change distributions."
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
    device = str(_required(cfg, "device"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requests CUDA, but CUDA is unavailable.")

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
    tokens = build_spatial_token_data(base_cache, context_cache)
    fitted = fit_spatial_context_transformer(
        tokens,
        targets,
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
    )
    metrics, arrays = evaluate_distributional_predictor(
        base_cache,
        feature_variants,
        targets,
        fitted,
        static_pca_dim=int(_required(cfg, "evaluation.static_pca_dim")),
        neighbors=int(_required(cfg, "evaluation.neighbors")),
        seed=int(_required(cfg, "evaluation.seed")),
    )
    model = fitted.model
    metrics["data_counts"] = {
        "parents": int(base_cache.parent_z.shape[0]),
        "branches": int(base_cache.future_z.shape[0]),
        "center_atoms_per_parent": int(base_cache.parent_z.shape[1]),
        "tokens_per_center": int(tokens.embeddings.shape[1]),
        "futures_per_parent": int(
            np.min(
                np.bincount(
                    np.asarray(base_cache.branch_parent_index, dtype=np.int64)
                )
            )
        ),
    }
    metrics["parameter_count"] = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "descriptor_dim": model.descriptor_dim,
            "hidden_dim": model.hidden_dim,
            "heads": model.heads,
            "blocks": model.block_count,
            "rbf_dim": model.rbf_dim,
            "maximum_radius": model.maximum_radius,
            "representation_dim": model.representation_dim,
            "target_dim": model.target_dim,
            "dropout": model.dropout,
            "seed": fitted.seed,
            "embedding_mean": fitted.embedding_mean,
            "embedding_scale": fitted.embedding_scale,
            "descriptor_mean": fitted.descriptor_mean,
            "descriptor_scale": fitted.descriptor_scale,
        },
        output_dir / "model.pt",
    )
    save_distributional_preprocessing(targets, output_dir / "target_preprocessing.npz")
    np.savez(output_dir / "coordinates_and_predictions.npz", **arrays)
    write_json(output_dir / "metrics.json", metrics)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_multiscale_training(fitted, plot_dir / "training.png")
    plot_multiscale_retrieval(
        metrics["future_neighbor_consistency"],
        plot_dir / "future_distribution_retrieval.png",
    )
    print(
        f"[shooting-distribution] complete output={output_dir} "
        f"selected_seed={fitted.seed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
