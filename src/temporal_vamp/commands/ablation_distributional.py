#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from src.temporal_vamp.shooting_distribution import (
    evaluate_distributional_predictor,
    save_distributional_preprocessing,
)
from src.temporal_vamp.shooting_multiscale import (
    plot_multiscale_retrieval,
    plot_multiscale_training,
    write_json,
)
from src.temporal_vamp.shooting_spatial import (
    build_spatial_token_data,
    fit_spatial_context_transformer,
)
from src.temporal_vamp.commands.common import (
    required,
    resolve_path,
    prepare_run,
    load_context_features,
    distributional_targets,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ablation 3: predict multi-bandwidth RFF kernel means of sibling "
            "future-change distributions."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = resolve_path(args.config)
    cfg, output_dir = prepare_run(config_path)
    device = str(required(cfg, "device"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requests CUDA, but CUDA is unavailable.")

    base_cache, context_cache, feature_variants = load_context_features(cfg)
    targets = distributional_targets(base_cache, cfg)
    tokens = build_spatial_token_data(base_cache, context_cache)
    fitted = fit_spatial_context_transformer(
        tokens,
        targets,
        device=device,
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
    metrics, arrays = evaluate_distributional_predictor(
        base_cache,
        feature_variants,
        targets,
        fitted,
        static_pca_dim=int(required(cfg, "evaluation.static_pca_dim")),
        neighbors=int(required(cfg, "evaluation.neighbors")),
        seed=int(required(cfg, "evaluation.seed")),
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
