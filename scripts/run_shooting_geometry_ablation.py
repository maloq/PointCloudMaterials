#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
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
from src.temporal_vamp.shooting_geometry import (
    fit_distributional_geometry_transformer,
)
from src.temporal_vamp.shooting_multiscale import (
    build_multiscale_feature_variants,
    plot_multiscale_retrieval,
    plot_multiscale_training,
    write_json,
)
from src.temporal_vamp.shooting_spatial import build_spatial_token_data


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Geometry shooting configuration requires {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ablation 4: add matched cross-run future-neighbour KL and VICReg "
            "to the distributional predictive-state model."
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
    initial_checkpoint_value = OmegaConf.select(
        cfg, "initial_checkpoint", default=None
    )
    initial_state_dict = None
    if initial_checkpoint_value is not None:
        initial_checkpoint_path = _resolve_path(initial_checkpoint_value)
        initial_payload = torch.load(
            initial_checkpoint_path, map_location="cpu", weights_only=False
        )
        initial_state_dict = initial_payload["state_dict"]
        print(
            f"[shooting-geometry] warm_start={initial_checkpoint_path}",
            flush=True,
        )
    fitted, geometry_diagnostics = fit_distributional_geometry_transformer(
        base_cache,
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
        maximum_epochs=int(_required(cfg, "training.maximum_epochs")),
        patience=int(_required(cfg, "training.patience")),
        seeds=[int(value) for value in _required(cfg, "training.seeds")],
        prediction_weight=float(_required(cfg, "objective.prediction_weight")),
        neighbor_kl_weight=float(_required(cfg, "objective.neighbor_kl_weight")),
        teacher_temperature_scale=float(
            _required(cfg, "objective.teacher_temperature_scale")
        ),
        student_temperature_scale=float(
            _required(cfg, "objective.student_temperature_scale")
        ),
        variance_weight=float(_required(cfg, "objective.variance_weight")),
        covariance_weight=float(_required(cfg, "objective.covariance_weight")),
        initial_state_dict=initial_state_dict,
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
    metrics["scientific_contract"].update(
        {
            "ablation": 4,
            "model_change_from_ablation_3": (
                "matched cross-run future-neighbour KL plus VICReg only"
            ),
            "excluded_changes": "no pretraining, velocity, or encoder fine-tuning",
        }
    )
    metrics["geometry_objective"] = geometry_diagnostics
    seed_retrieval: dict[str, Any] = {}
    for seed_value in sorted(fitted.predictions_by_seed):
        seed_fitted = replace(fitted, seed=int(seed_value))
        seed_metrics, _ = evaluate_distributional_predictor(
            base_cache,
            feature_variants,
            targets,
            seed_fitted,
            static_pca_dim=int(_required(cfg, "evaluation.static_pca_dim")),
            neighbors=int(_required(cfg, "evaluation.neighbors")),
            seed=int(_required(cfg, "evaluation.seed")),
        )
        seed_retrieval[str(seed_value)] = {
            horizon: {
                "representation": values["distributional_transformer_representation"],
                "predicted_kernel_mean": values["predicted_kernel_mean"],
                "gain_over_local_pca_percent": {
                    name: values["gain_over_local_pca_percent"][name]
                    for name in (
                        "distributional_transformer_representation",
                        "predicted_kernel_mean",
                    )
                },
            }
            for horizon, values in seed_metrics["future_neighbor_consistency"].items()
        }
    metrics["seed_future_neighbor_consistency"] = seed_retrieval
    model = fitted.model
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
    all_seed_arrays = {
        **arrays,
        **{
            f"seed_{seed_value}_prediction": prediction
            for seed_value, prediction in fitted.predictions_by_seed.items()
        },
        **{
            f"seed_{seed_value}_representation": representation
            for seed_value, representation in fitted.representations_by_seed.items()
        },
    }
    np.savez(output_dir / "coordinates_and_predictions.npz", **all_seed_arrays)
    write_json(output_dir / "metrics.json", metrics)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_multiscale_training(fitted, plot_dir / "training.png")
    plot_multiscale_retrieval(
        metrics["future_neighbor_consistency"],
        plot_dir / "future_distribution_retrieval.png",
    )
    print(
        f"[shooting-geometry] complete output={output_dir} "
        f"selected_seed={fitted.seed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
