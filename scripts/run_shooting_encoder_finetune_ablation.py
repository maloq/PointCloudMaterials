#!/usr/bin/env python3
"""Ablation 6: fine-tune only the final GeoFrameV2 block on shooting targets."""

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

from src.data_utils.shooting_dataset import load_shooting_campaigns_snapshot
from src.temporal_vamp.embeddings import load_frozen_encoder
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_distribution import (
    evaluate_distributional_predictor,
    prepare_distributional_target_data,
    save_distributional_preprocessing,
)
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_encoder_finetune import (
    ShootingGeoFrameActivationCache,
    extract_shooting_geoframe_activation_cache,
    fit_last_geoframe_block_predictor,
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
        raise KeyError(f"Encoder fine-tuning ablation requires configuration key {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def _load_pretrained_backbones(path: Path) -> dict[int, dict[str, torch.Tensor]]:
    if not path.is_file():
        raise FileNotFoundError(f"Ordinary pretrained-backbone checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    states = payload.get("backbone_states")
    if not isinstance(states, dict):
        raise TypeError(f"Checkpoint has no backbone_states dictionary: {path}")
    return {int(seed): state for seed, state in states.items()}


def _controlled_comparison(
    current: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for horizon, current_values in current["future_neighbor_consistency"].items():
        baseline_values = baseline["future_neighbor_consistency"][horizon]
        result[horizon] = {}
        for space in (
            "distributional_transformer_representation",
            "predicted_kernel_mean",
        ):
            current_gain = float(current_values["gain_over_local_pca_percent"][space])
            baseline_gain = float(baseline_values["gain_over_local_pca_percent"][space])
            current_distance = float(
                current_values[space]["mean_ensemble_future_distance"]
            )
            baseline_distance = float(
                baseline_values[space]["mean_ensemble_future_distance"]
            )
            result[horizon][space] = {
                "baseline_gain_over_local_pca_percent": baseline_gain,
                "current_gain_over_local_pca_percent": current_gain,
                "gain_change_percentage_points": current_gain - baseline_gain,
                "future_distance_change_percent": 100.0
                * (current_distance / baseline_distance - 1.0),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "extract", "train"), default="all")
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
    expected_parents = int(_required(cfg, "data.expected_parent_count"))
    expected_branches = int(_required(cfg, "data.expected_branch_count"))
    if len(snapshot.parents) != expected_parents or len(snapshot.branches) != expected_branches:
        raise RuntimeError(
            "Ablation-6 shooting snapshot changed: "
            f"expected={expected_parents}/{expected_branches}, "
            f"observed={len(snapshot.parents)}/{len(snapshot.branches)}."
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
    frozen_encoder = load_frozen_encoder(
        _resolve_path(_required(cfg, "encoder.checkpoint")),
        device=device,
        repeats=int(_required(cfg, "encoder.repeats")),
        seed=int(_required(cfg, "encoder.seed")),
        representation_source=str(_required(cfg, "encoder.representation_source")),
    )
    activation_path = output_dir / "parent_layer4_activations"
    if args.stage in {"all", "extract"}:
        activations = extract_shooting_geoframe_activation_cache(
            snapshot,
            base_cache,
            context_cache,
            encoder=frozen_encoder,
            cache_path=activation_path,
            num_points=int(_required(cfg, "data.num_points")),
            radius=float(_required(cfg, "data.radius")),
            context_center_count=int(_required(cfg, "context.center_count")),
            point_cloud_batch_size=int(_required(cfg, "encoder.point_cloud_batch_size")),
            environment_batch_size=int(_required(cfg, "encoder.environment_batch_size")),
            environment_num_workers=int(_required(cfg, "encoder.environment_num_workers")),
            force_recompute=bool(_required(cfg, "cache.force_recompute")),
        )
    else:
        activations = ShootingGeoFrameActivationCache.load(activation_path)
    if args.stage == "extract":
        write_json(
            output_dir / "extraction_summary.json",
            {
                "parents": int(activations.tokens_before_last.shape[0]),
                "center_atoms": int(activations.tokens_before_last.shape[1]),
                "context_tokens": int(activations.tokens_before_last.shape[2]),
                "group_tokens": int(activations.tokens_before_last.shape[3]),
                "activation_dim": int(activations.tokens_before_last.shape[4]),
            },
        )
        return

    initial_backbones = _load_pretrained_backbones(
        _resolve_path(_required(cfg, "initialization.ordinary_pretrained_backbones"))
    )
    fitted = fit_last_geoframe_block_predictor(
        activations,
        frozen_encoder,
        tokens,
        targets,
        initial_backbone_states=initial_backbones,
        device=device,
        hidden_dim=int(_required(cfg, "model.hidden_dim")),
        heads=int(_required(cfg, "model.heads")),
        blocks=int(_required(cfg, "model.blocks")),
        rbf_dim=int(_required(cfg, "model.rbf_dim")),
        maximum_radius=float(_required(cfg, "data.radius")),
        representation_dim=int(_required(cfg, "model.representation_dim")),
        dropout=float(_required(cfg, "model.dropout")),
        context_learning_rate=float(_required(cfg, "training.context_learning_rate")),
        encoder_learning_rate=float(_required(cfg, "training.encoder_learning_rate")),
        weight_decay=float(_required(cfg, "training.weight_decay")),
        batch_size=int(_required(cfg, "training.batch_size")),
        maximum_epochs=int(_required(cfg, "training.maximum_epochs")),
        patience=int(_required(cfg, "training.patience")),
        gradient_clip_norm=float(_required(cfg, "training.gradient_clip_norm")),
        mixed_precision=bool(_required(cfg, "training.mixed_precision")),
        seeds=[int(value) for value in _required(cfg, "training.seeds")],
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
        targets,
        fitted.spatial,
        static_pca_dim=int(_required(cfg, "evaluation.static_pca_dim")),
        neighbors=int(_required(cfg, "evaluation.neighbors")),
        seed=int(_required(cfg, "evaluation.seed")),
    )
    metrics["scientific_contract"] = {
        "ablation": 6,
        "only_change_from_ablation_5": (
            "fine-tune GeoFrameV2 transformer layer 5 and its final normalization "
            "during shooting optimization"
        ),
        "frozen_encoder_components": (
            "point grouping, patch encoders, pair geometry, transformer layers 0-4, "
            "and VICReg projector"
        ),
        "ordinary_pretraining": "same per-seed spatial backbones as ablation 5",
        "shooting_target_and_evaluation": "unchanged ablation-3 distributional RFF contract",
        "mixed_precision": bool(_required(cfg, "training.mixed_precision")),
    }
    metrics["encoder_fine_tuning"] = {
        "parameter_names": list(fitted.trainable_encoder_parameter_names),
        "parameter_count": fitted.trainable_encoder_parameter_count,
        "initial_embedding_max_abs_error": fitted.initial_embedding_max_abs_error,
        "encoder_learning_rate": float(_required(cfg, "training.encoder_learning_rate")),
        "context_learning_rate": float(_required(cfg, "training.context_learning_rate")),
    }
    comparisons: dict[str, Any] = {}
    for name, path_value in {
        "ablation3": _required(cfg, "comparison.ablation3_metrics"),
        "ablation5": _required(cfg, "comparison.ablation5_metrics"),
    }.items():
        path = _resolve_path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            baseline = json.load(handle)
        comparisons[name] = {
            "metrics_path": str(path),
            "selected_seed": baseline["selected_seed"],
            "controlled_retrieval_change": _controlled_comparison(metrics, baseline),
        }
    metrics["comparisons"] = comparisons
    model = fitted.spatial.model
    torch.save(
        {
            "spatial_state_dict": model.state_dict(),
            "encoder_last_block_state_dict": fitted.encoder_state,
            "encoder_checkpoint": str(frozen_encoder.checkpoint_path),
            "trainable_encoder_parameter_names": fitted.trainable_encoder_parameter_names,
            "seed": fitted.spatial.seed,
            "embedding_mean": fitted.spatial.embedding_mean,
            "embedding_scale": fitted.spatial.embedding_scale,
            "descriptor_mean": fitted.spatial.descriptor_mean,
            "descriptor_scale": fitted.spatial.descriptor_scale,
        },
        output_dir / "model.pt",
    )
    save_distributional_preprocessing(targets, output_dir / "target_preprocessing.npz")
    np.savez(output_dir / "coordinates_and_predictions.npz", **arrays)
    write_json(output_dir / "metrics.json", metrics)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_multiscale_training(fitted.spatial, plot_dir / "shooting_training.png")
    plot_multiscale_retrieval(
        metrics["future_neighbor_consistency"],
        plot_dir / "future_distribution_retrieval.png",
    )
    print(
        f"[ablation-6] complete output={output_dir} selected_seed={fitted.spatial.seed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
