#!/usr/bin/env python3
"""Train and evaluate the short-horizon shooting velocity control."""

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
from src.temporal_vamp.ordinary_pretraining import PretrainedSpatialBackbones
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_ballistic import (
    evaluate_ballistic_rollout,
    extract_shooting_ballistic_embedding_cache,
    plot_ballistic_rollout,
)
from src.temporal_vamp.shooting_distribution import (
    evaluate_distributional_predictor,
    prepare_distributional_target_data,
    save_distributional_preprocessing,
)
from src.temporal_vamp.shooting_dynamics import ShootingDynamicalFeatureCache
from src.temporal_vamp.shooting_embeddings import (
    ShootingEmbeddingCache,
    extract_shooting_embedding_cache,
)
from src.temporal_vamp.shooting_multiscale import (
    build_multiscale_feature_variants,
    plot_multiscale_retrieval,
    write_json,
)
from src.temporal_vamp.shooting_short_horizon import (
    evaluate_short_horizon_velocity,
    plot_short_horizon_velocity,
)
from src.temporal_vamp.shooting_spatial import (
    build_spatial_token_data,
    fit_spatial_context_transformer,
)


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Short-horizon ablation requires configuration key {path!r}.")
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


def _save_position_model(path: Path, fitted: Any) -> None:
    model = fitted.model
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": int(fitted.seed),
            "embedding_mean": fitted.embedding_mean,
            "embedding_scale": fitted.embedding_scale,
            "descriptor_mean": fitted.descriptor_mean,
            "descriptor_scale": fitted.descriptor_scale,
            "hidden_dim": model.hidden_dim,
            "heads": model.heads,
            "blocks": model.block_count,
            "rbf_dim": model.rbf_dim,
            "maximum_radius": model.maximum_radius,
            "representation_dim": model.representation_dim,
            "target_dim": model.target_dim,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "extract", "train", "evaluate", "ballistic"),
        default="all",
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
            "Short-horizon shooting snapshot changed: "
            f"parents={len(snapshot.parents)} expected={expected_parents}, "
            f"branches={len(snapshot.branches)} expected={expected_branches}."
        )
    write_json(output_dir / "dataset_snapshot.json", snapshot.to_dict())

    embedding_path = output_dir / "embeddings"
    if args.stage in {"all", "extract"}:
        encoder = load_frozen_encoder(
            _resolve_path(_required(cfg, "encoder.checkpoint")),
            device=device,
            repeats=int(_required(cfg, "encoder.repeats")),
            seed=int(_required(cfg, "encoder.seed")),
            representation_source=str(_required(cfg, "encoder.representation_source")),
        )
        cache = extract_shooting_embedding_cache(
            snapshot,
            encoder=encoder,
            cache_path=embedding_path,
            horizons_ps=[float(value) for value in _required(cfg, "target.horizons_ps")],
            center_atom_count=int(_required(cfg, "data.center_atom_count")),
            center_selection_seed=int(_required(cfg, "data.center_selection_seed")),
            num_points=int(_required(cfg, "data.num_points")),
            radius=float(_required(cfg, "data.radius")),
            spatial_context_center_count=0,
            spatial_context_aggregation="mean_std",
            point_cloud_batch_size=int(_required(cfg, "encoder.point_cloud_batch_size")),
            environment_batch_size=int(_required(cfg, "encoder.environment_batch_size")),
            environment_num_workers=int(_required(cfg, "encoder.environment_num_workers")),
            force_recompute=bool(_required(cfg, "cache.force_recompute")),
        )
    else:
        cache = ShootingEmbeddingCache.load(embedding_path)

    reference_cache = ShootingEmbeddingCache.load(
        _resolve_path(_required(cfg, "shooting.reference_embedding_cache"))
    )
    if not np.array_equal(cache.atom_ids, reference_cache.atom_ids):
        raise RuntimeError("Short-horizon cache changed the selected central atom IDs.")
    parent_embedding_error = float(
        np.max(
            np.abs(
                np.asarray(cache.parent_local_z, dtype=np.float32)
                - np.asarray(reference_cache.parent_local_z, dtype=np.float32)
            )
        )
    )
    if parent_embedding_error > 1.0e-5:
        raise RuntimeError(
            "Short-horizon extraction changed the deterministic current embeddings: "
            f"max_abs_error={parent_embedding_error}."
        )
    if args.stage == "extract":
        write_json(
            output_dir / "extraction_summary.json",
            {
                "horizons_ps": np.asarray(cache.horizons_ps).tolist(),
                "future_z_shape": list(cache.future_z.shape),
                "parent_embedding_max_abs_error": parent_embedding_error,
            },
        )
        print(f"[short-horizon] extraction complete: {embedding_path}", flush=True)
        return

    context_cache = ShootingContextTokenCache.load(
        _resolve_path(_required(cfg, "shooting.context_token_cache"))
    )
    dynamics = ShootingDynamicalFeatureCache.load(
        _resolve_path(_required(cfg, "shooting.dynamical_feature_cache"))
    )
    tokens = build_spatial_token_data(cache, context_cache)
    targets = prepare_distributional_target_data(
        cache,
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
    position_arrays_path = output_dir / "position_coordinates_and_predictions.npz"
    position_metrics_path = output_dir / "position_metrics.json"
    if args.stage in {"all", "train"}:
        pretrained = _load_pretrained(
            _resolve_path(_required(cfg, "initialization.ordinary_pretrained_backbones"))
        )
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
            initial_backbone_states=pretrained.states,
        )
        feature_variants = build_multiscale_feature_variants(
            cache,
            context_cache,
            radial_scales_angstrom=[
                float(value) for value in _required(cfg, "context.radial_scales_angstrom")
            ],
        )
        position_metrics, position_arrays = evaluate_distributional_predictor(
            cache,
            feature_variants,
            targets,
            fitted,
            static_pca_dim=int(_required(cfg, "evaluation.static_pca_dim")),
            neighbors=int(_required(cfg, "evaluation.neighbors")),
            seed=int(_required(cfg, "evaluation.seed")),
        )
        position_metrics["scientific_contract"] = {
            "experiment": "short-horizon position-only control",
            "horizons_ps": targets.selected_horizons_ps.tolist(),
            "initialization": "same ordinary-pretrained spatial backbones as ablation 5",
            "target": "empirical sibling future-change RFF kernel mean",
            "split": "same source-run optimization/selection/validation isolation",
        }
        write_json(position_metrics_path, position_metrics)
        np.savez(
            position_arrays_path,
            **{
                name: np.asarray(values, dtype=np.float32)
                for name, values in position_arrays.items()
            },
        )
        _save_position_model(output_dir / "position_model.pt", fitted)
        save_distributional_preprocessing(targets, output_dir / "target_preprocessing.npz")
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        plot_multiscale_retrieval(
            position_metrics["future_neighbor_consistency"],
            plot_dir / "position_retrieval.png",
        )
    else:
        with position_metrics_path.open("r", encoding="utf-8") as handle:
            position_metrics = json.load(handle)
    if args.stage == "train":
        print(f"[short-horizon] position training complete: {output_dir}", flush=True)
        return

    with np.load(position_arrays_path, allow_pickle=False) as payload:
        position_arrays = {name: payload[name].copy() for name in payload.files}
    result = evaluate_short_horizon_velocity(
        cache,
        targets,
        dynamics,
        position_arrays,
        velocity_pca_dimensions=[
            int(value) for value in _required(cfg, "ridge.velocity_pca_dimensions")
        ],
        ridge_alphas=[float(value) for value in _required(cfg, "ridge.alphas")],
        neighbors=int(_required(cfg, "evaluation.neighbors")),
        seed=int(_required(cfg, "evaluation.seed")),
    )
    metrics = {
        "scientific_contract": {
            "experiment": "ablation 7b short-horizon momentum control",
            "horizons_ps": targets.selected_horizons_ps.tolist(),
            "langevin_thermostat_time_ps": 0.3,
            "position_model": "same frozen GeoFrameV2 and ordinary-pretrained spatial architecture as ablation 5",
            "velocity_model": "selected PCA plus ridge residual on individual branch RFF signatures",
            "query_split": "held-out source runs only",
            "candidate_filter": "different source run, exact temperature, exact parent phase",
        },
        "cache_validation": {
            "parent_embedding_max_abs_error": parent_embedding_error,
            "future_z_shape": list(cache.future_z.shape),
            "velocity_feature_dim": int(dynamics.velocity_features.shape[-1]),
        },
        "target_diagnostics": targets.diagnostics,
        "position_only": position_metrics,
        "velocity_conditioning": result.metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    np.savez(output_dir / "velocity_coordinates_and_predictions.npz", **result.arrays)
    np.savez(output_dir / "velocity_ridge_model.npz", **result.model_arrays)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_short_horizon_velocity(
        result.metrics, plot_dir / "short_horizon_velocity.png"
    )
    if args.stage == "evaluate":
        print(f"[short-horizon] velocity evaluation complete: {output_dir}", flush=True)
        return

    ballistic_encoder = load_frozen_encoder(
        _resolve_path(_required(cfg, "encoder.checkpoint")),
        device=device,
        repeats=int(_required(cfg, "encoder.repeats")),
        seed=int(_required(cfg, "encoder.seed")),
        representation_source=str(_required(cfg, "encoder.representation_source")),
    )
    ballistic_cache = extract_shooting_ballistic_embedding_cache(
        snapshot,
        cache,
        encoder=ballistic_encoder,
        cache_path=output_dir / "ballistic_embeddings",
        horizons_ps=[float(value) for value in _required(cfg, "target.horizons_ps")],
        num_points=int(_required(cfg, "data.num_points")),
        radius=float(_required(cfg, "data.radius")),
        point_cloud_batch_size=int(_required(cfg, "encoder.point_cloud_batch_size")),
        environment_batch_size=int(_required(cfg, "encoder.environment_batch_size")),
        environment_num_workers=int(_required(cfg, "encoder.environment_num_workers")),
        force_recompute=bool(_required(cfg, "cache.force_recompute")),
    )
    ballistic_result = evaluate_ballistic_rollout(
        cache,
        targets,
        ballistic_cache,
        position_arrays,
        result,
        ballistic_pca_dimensions=[
            int(value) for value in _required(cfg, "ridge.ballistic_pca_dimensions")
        ],
        ridge_alphas=[float(value) for value in _required(cfg, "ridge.alphas")],
        neighbors=int(_required(cfg, "evaluation.neighbors")),
        seed=int(_required(cfg, "evaluation.seed")),
    )
    metrics["ballistic_rollout"] = ballistic_result.metrics
    metrics["scientific_contract"]["ballistic_control"] = (
        "full time-zero atomwise velocity field propagated without forces and re-encoded "
        "by the same frozen GeoFrameV2 encoder"
    )
    write_json(output_dir / "metrics.json", metrics)
    np.savez(
        output_dir / "ballistic_coordinates_and_predictions.npz",
        **ballistic_result.arrays,
    )
    np.savez(output_dir / "ballistic_ridge_model.npz", **ballistic_result.model_arrays)
    plot_ballistic_rollout(
        ballistic_result.metrics, plot_dir / "ballistic_rollout.png"
    )
    print(f"[short-horizon] complete output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
