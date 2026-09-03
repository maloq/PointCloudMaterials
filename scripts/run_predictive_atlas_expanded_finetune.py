#!/usr/bin/env python3
"""Long temporal-atlas training on expanded centers plus encoder fine-tuning."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")

from src.data_utils.shooting_dataset import load_shooting_campaigns_snapshot
from src.temporal_vamp.embeddings import load_frozen_encoder
from src.temporal_vamp.predictive_atlas import (
    FittedPredictiveAtlas,
    PredictiveAtlas,
    _bootstrap_gain,
    build_atlas_baseline_spaces,
    evaluate_predictive_atlas,
    fit_predictive_atlas,
    parent_temperature_conditioning,
    prepare_joint_path_target_data_from_kernel,
    select_atlas_witnesses,
)
from src.temporal_vamp.predictive_atlas_finetune import (
    fit_predictive_atlas_last_geoframe_block,
)
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_encoder_finetune import (
    ShootingGeoFrameActivationCache,
    extract_shooting_geoframe_activation_cache,
)
from src.temporal_vamp.shooting_history import ShootingHistoryEmbeddingCache
from src.temporal_vamp.shooting_multiscale import (
    build_multiscale_feature_variants,
    write_json,
)
from src.temporal_vamp.shooting_spatial import build_spatial_token_data


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Expanded-atlas configuration requires {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def _load_temporal_atlas(path: Path) -> tuple[FittedPredictiveAtlas, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Temporal atlas checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config = dict(payload["model"])
    model = PredictiveAtlas(**model_config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    fitted = FittedPredictiveAtlas(
        model=model,
        embedding_mean=np.asarray(payload["embedding_mean"]),
        embedding_scale=np.asarray(payload["embedding_scale"]),
        descriptor_mean=np.asarray(payload["descriptor_mean"]),
        descriptor_scale=np.asarray(payload["descriptor_scale"]),
        conditioning_mean=np.asarray(payload["conditioning_mean"]),
        conditioning_scale=np.asarray(payload["conditioning_scale"]),
        seed=int(payload["seed"]),
        histories={},
        seed_metrics={},
        predictions_by_seed={},
        representations_by_seed={},
        history_delta_mean=np.asarray(payload["history_delta_mean"]),
        history_delta_scale=np.asarray(payload["history_delta_scale"]),
    )
    return fitted, payload


def _plot_training(
    frozen: FittedPredictiveAtlas,
    fine_tuned: FittedPredictiveAtlas,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed, history in sorted(frozen.histories.items()):
        axes[0].plot(history["selection"], label=f"frozen seed {seed}")
    for seed, history in sorted(fine_tuned.histories.items()):
        axes[1].plot(history["selection"], label=f"fine-tune seed {seed}")
    for axis, title in zip(
        axes,
        ("Expanded frozen encoder", "Last-block encoder fine-tuning"),
    ):
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel("source-run selection MSE")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_retrieval(retrieval: dict[str, Any], path: Path) -> None:
    names = [
        "local_pca_32d",
        "predicted_joint_path_mean_embedding_expanded_frozen",
        "predicted_joint_path_mean_embedding_expanded_finetuned",
        "empirical_joint_path_mean_embedding_oracle",
    ]
    labels = ["static PCA", "expanded frozen", "encoder tuned", "oracle"]
    values = [
        retrieval[name]["mean_heldout_empirical_mmd_distance"] for name in names
    ]
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.bar(labels, values, color=["#777777", "#4472c4", "#c44e52", "#55a868"])
    axis.set_ylabel("held-out empirical future-path distance")
    axis.set_title("Expanded temporal atlas (lower is better)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _checkpoint_payload(
    fitted: FittedPredictiveAtlas,
    *,
    history_spec: dict[str, Any],
) -> dict[str, Any]:
    model = fitted.model
    return {
        "state_dict": model.state_dict(),
        "model": {
            "embedding_dim": model.embedding_dim,
            "descriptor_dim": model.descriptor_dim,
            "conditioning_dim": model.conditioning_dim,
            "hidden_dim": model.hidden_dim,
            "heads": model.heads,
            "blocks": model.block_count,
            "rbf_dim": model.rbf_dim,
            "maximum_radius": model.maximum_radius,
            "latent_dim": model.latent_dim,
            "decoder_hidden_dim": model.decoder_hidden_dim,
            "target_dim": model.target_dim,
            "dropout": model.dropout,
            "history_lag_count": model.history_lag_count,
        },
        "seed": fitted.seed,
        "embedding_mean": fitted.embedding_mean,
        "embedding_scale": fitted.embedding_scale,
        "descriptor_mean": fitted.descriptor_mean,
        "descriptor_scale": fitted.descriptor_scale,
        "conditioning_mean": fitted.conditioning_mean,
        "conditioning_scale": fitted.conditioning_scale,
        "history_delta_mean": fitted.history_delta_mean,
        "history_delta_scale": fitted.history_delta_scale,
        "history_spec": history_spec,
    }


def run(config_path: str | Path, *, stage: str) -> dict[str, Any]:
    cfg: DictConfig = OmegaConf.load(_resolve_path(config_path))
    OmegaConf.resolve(cfg)
    if stage not in {"all", "extract", "train"}:
        raise ValueError(f"Expanded-atlas stage must be all, extract, or train: {stage}")
    output_dir = _resolve_path(_required(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    device = str(_required(cfg, "device"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requests CUDA, but CUDA is unavailable.")

    snapshot = load_shooting_campaigns_snapshot(
        [_resolve_path(value) for value in _required(cfg, "data.campaign_roots")],
        temperatures_K=[
            float(value) for value in _required(cfg, "data.temperatures_K")
        ],
        minimum_complete_branches_per_parent=int(
            _required(cfg, "data.minimum_complete_branches_per_parent")
        ),
    )
    expected = (
        int(_required(cfg, "data.expected_parent_count")),
        int(_required(cfg, "data.expected_branch_count")),
    )
    if (len(snapshot.parents), len(snapshot.branches)) != expected:
        raise RuntimeError(
            f"Expanded snapshot changed: expected={expected}, "
            f"observed={(len(snapshot.parents), len(snapshot.branches))}."
        )
    base_run = _resolve_path(_required(cfg, "initialization.position_atlas_run"))
    cache = ShootingEmbeddingCache.load(base_run / "embeddings")
    context_cache = ShootingContextTokenCache.load(base_run / "context_tokens")
    history_cache = ShootingHistoryEmbeddingCache.load(output_dir / "history_embeddings")
    tokens = build_spatial_token_data(cache, context_cache)
    history_embeddings = np.asarray(history_cache.history_z, dtype=np.float32).reshape(
        tokens.embeddings.shape[0],
        history_cache.history_z.shape[2],
        history_cache.history_z.shape[3],
        history_cache.history_z.shape[4],
    )
    target = prepare_joint_path_target_data_from_kernel(
        cache,
        kernel_path=_resolve_path(_required(cfg, "initialization.fixed_path_kernel")),
        selection_source_velocity_seeds=[
            int(value) for value in _required(cfg, "split.selection_source_velocity_seeds")
        ],
        rff_device=device,
        rff_batch_size=int(_required(cfg, "target.rff_batch_size")),
    )
    frozen_encoder = load_frozen_encoder(
        _resolve_path(_required(cfg, "encoder.checkpoint")),
        device=device,
        repeats=int(_required(cfg, "encoder.repeats")),
        seed=int(_required(cfg, "encoder.seed")),
        representation_source=str(_required(cfg, "encoder.representation_source")),
    )
    activation_path = output_dir / "central_current_layer4_activations"
    if stage in {"all", "extract"}:
        activations = extract_shooting_geoframe_activation_cache(
            snapshot,
            cache,
            context_cache,
            encoder=frozen_encoder,
            cache_path=activation_path,
            num_points=int(_required(cfg, "data.num_points")),
            radius=float(_required(cfg, "data.radius")),
            context_center_count=0,
            point_cloud_batch_size=int(_required(cfg, "encoder.point_cloud_batch_size")),
            environment_batch_size=int(
                _required(cfg, "encoder.environment_batch_size")
            ),
            environment_num_workers=int(
                _required(cfg, "encoder.environment_num_workers")
            ),
            force_recompute=bool(_required(cfg, "cache.force_recompute")),
        )
    else:
        activations = ShootingGeoFrameActivationCache.load(activation_path)
    extraction = {
        "parents": int(cache.parent_z.shape[0]),
        "branches": int(cache.future_z.shape[0]),
        "centers_per_parent": int(cache.parent_z.shape[1]),
        "state_rows": int(tokens.embeddings.shape[0]),
        "history_lags": int(history_embeddings.shape[1]),
        "spatial_tokens": int(tokens.embeddings.shape[1]),
        "fine_tuned_encoder_tokens": "current central token only",
        "activation_shapes": activations.manifest["array_shapes"],
    }
    write_json(output_dir / "extraction_summary.json", extraction)
    if stage == "extract":
        return extraction

    initial_atlas, _ = _load_temporal_atlas(
        _resolve_path(_required(cfg, "initialization.temporal_atlas_checkpoint"))
    )
    frozen_fitted = fit_predictive_atlas(
        tokens,
        target,
        parent_temperature_conditioning(cache),
        device=device,
        hidden_dim=int(_required(cfg, "model.hidden_dim")),
        heads=int(_required(cfg, "model.heads")),
        blocks=int(_required(cfg, "model.blocks")),
        rbf_dim=int(_required(cfg, "model.rbf_dim")),
        maximum_radius=float(_required(cfg, "data.radius")),
        latent_dim=int(_required(cfg, "model.latent_dim")),
        decoder_hidden_dim=int(_required(cfg, "model.decoder_hidden_dim")),
        dropout=float(_required(cfg, "model.dropout")),
        learning_rate=float(_required(cfg, "training.learning_rate")),
        weight_decay=float(_required(cfg, "training.weight_decay")),
        variance_weight=float(_required(cfg, "training.variance_weight")),
        covariance_weight=float(_required(cfg, "training.covariance_weight")),
        batch_size=int(_required(cfg, "training.batch_size")),
        maximum_epochs=int(_required(cfg, "training.maximum_epochs")),
        patience=int(_required(cfg, "training.patience")),
        seeds=[int(value) for value in _required(cfg, "training.seeds")],
        initial_backbone_states=None,
        history_embeddings=history_embeddings,
        initial_model_state=initial_atlas.model.state_dict(),
    )
    torch.save(
        _checkpoint_payload(frozen_fitted, history_spec=history_cache.manifest["spec"]),
        output_dir / "model_expanded_frozen_encoder.pt",
    )

    fine_tuned = fit_predictive_atlas_last_geoframe_block(
        activations,
        frozen_encoder,
        tokens,
        history_embeddings,
        target,
        parent_temperature_conditioning(cache),
        frozen_fitted,
        device=device,
        atlas_learning_rate=float(_required(cfg, "fine_tuning.atlas_learning_rate")),
        encoder_learning_rate=float(
            _required(cfg, "fine_tuning.encoder_learning_rate")
        ),
        weight_decay=float(_required(cfg, "fine_tuning.weight_decay")),
        variance_weight=float(_required(cfg, "fine_tuning.variance_weight")),
        covariance_weight=float(_required(cfg, "fine_tuning.covariance_weight")),
        batch_size=int(_required(cfg, "fine_tuning.batch_size")),
        maximum_epochs=int(_required(cfg, "fine_tuning.maximum_epochs")),
        patience=int(_required(cfg, "fine_tuning.patience")),
        gradient_clip_norm=float(_required(cfg, "fine_tuning.gradient_clip_norm")),
        mixed_precision=bool(_required(cfg, "fine_tuning.mixed_precision")),
        seeds=[int(value) for value in _required(cfg, "fine_tuning.seeds")],
    )
    fine_fitted = fine_tuned.atlas
    fine_payload = _checkpoint_payload(
        fine_fitted, history_spec=history_cache.manifest["spec"]
    )
    fine_payload["fine_tuned_encoder_state"] = fine_tuned.encoder_state
    fine_payload["fine_tuned_encoder_parameter_names"] = (
        fine_tuned.trainable_encoder_parameter_names
    )
    torch.save(fine_payload, output_dir / "model_expanded_finetuned_encoder.pt")

    feature_variants = build_multiscale_feature_variants(
        cache,
        context_cache,
        radial_scales_angstrom=[
            float(value) for value in _required(cfg, "context.radial_scales_angstrom")
        ],
    )
    spaces, vamp = build_atlas_baseline_spaces(
        cache,
        target,
        feature_variants,
        frozen_fitted,
        marginal_prediction=None,
        static_pca_dim=int(_required(cfg, "evaluation.static_pca_dim")),
        vamp_horizon_ps=float(_required(cfg, "evaluation.vamp.horizon_ps")),
        vamp_dimension=int(_required(cfg, "evaluation.vamp.dimension")),
        vamp_regularization=float(_required(cfg, "evaluation.vamp.regularization")),
        vamp_eigenvalue_cutoff=float(
            _required(cfg, "evaluation.vamp.eigenvalue_cutoff")
        ),
    )
    spaces["predicted_joint_path_mean_embedding_expanded_frozen"] = spaces.pop(
        "predicted_joint_path_mean_embedding"
    )
    spaces["history_atlas_latent_32d_expanded_frozen"] = spaces.pop(
        "atlas_latent_32d"
    )
    spaces["predicted_joint_path_mean_embedding_expanded_finetuned"] = (
        fine_fitted.predictions_by_seed[fine_fitted.seed] * target.target_scale
        + target.target_mean
    )
    spaces["history_atlas_latent_32d_expanded_finetuned"] = (
        fine_fitted.representations_by_seed[fine_fitted.seed]
    )
    evaluation, evaluation_arrays, _ = evaluate_predictive_atlas(
        cache,
        target,
        spaces,
        static_space_name=f"local_pca_{int(_required(cfg, 'evaluation.static_pca_dim'))}d",
        neighbors=int(_required(cfg, "evaluation.neighbors")),
        static_caliper_candidates=int(
            _required(cfg, "evaluation.static_caliper_candidates")
        ),
        crystalline_fraction_tolerance=float(
            _required(cfg, "evaluation.crystalline_fraction_tolerance")
        ),
        pairwise_samples=int(_required(cfg, "evaluation.pairwise_samples")),
        exact_mmd_pairs=int(_required(cfg, "evaluation.exact_mmd_pairs")),
        bootstrap_samples=int(_required(cfg, "evaluation.bootstrap_samples")),
        seed=int(_required(cfg, "evaluation.seed")),
    )
    validation_rows = np.asarray(evaluation_arrays["validation_rows"], dtype=np.int64)
    source_ids = np.repeat(
        np.asarray(
            [
                str(parent["source_run_id"])
                for parent in cache.manifest["snapshot"]["parents"]
            ]
        ),
        int(cache.parent_z.shape[1]),
    )[validation_rows]
    direct_gain = _bootstrap_gain(
        np.asarray(
            evaluation_arrays[
                "query_future_distance__predicted_joint_path_mean_embedding_expanded_finetuned"
            ]
        ),
        np.asarray(
            evaluation_arrays[
                "query_future_distance__predicted_joint_path_mean_embedding_expanded_frozen"
            ]
        ),
        source_ids.tolist(),
        samples=int(_required(cfg, "evaluation.bootstrap_samples")),
        seed=int(_required(cfg, "evaluation.seed")) + 211,
    )
    witnesses = select_atlas_witnesses(
        cache,
        target,
        spaces,
        evaluation_arrays,
        static_space_name=f"local_pca_{int(_required(cfg, 'evaluation.static_pca_dim'))}d",
        atlas_space_name="predicted_joint_path_mean_embedding_expanded_finetuned",
        count=int(_required(cfg, "evaluation.witness_count")),
    )
    metrics = {
        "scientific_contract": {
            "expansion": "512 deterministic central atoms per parent versus 64",
            "independent_parent_count_unchanged": 40,
            "future_branches_per_parent": 12,
            "history_lags_ps": [12, 9, 6, 3],
            "future_target_encoder": "frozen GeoFrame teacher",
            "present_encoder_fine_tuning": (
                "GeoFrameV2 final transformer block and norm for the current central "
                "token; satellites and all history tokens remain frozen"
            ),
            "path_kernel": "fixed from the original 64-center atlas",
        },
        "extraction": extraction,
        "target_diagnostics": target.diagnostics,
        "expanded_frozen_training": {
            "selected_seed": int(frozen_fitted.seed),
            "seed_metrics": {
                str(seed): values
                for seed, values in frozen_fitted.seed_metrics.items()
            },
        },
        "encoder_fine_tuning": {
            "selected_seed": int(fine_fitted.seed),
            "trainable_parameter_count": int(
                fine_tuned.trainable_encoder_parameter_count
            ),
            "trainable_parameter_names": list(
                fine_tuned.trainable_encoder_parameter_names
            ),
            "initial_embedding_max_abs_error": float(
                fine_tuned.initial_embedding_max_abs_error
            ),
            "seed_metrics": {
                str(seed): values for seed, values in fine_fitted.seed_metrics.items()
            },
        },
        "evaluation": evaluation,
        "direct_finetune_gain_over_expanded_frozen": direct_gain,
    }
    vamp.save(output_dir / "vamp_baseline.npz")
    np.savez(
        output_dir / "coordinates_and_predictions.npz",
        atom_ids=np.tile(np.asarray(cache.atom_ids), int(cache.parent_z.shape[0])),
        parent_index=np.repeat(
            np.arange(int(cache.parent_z.shape[0])), int(cache.parent_z.shape[1])
        ),
        expanded_frozen_latent=frozen_fitted.representations_by_seed[
            frozen_fitted.seed
        ],
        expanded_frozen_prediction=spaces[
            "predicted_joint_path_mean_embedding_expanded_frozen"
        ],
        expanded_finetuned_latent=fine_fitted.representations_by_seed[
            fine_fitted.seed
        ],
        expanded_finetuned_prediction=spaces[
            "predicted_joint_path_mean_embedding_expanded_finetuned"
        ],
        empirical_joint_path_mean_embedding=target.empirical_mean_embedding,
        **evaluation_arrays,
    )
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "witnesses.json", {"witnesses": witnesses})
    _plot_training(frozen_fitted, fine_fitted, plot_dir / "training.png")
    _plot_retrieval(evaluation["retrieval"], plot_dir / "retrieval.png")
    print(
        f"[predictive-atlas-expanded] complete output={output_dir} "
        f"frozen_seed={frozen_fitted.seed} finetune_seed={fine_fitted.seed}",
        flush=True,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "extract", "train"), default="all")
    args = parser.parse_args()
    run(args.config, stage=args.stage)


if __name__ == "__main__":
    main()
