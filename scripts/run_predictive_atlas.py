#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    build_atlas_baseline_spaces,
    compute_pullback_spectrum,
    evaluate_predictive_atlas,
    fit_predictive_atlas,
    parent_temperature_conditioning,
    prepare_joint_path_target_data,
    save_path_kernel,
    select_atlas_witnesses,
)
from src.temporal_vamp.shooting_context import (
    ShootingContextTokenCache,
    extract_shooting_context_token_cache,
)
from src.temporal_vamp.shooting_distribution import prepare_distributional_target_data
from src.temporal_vamp.shooting_embeddings import (
    ShootingEmbeddingCache,
    extract_shooting_embedding_cache,
)
from src.temporal_vamp.shooting_multiscale import (
    build_multiscale_feature_variants,
    write_json,
)
from src.temporal_vamp.shooting_spatial import (
    build_spatial_token_data,
    fit_spatial_context_transformer,
)


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Predictive-atlas configuration requires {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve_device(raw: str) -> str:
    requested = str(raw).strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={raw!r} requests CUDA, but torch.cuda.is_available() is false."
        )
    return str(raw)


def _load_pretrained_backbones(path: Path) -> dict[int, dict[str, torch.Tensor]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Ordinary temporal-pretraining checkpoint is missing: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "backbone_states" not in payload or not isinstance(
        payload["backbone_states"], dict
    ):
        raise RuntimeError(
            f"Pretraining checkpoint has no backbone_states mapping: {path}"
        )
    return {
        int(seed): {name: value for name, value in state.items()}
        for seed, state in payload["backbone_states"].items()
    }


def _plot_training(histories: dict[int, dict[str, list[float]]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed, history in sorted(histories.items()):
        axes[0].plot(history["optimization"], label=f"seed {seed}")
        axes[1].plot(history["selection"], label=f"seed {seed}")
    axes[0].set_title("Optimization prediction MSE")
    axes[1].set_title("Source-run selection MSE")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_retrieval(metrics: dict[str, Any], path: Path) -> None:
    retrieval = metrics["retrieval"]
    names = list(retrieval)
    values = [retrieval[name]["distance_over_matched_random"] for name in names]
    figure, axis = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(names))
    axis.bar(positions, values, color="#4472c4")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(positions, names, rotation=35, ha="right")
    axis.set_ylabel("held-out future-path distance / matched random")
    axis.set_title("Predictive-atlas cross-run retrieval (lower is better)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_atlas(
    representations: np.ndarray,
    optimization_rows: np.ndarray,
    temperatures: np.ndarray,
    phases: np.ndarray,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(representations, dtype=np.float64)
    mean = values[optimization_rows].mean(axis=0)
    centered = values[optimization_rows] - mean
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    components = right[:2]
    coordinates = (values - mean) @ components.T
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    scatter = axes[0].scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=temperatures,
        s=7,
        alpha=0.55,
        cmap="viridis",
    )
    figure.colorbar(scatter, ax=axes[0], label="temperature (K)")
    for phase in sorted(set(phases.tolist())):
        rows = phases == phase
        axes[1].scatter(
            coordinates[rows, 0],
            coordinates[rows, 1],
            s=7,
            alpha=0.55,
            label=str(phase),
        )
    for axis in axes:
        axis.set_xlabel("atlas latent PCA 1")
        axis.set_ylabel("atlas latent PCA 2")
        axis.grid(alpha=0.2)
    axes[0].set_title("Temperature")
    axes[1].set_title("Parent phase")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return coordinates, mean, components


def _plot_pullback(eigenvalues: np.ndarray, effective_rank: np.ndarray, path: Path) -> None:
    normalized = eigenvalues / np.maximum(eigenvalues[:, -1:], 1.0e-12)
    quantiles = np.quantile(normalized[:, ::-1], [0.1, 0.5, 0.9], axis=0)
    modes = np.arange(1, normalized.shape[1] + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(modes, quantiles[1], color="#c44e52", label="median")
    axes[0].fill_between(
        modes, quantiles[0], quantiles[2], color="#c44e52", alpha=0.25
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("pullback eigenmode")
    axes[0].set_ylabel("eigenvalue / largest eigenvalue")
    axes[0].grid(alpha=0.25)
    unique, counts = np.unique(effective_rank, return_counts=True)
    axes[1].bar(unique, counts)
    axes[1].set_xlabel("effective pullback rank")
    axes[1].set_ylabel("sampled states")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _metadata_arrays(cache: ShootingEmbeddingCache) -> dict[str, np.ndarray]:
    parents = cache.manifest["snapshot"]["parents"]
    parent_count, center_count = cache.parent_z.shape[:2]
    return {
        "parent_index": np.repeat(np.arange(parent_count), center_count),
        "atom_id": np.tile(np.asarray(cache.atom_ids, dtype=np.int64), parent_count),
        "temperature_K": np.repeat(
            np.asarray([float(parent["temperature_K"]) for parent in parents]),
            center_count,
        ),
        "phase": np.repeat(
            np.asarray([str(parent["phase"]) for parent in parents]), center_count
        ),
        "source_run_id": np.repeat(
            np.asarray([str(parent["source_run_id"]) for parent in parents]),
            center_count,
        ),
        "source_split": np.repeat(
            np.asarray([str(parent["source_split"]) for parent in parents]),
            center_count,
        ),
    }


def run(config_path: str | Path, *, stage: str) -> dict[str, Any]:
    config_file = _resolve_path(config_path)
    cfg: DictConfig = OmegaConf.load(config_file)
    OmegaConf.resolve(cfg)
    resolved_stage = str(stage).lower()
    if resolved_stage not in {"all", "extract", "train"}:
        raise ValueError(
            f"Predictive-atlas stage must be all, extract, or train; got {stage!r}."
        )
    output_dir = _resolve_path(_required(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    device = _resolve_device(str(_required(cfg, "device")))
    snapshot = load_shooting_campaigns_snapshot(
        [_resolve_path(value) for value in _required(cfg, "data.campaign_roots")],
        temperatures_K=[
            float(value) for value in _required(cfg, "data.temperatures_K")
        ],
        minimum_complete_branches_per_parent=int(
            _required(cfg, "data.minimum_complete_branches_per_parent")
        ),
    )
    expected_parents = int(_required(cfg, "data.expected_parent_count"))
    expected_branches = int(_required(cfg, "data.expected_branch_count"))
    if len(snapshot.parents) != expected_parents or len(snapshot.branches) != expected_branches:
        raise RuntimeError(
            "Predictive-atlas snapshot violates the configured data contract: "
            f"expected parents/branches={expected_parents}/{expected_branches}, "
            f"observed={len(snapshot.parents)}/{len(snapshot.branches)}."
        )
    write_json(output_dir / "dataset_snapshot.json", snapshot.to_dict())

    embedding_path = output_dir / "embeddings"
    context_path = output_dir / "context_tokens"
    if resolved_stage in {"all", "extract"}:
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
            horizons_ps=[float(value) for value in _required(cfg, "data.horizons_ps")],
            center_atom_count=int(_required(cfg, "data.center_atom_count")),
            center_selection_seed=int(_required(cfg, "data.center_selection_seed")),
            num_points=int(_required(cfg, "data.num_points")),
            radius=float(_required(cfg, "data.radius")),
            spatial_context_center_count=int(
                _required(cfg, "data.legacy_context_center_count")
            ),
            spatial_context_aggregation="mean_std",
            point_cloud_batch_size=int(_required(cfg, "encoder.point_cloud_batch_size")),
            environment_batch_size=int(
                _required(cfg, "encoder.environment_batch_size")
            ),
            environment_num_workers=int(
                _required(cfg, "encoder.environment_num_workers")
            ),
            force_recompute=bool(_required(cfg, "cache.force_recompute")),
        )
        context_cache = extract_shooting_context_token_cache(
            snapshot,
            cache,
            encoder=encoder,
            cache_path=context_path,
            num_points=int(_required(cfg, "data.num_points")),
            radius=float(_required(cfg, "data.radius")),
            context_center_count=int(_required(cfg, "context.center_count")),
            point_cloud_batch_size=int(_required(cfg, "encoder.point_cloud_batch_size")),
            environment_batch_size=int(
                _required(cfg, "encoder.environment_batch_size")
            ),
            environment_num_workers=int(
                _required(cfg, "encoder.environment_num_workers")
            ),
            steinhardt_shell_min_neighbors=int(
                _required(cfg, "context.steinhardt_shell_min_neighbors")
            ),
            steinhardt_shell_max_neighbors=int(
                _required(cfg, "context.steinhardt_shell_max_neighbors")
            ),
            force_recompute=bool(_required(cfg, "cache.force_recompute")),
        )
    else:
        cache = ShootingEmbeddingCache.load(embedding_path)
        context_cache = ShootingContextTokenCache.load(context_path)
    extraction = {
        "parents": int(cache.parent_z.shape[0]),
        "branches": int(cache.future_z.shape[0]),
        "futures_per_parent": int(
            np.bincount(np.asarray(cache.branch_parent_index, dtype=np.int64)).min()
        ),
        "center_atoms_per_parent": int(cache.parent_z.shape[1]),
        "tokens_per_center": int(context_cache.satellite_z.shape[2] + 1),
        "horizons_ps": np.asarray(cache.horizons_ps).tolist(),
        "encoder_checkpoint": str(cache.manifest["spec"]["checkpoint"]),
    }
    write_json(output_dir / "extraction_summary.json", extraction)
    if resolved_stage == "extract":
        return extraction

    tokens = build_spatial_token_data(cache, context_cache)
    feature_variants = build_multiscale_feature_variants(
        cache,
        context_cache,
        radial_scales_angstrom=[
            float(value) for value in _required(cfg, "context.radial_scales_angstrom")
        ],
    )
    target = prepare_joint_path_target_data(
        cache,
        horizons_ps=[float(value) for value in _required(cfg, "target.horizons_ps")],
        horizon_weights=[
            float(value) for value in _required(cfg, "target.horizon_weights")
        ],
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
        rff_device=device,
        rff_batch_size=int(_required(cfg, "target.rff_batch_size")),
    )
    pretrained = _load_pretrained_backbones(
        _resolve_path(_required(cfg, "initialization.pretrained_backbones"))
    )
    marginal_target = prepare_distributional_target_data(
        cache,
        horizons_ps=[
            float(value) for value in _required(cfg, "marginal_baseline.horizons_ps")
        ],
        change_pca_dim=int(_required(cfg, "marginal_baseline.change_pca_dim")),
        rff_features_per_bandwidth=int(
            _required(cfg, "marginal_baseline.rff_features_per_bandwidth")
        ),
        bandwidth_multipliers=[
            float(value)
            for value in _required(cfg, "marginal_baseline.bandwidth_multipliers")
        ],
        selection_source_velocity_seeds=[
            int(value) for value in _required(cfg, "split.selection_source_velocity_seeds")
        ],
        seed=int(_required(cfg, "marginal_baseline.seed")),
    )
    marginal_fitted = fit_spatial_context_transformer(
        tokens,
        marginal_target,
        device=device,
        hidden_dim=int(_required(cfg, "model.hidden_dim")),
        heads=int(_required(cfg, "model.heads")),
        blocks=int(_required(cfg, "model.blocks")),
        rbf_dim=int(_required(cfg, "model.rbf_dim")),
        maximum_radius=float(_required(cfg, "data.radius")),
        representation_dim=int(_required(cfg, "marginal_baseline.representation_dim")),
        dropout=float(_required(cfg, "model.dropout")),
        learning_rate=float(_required(cfg, "training.learning_rate")),
        weight_decay=float(_required(cfg, "training.weight_decay")),
        batch_size=int(_required(cfg, "training.batch_size")),
        maximum_epochs=int(_required(cfg, "training.maximum_epochs")),
        patience=int(_required(cfg, "training.patience")),
        seeds=[int(value) for value in _required(cfg, "training.seeds")],
        initial_backbone_states=pretrained,
    )
    marginal_prediction = (
        marginal_fitted.predictions_by_seed[marginal_fitted.seed]
        * marginal_target.target_scale
        + marginal_target.target_mean
    )
    conditioning = parent_temperature_conditioning(cache)
    fitted = fit_predictive_atlas(
        tokens,
        target,
        conditioning,
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
        initial_backbone_states=pretrained,
    )
    spaces, vamp = build_atlas_baseline_spaces(
        cache,
        target,
        feature_variants,
        fitted,
        marginal_prediction=marginal_prediction,
        static_pca_dim=int(_required(cfg, "evaluation.static_pca_dim")),
        vamp_horizon_ps=float(_required(cfg, "evaluation.vamp.horizon_ps")),
        vamp_dimension=int(_required(cfg, "evaluation.vamp.dimension")),
        vamp_regularization=float(_required(cfg, "evaluation.vamp.regularization")),
        vamp_eigenvalue_cutoff=float(
            _required(cfg, "evaluation.vamp.eigenvalue_cutoff")
        ),
    )
    static_name = f"local_pca_{int(_required(cfg, 'evaluation.static_pca_dim'))}d"
    atlas_name = "predicted_joint_path_mean_embedding"
    evaluation, evaluation_arrays, _ = evaluate_predictive_atlas(
        cache,
        target,
        spaces,
        static_space_name=static_name,
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
    witnesses = select_atlas_witnesses(
        cache,
        target,
        spaces,
        evaluation_arrays,
        static_space_name=static_name,
        atlas_space_name=atlas_name,
        count=int(_required(cfg, "evaluation.witness_count")),
    )
    rng = np.random.default_rng(int(_required(cfg, "evaluation.seed")))
    validation_rows = target.split_rows["validation"]
    pullback_rows = np.sort(
        rng.choice(
            validation_rows,
            size=min(
                int(_required(cfg, "evaluation.pullback.maximum_states")),
                int(validation_rows.size),
            ),
            replace=False,
        )
    )
    pullback_eigenvalues, pullback_rank = compute_pullback_spectrum(
        fitted,
        rows=pullback_rows,
        device=device,
        batch_size=int(_required(cfg, "evaluation.pullback.batch_size")),
        relative_eigenvalue_cutoff=float(
            _required(cfg, "evaluation.pullback.relative_eigenvalue_cutoff")
        ),
    )
    metadata = _metadata_arrays(cache)
    atlas_coordinates, atlas_pca_mean, atlas_pca_components = _plot_atlas(
        fitted.representations_by_seed[fitted.seed],
        target.split_rows["optimization"],
        metadata["temperature_K"],
        metadata["phase"],
        output_dir / "plots" / "atlas_latent.png",
    )
    pullback_summary = {
        "sampled_states": int(pullback_rows.size),
        "relative_eigenvalue_cutoff": float(
            _required(cfg, "evaluation.pullback.relative_eigenvalue_cutoff")
        ),
        "effective_rank_minimum": int(pullback_rank.min()),
        "effective_rank_median": float(np.median(pullback_rank)),
        "effective_rank_maximum": int(pullback_rank.max()),
    }
    metrics = {
        "scientific_contract": {
            "input": (
                "one parent position state, temperature, and 17 invariant GeoFrameV2 "
                "central/satellite context tokens"
            ),
            "target": (
                "conditional mean embedding of the joint 6/12/24 ps future-change "
                "representation path across 12 independent shooting branches"
            ),
            "claim_scope": (
                "future laws on the frozen GeoFrameV2 target-representation space, "
                "not injective raw point-cloud laws"
            ),
            "primary_distance": (
                "Euclidean chordal distance between predicted path mean embeddings"
            ),
        },
        "data_counts": extraction,
        "target_diagnostics": target.diagnostics,
        "atlas_training": {
            "selected_seed": int(fitted.seed),
            "seed_metrics": {str(key): value for key, value in fitted.seed_metrics.items()},
        },
        "marginal_baseline_training": {
            "selected_seed": int(marginal_fitted.seed),
            "seed_metrics": {
                str(key): value for key, value in marginal_fitted.seed_metrics.items()
            },
        },
        "evaluation": evaluation,
        "pullback": pullback_summary,
        "parameter_count": int(
            sum(parameter.numel() for parameter in fitted.model.parameters())
        ),
    }

    plot_dir = output_dir / "plots"
    _plot_training(fitted.histories, plot_dir / "training.png")
    _plot_retrieval(evaluation, plot_dir / "future_path_retrieval.png")
    _plot_pullback(
        pullback_eigenvalues, pullback_rank, plot_dir / "pullback_spectrum.png"
    )
    model = fitted.model
    torch.save(
        {
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
            },
            "seed": fitted.seed,
            "embedding_mean": fitted.embedding_mean,
            "embedding_scale": fitted.embedding_scale,
            "descriptor_mean": fitted.descriptor_mean,
            "descriptor_scale": fitted.descriptor_scale,
            "conditioning_mean": fitted.conditioning_mean,
            "conditioning_scale": fitted.conditioning_scale,
            "pretrained_backbones": str(
                _resolve_path(_required(cfg, "initialization.pretrained_backbones"))
            ),
        },
        output_dir / "model.pt",
    )
    save_path_kernel(target, output_dir / "path_kernel.npz")
    vamp.save(output_dir / "vamp_baseline.npz")
    coordinate_payload = {
        **metadata,
        "atlas_latent": fitted.representations_by_seed[fitted.seed],
        "atlas_latent_pca2": atlas_coordinates,
        "atlas_latent_pca_mean": atlas_pca_mean,
        "atlas_latent_pca_components": atlas_pca_components,
        "predicted_joint_path_mean_embedding": spaces[atlas_name],
        "empirical_joint_path_mean_embedding": target.empirical_mean_embedding,
        "predicted_marginal_mean_embeddings": marginal_prediction,
        "pullback_rows": pullback_rows,
        "pullback_eigenvalues": pullback_eigenvalues,
        "pullback_effective_rank": pullback_rank,
        **evaluation_arrays,
    }
    for name, values in spaces.items():
        coordinate_payload[f"space__{name}"] = values
    np.savez(output_dir / "coordinates_and_predictions.npz", **coordinate_payload)
    write_json(output_dir / "witnesses.json", {"witnesses": witnesses})
    write_json(output_dir / "metrics.json", metrics)
    print(
        f"[predictive-atlas] complete output={output_dir} seed={fitted.seed}",
        flush=True,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a low-dimensional conditional mean-embedding atlas of joint "
            "shooting future paths."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "extract", "train"), default="all")
    args = parser.parse_args()
    run(args.config, stage=args.stage)


if __name__ == "__main__":
    main()
