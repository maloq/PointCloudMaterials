#!/usr/bin/env python3
"""Train a joint-path predictive atlas with explicit source-trajectory history."""

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
    _bootstrap_gain,
    compute_pullback_spectrum,
    evaluate_predictive_atlas,
    fit_predictive_atlas,
    parent_temperature_conditioning,
    prepare_joint_path_target_data,
    save_path_kernel,
    select_atlas_witnesses,
)
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_history import (
    ShootingHistoryEmbeddingCache,
    extract_shooting_history_embedding_cache,
)
from src.temporal_vamp.shooting_multiscale import write_json
from src.temporal_vamp.shooting_spatial import build_spatial_token_data


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"History-atlas configuration requires {path!r}.")
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


def _load_baseline_spaces(path: Path, row_count: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Position-only atlas coordinates are missing: {path}")
    spaces: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as payload:
        for key in payload.files:
            if key.startswith("space__"):
                values = payload[key].copy()
                if values.shape[0] != int(row_count):
                    raise RuntimeError(
                        f"Baseline space {key} has {values.shape[0]} rows; "
                        f"the history experiment requires {row_count}."
                    )
                spaces[key.removeprefix("space__")] = values
    required = {"local_pca_32d", "predicted_joint_path_mean_embedding"}
    if not required.issubset(spaces):
        raise RuntimeError(
            f"Position-only atlas lacks required spaces: {sorted(required - spaces.keys())}."
        )
    return spaces


def _plot_training(histories: dict[int, dict[str, list[float]]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed, history in sorted(histories.items()):
        axes[0].plot(history["optimization"], label=f"seed {seed}")
        axes[1].plot(history["selection"], label=f"seed {seed}")
    for axis, title in zip(axes, ("Optimization MSE", "Source-run selection MSE")):
        axis.set_title(title)
        axis.set_xlabel("fine-tuning epoch")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_comparison(retrieval: dict[str, Any], path: Path) -> None:
    names = [
        "local_pca_32d",
        "predicted_joint_path_mean_embedding",
        "predicted_joint_path_mean_embedding_history",
        "empirical_joint_path_mean_embedding_oracle",
    ]
    labels = ["static PCA", "position atlas", "history atlas", "oracle"]
    values = [
        retrieval[name]["mean_heldout_empirical_mmd_distance"] for name in names
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values, color=["#777777", "#4472c4", "#c44e52", "#55a868"])
    axis.set_ylabel("held-out empirical future-path distance")
    axis.set_title("Matched cross-run future-law retrieval (lower is better)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config_path: str | Path, *, stage: str) -> dict[str, Any]:
    config_file = _resolve_path(config_path)
    cfg: DictConfig = OmegaConf.load(config_file)
    OmegaConf.resolve(cfg)
    resolved_stage = str(stage).lower()
    if resolved_stage not in {"all", "extract", "train"}:
        raise ValueError(f"History-atlas stage must be all, extract, or train: {stage!r}")
    output_dir = _resolve_path(_required(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
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
    expected = (
        int(_required(cfg, "data.expected_parent_count")),
        int(_required(cfg, "data.expected_branch_count")),
    )
    observed = (len(snapshot.parents), len(snapshot.branches))
    if observed != expected:
        raise RuntimeError(
            f"History-atlas snapshot violates its data contract: "
            f"expected={expected}, observed={observed}."
        )

    base_run = _resolve_path(_required(cfg, "initialization.position_atlas_run"))
    cache = ShootingEmbeddingCache.load(base_run / "embeddings")
    context_cache = ShootingContextTokenCache.load(base_run / "context_tokens")
    history_path = output_dir / "history_embeddings"
    if resolved_stage in {"all", "extract"}:
        encoder = load_frozen_encoder(
            _resolve_path(_required(cfg, "encoder.checkpoint")),
            device=device,
            repeats=int(_required(cfg, "encoder.repeats")),
            seed=int(_required(cfg, "encoder.seed")),
            representation_source=str(_required(cfg, "encoder.representation_source")),
        )
        history_cache = extract_shooting_history_embedding_cache(
            snapshot,
            cache,
            context_cache,
            encoder=encoder,
            source_trajectory_root=_resolve_path(
                _required(cfg, "history.source_trajectory_root")
            ),
            cache_path=history_path,
            lag_frames=[int(value) for value in _required(cfg, "history.lag_frames")],
            lag_times_ps=[
                float(value) for value in _required(cfg, "history.lag_times_ps")
            ],
            source_sample_interval_ps=float(
                _required(cfg, "history.source_sample_interval_ps")
            ),
            num_points=int(_required(cfg, "data.num_points")),
            radius=float(_required(cfg, "data.radius")),
            context_center_count=int(_required(cfg, "context.center_count")),
            point_cloud_batch_size=int(
                _required(cfg, "encoder.point_cloud_batch_size")
            ),
            force_recompute=bool(_required(cfg, "cache.force_recompute")),
        )
    else:
        history_cache = ShootingHistoryEmbeddingCache.load(history_path)
    extraction = {
        "parents": int(history_cache.history_z.shape[0]),
        "center_atoms_per_parent": int(history_cache.history_z.shape[1]),
        "history_lags": int(history_cache.history_z.shape[2]),
        "tokens_per_center": int(history_cache.history_z.shape[3]),
        "embedding_dim": int(history_cache.history_z.shape[4]),
        "lag_times_ps_oldest_to_newest": history_cache.manifest["spec"][
            "lag_times_ps_oldest_to_newest"
        ],
        "maximum_current_embedding_error": history_cache.manifest[
            "current_embedding_max_abs_error"
        ],
    }
    write_json(output_dir / "extraction_summary.json", extraction)
    if resolved_stage == "extract":
        return extraction

    tokens = build_spatial_token_data(cache, context_cache)
    history_embeddings = np.asarray(history_cache.history_z, dtype=np.float32).reshape(
        tokens.embeddings.shape[0],
        history_cache.history_z.shape[2],
        history_cache.history_z.shape[3],
        history_cache.history_z.shape[4],
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
    position_model_path = base_run / "model.pt"
    if not position_model_path.is_file():
        raise FileNotFoundError(
            f"Position-only atlas checkpoint is missing: {position_model_path}"
        )
    position_model = torch.load(
        position_model_path, map_location="cpu", weights_only=False
    )
    fitted = fit_predictive_atlas(
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
        initial_model_state=position_model["state_dict"],
    )
    raw_prediction = (
        fitted.predictions_by_seed[fitted.seed] * target.target_scale
        + target.target_mean
    )
    spaces = _load_baseline_spaces(
        base_run / "coordinates_and_predictions.npz", tokens.embeddings.shape[0]
    )
    spaces["predicted_joint_path_mean_embedding_history"] = raw_prediction
    spaces[f"history_atlas_latent_{fitted.model.latent_dim}d"] = (
        fitted.representations_by_seed[fitted.seed]
    )
    static_name = f"local_pca_{int(_required(cfg, 'evaluation.static_pca_dim'))}d"
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
    selected_rows = np.asarray(evaluation_arrays["validation_rows"], dtype=np.int64)
    source_ids = np.repeat(
        np.asarray(
            [
                str(parent["source_run_id"])
                for parent in cache.manifest["snapshot"]["parents"]
            ]
        ),
        int(cache.parent_z.shape[1]),
    )[selected_rows]
    direct_gain = _bootstrap_gain(
        np.asarray(
            evaluation_arrays[
                "query_future_distance__predicted_joint_path_mean_embedding_history"
            ]
        ),
        np.asarray(
            evaluation_arrays[
                "query_future_distance__predicted_joint_path_mean_embedding"
            ]
        ),
        source_ids.tolist(),
        samples=int(_required(cfg, "evaluation.bootstrap_samples")),
        seed=int(_required(cfg, "evaluation.seed")) + 101,
    )
    witnesses = select_atlas_witnesses(
        cache,
        target,
        spaces,
        evaluation_arrays,
        static_space_name=static_name,
        atlas_space_name="predicted_joint_path_mean_embedding_history",
        count=int(_required(cfg, "evaluation.witness_count")),
    )
    rng = np.random.default_rng(int(_required(cfg, "evaluation.seed")))
    pullback_rows = np.sort(
        rng.choice(
            target.split_rows["validation"],
            size=min(
                int(_required(cfg, "evaluation.pullback.maximum_states")),
                int(target.split_rows["validation"].size),
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
    metrics = {
        "scientific_contract": {
            "input": (
                "current 17-token invariant spatial environment, temperature, and "
                "the same token identities at t-12/t-9/t-6/t-3 ps"
            ),
            "history_is_past_only": True,
            "target": (
                "conditional mean embedding of same-branch 6/12/24 ps future paths"
            ),
            "comparison": (
                "warm-started position-only atlas, identical target, splits, "
                "candidate caliper, and evaluation seed"
            ),
        },
        "extraction": extraction,
        "training": {
            "selected_seed": int(fitted.seed),
            "seed_metrics": {
                str(seed): value for seed, value in fitted.seed_metrics.items()
            },
        },
        "evaluation": evaluation,
        "direct_gain_over_position_only": direct_gain,
        "pullback": {
            "sampled_states": int(pullback_rows.size),
            "effective_rank_minimum": int(pullback_rank.min()),
            "effective_rank_median": float(np.median(pullback_rank)),
            "effective_rank_maximum": int(pullback_rank.max()),
        },
        "parameter_count": int(
            sum(parameter.numel() for parameter in fitted.model.parameters())
        ),
    }
    torch.save(
        {
            "state_dict": fitted.model.state_dict(),
            "model": {
                "embedding_dim": fitted.model.embedding_dim,
                "descriptor_dim": fitted.model.descriptor_dim,
                "conditioning_dim": fitted.model.conditioning_dim,
                "hidden_dim": fitted.model.hidden_dim,
                "heads": fitted.model.heads,
                "blocks": fitted.model.block_count,
                "rbf_dim": fitted.model.rbf_dim,
                "maximum_radius": fitted.model.maximum_radius,
                "latent_dim": fitted.model.latent_dim,
                "decoder_hidden_dim": fitted.model.decoder_hidden_dim,
                "target_dim": fitted.model.target_dim,
                "dropout": fitted.model.dropout,
                "history_lag_count": fitted.model.history_lag_count,
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
            "history_spec": history_cache.manifest["spec"],
            "position_initialization": str(position_model_path),
        },
        output_dir / "model.pt",
    )
    save_path_kernel(target, output_dir / "path_kernel.npz")
    np.savez(
        output_dir / "coordinates_and_predictions.npz",
        history_atlas_latent=fitted.representations_by_seed[fitted.seed],
        predicted_joint_path_mean_embedding_history=raw_prediction,
        pullback_rows=pullback_rows,
        pullback_eigenvalues=pullback_eigenvalues,
        pullback_effective_rank=pullback_rank,
        **evaluation_arrays,
    )
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "witnesses.json", {"witnesses": witnesses})
    _plot_training(fitted.histories, plot_dir / "training.png")
    _plot_comparison(evaluation["retrieval"], plot_dir / "retrieval_comparison.png")
    print(
        f"[predictive-atlas-history] complete output={output_dir} seed={fitted.seed}",
        flush=True,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a joint future-path predictive atlas with explicit past context."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "extract", "train"), default="all")
    args = parser.parse_args()
    run(args.config, stage=args.stage)


if __name__ == "__main__":
    main()
