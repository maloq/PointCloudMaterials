#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.temporal_vamp.predictability_map import (
    aggregate_by_parent,
    branch_features_for_horizon,
    build_input_ladder,
    fit_dense_probe,
    json_ready,
    parent_metadata,
    plot_atlas_metadata,
    plot_input_ladder,
    plot_latent_vector_field,
    plot_noise_ceiling,
    plot_predictability_heatmap,
    plot_shooting_fans,
    predictive_latent_dynamics,
    regression_metrics,
    save_fitted_probe,
    source_bootstrap_r2,
    split_shot_noise_ceiling,
    static_predictive_disagreement,
)
from src.temporal_vamp.predictive_atlas import prepare_joint_path_target_data_from_kernel
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_distribution import (
    prepare_distributional_target_data,
    save_distributional_preprocessing,
)
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Predictability-map configuration requires {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def _probe_kwargs(cfg: DictConfig) -> dict[str, Any]:
    return {
        "device": str(_required(cfg, "device")),
        "hidden_dim": int(_required(cfg, "model.hidden_dim")),
        "latent_dim": int(_required(cfg, "model.latent_dim")),
        "dropout": float(_required(cfg, "model.dropout")),
        "learning_rate": float(_required(cfg, "training.learning_rate")),
        "weight_decay": float(_required(cfg, "training.weight_decay")),
        "batch_size": int(_required(cfg, "training.batch_size")),
        "maximum_epochs": int(_required(cfg, "training.maximum_epochs")),
        "patience": int(_required(cfg, "training.patience")),
        "seeds": [int(value) for value in _required(cfg, "training.seeds")],
    }


def _parent_rows(parent_indices: np.ndarray, center_count: int) -> np.ndarray:
    return np.concatenate(
        [
            np.arange(
                int(parent) * int(center_count),
                (int(parent) + 1) * int(center_count),
                dtype=np.int64,
            )
            for parent in parent_indices.tolist()
        ]
    )


def _branch_validation_metrics(
    parent_prediction: np.ndarray,
    branch_target: np.ndarray,
    branch_parent: np.ndarray,
    validation_parents: np.ndarray,
) -> dict[str, float]:
    center_count = int(parent_prediction.shape[0] // (int(branch_parent.max()) + 1))
    selected_branches = np.flatnonzero(np.isin(branch_parent, validation_parents))
    predictions = np.stack(
        [
            parent_prediction[
                int(parent) * center_count : (int(parent) + 1) * center_count
            ]
            for parent in branch_parent[selected_branches].tolist()
        ],
        axis=0,
    ).reshape(-1, parent_prediction.shape[-1])
    targets = branch_target[selected_branches].reshape(-1, branch_target.shape[-1])
    rows = np.arange(targets.shape[0], dtype=np.int64)
    return regression_metrics(predictions, targets, rows)


def _leave_one_out_sibling_reference_metrics(
    branch_target: np.ndarray,
    branch_parent: np.ndarray,
    validation_parents: np.ndarray,
) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for parent in validation_parents.tolist():
        branches = np.flatnonzero(branch_parent == int(parent))
        total = branch_target[branches].sum(axis=0)
        for branch in branches.tolist():
            predictions.append((total - branch_target[branch]) / float(branches.size - 1))
            targets.append(branch_target[branch])
    prediction = np.stack(predictions).reshape(-1, branch_target.shape[-1])
    target = np.stack(targets).reshape(-1, branch_target.shape[-1])
    return regression_metrics(
        prediction, target, np.arange(target.shape[0], dtype=np.int64)
    )


def _write_csv_tables(
    output_dir: Path,
    horizons: list[float],
    noise: dict[str, Any],
    horizon_metrics: dict[str, Any],
    ladder_metrics: dict[str, Any],
) -> None:
    with (output_dir / "predictability_by_horizon.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "horizon_ps",
                "target",
                "validation_r2",
                "finite_shot_reference_or_reliability",
            ]
        )
        for horizon in horizons:
            key = f"{horizon:g}ps"
            ceilings = {
                "individual_future": horizon_metrics[key]["individual_sibling_reference"]["r2"],
                "mean_future": noise[key]["mean_future"][
                    "estimated_full_shot_reliability_mean"
                ],
                "future_law": noise[key]["future_law"][
                    "estimated_full_shot_reliability_mean"
                ],
                "log_variance": noise[key]["log_variance"][
                    "estimated_full_shot_reliability_mean"
                ],
            }
            scores = {
                "individual_future": horizon_metrics[key]["individual_future"]["r2"],
                "mean_future": horizon_metrics[key]["mean_future"]["validation"]["r2"],
                "future_law": horizon_metrics[key]["future_law"]["validation"]["r2"],
                "log_variance": horizon_metrics[key]["log_variance"]["validation"]["r2"],
            }
            for target in scores:
                writer.writerow([horizon, target, scores[target], ceilings[target]])
    with (output_dir / "input_ladder.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "input",
                "input_dim",
                "selected_seed",
                "selection_r2",
                "validation_r2",
                "source_bootstrap_low",
                "source_bootstrap_high",
            ]
        )
        for name, values in ladder_metrics.items():
            writer.writerow(
                [
                    name,
                    values["input_dim"],
                    values["selected_seed"],
                    values["selection"]["r2"],
                    values["validation"]["r2"],
                    values["bootstrap"]["ci95"][0],
                    values["bootstrap"]["ci95"][1],
                ]
            )


def _write_summary(
    output_dir: Path,
    metrics: dict[str, Any],
    config_path: Path,
) -> None:
    ladder = metrics["input_ladder"]
    best_name = metrics["selected_horizon_input"]
    best = ladder[best_name]
    horizon = metrics["predictability_by_horizon"]
    shot_count = int(metrics["data"]["shots_per_parent"])
    lines = [
        "# Predictability map: results",
        "",
        "This folder measures which aspects of the Al MEAM shooting dynamics are reproducibly predictable from the present state.",
        "",
        "## Data and protocol",
        "",
        f"- {metrics['data']['parents']} parent configurations, {metrics['data']['branches']} independent shooting branches, {metrics['data']['shots_per_parent']} shots per parent.",
        f"- {metrics['data']['center_atoms_per_parent']} fixed atom IDs per parent; horizons {metrics['data']['horizons_ps']} ps.",
        "- Optimization/model-selection/final validation are separated by source MD run.",
        f"- Split-shot reliability uses repeated random {shot_count // 2}+{shot_count // 2} branch partitions on final-validation parents.",
        f"- The individual-future reference is a finite-sample leave-one-shot-out mean of the other {shot_count - 1} siblings. It is not a theoretical ceiling: a model trained across many parents can denoise better than this noisy reference.",
        "",
        "## Main results",
        "",
        f"The input selected on model-selection data was **{best_name}** (validation joint-law R2={best['validation']['r2']:.3f}, source-bootstrap 95% interval {best['bootstrap']['ci95'][0]:.3f} to {best['bootstrap']['ci95'][1]:.3f}).",
        "",
        "| Horizon | Individual future R2 | Conditional mean R2 | Future-law R2 | Log-variance R2 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for key, value in horizon.items():
        lines.append(
            f"| {key[:-2]} | {value['individual_future']['r2']:.3f} | "
            f"{value['mean_future']['validation']['r2']:.3f} | "
            f"{value['future_law']['validation']['r2']:.3f} | "
            f"{value['log_variance']['validation']['r2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpret distributional model scores together with `plots/noise_ceiling.png`: a low score on a low-reliability target is primarily a data/shot-statistics limitation. The individual-future comparison in `plots/predictability_heatmap.png` is instead a finite-sibling reference, not an upper bound.",
            "",
            "The shooting-fan and vector-field figures use the local-only predictive probe because that same map can be evaluated on every cached future local GeoFrame embedding. It is a diagnostic projection, not the full 17-token production atlas. Future-state application is out-of-training-distribution and is labelled accordingly.",
            "",
            "## Files",
            "",
            "- `metrics.json`: complete numerical results and scientific contracts.",
            "- `predictability_by_horizon.csv`: compact target/horizon table.",
            "- `input_ladder.csv`: controlled present-information ablation.",
            "- `analysis_arrays.npz`: latent coordinates and shooting trajectories used by the figures.",
            "- `models/`: selected probe parameters and preprocessing statistics.",
            "- `plots/`: publication-style diagnostics.",
            f"- `resolved_config.yaml`: resolved copy of `{config_path}`.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path) -> dict[str, Any]:
    resolved_config_path = _resolve_path(config_path)
    cfg: DictConfig = OmegaConf.load(resolved_config_path)
    OmegaConf.resolve(cfg)
    output_dir = _resolve_path(_required(cfg, "output_dir"))
    if output_dir.exists():
        raise FileExistsError(
            f"Predictability-map output already exists and will not be overwritten: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    (output_dir / "plots").mkdir()
    (output_dir / "models").mkdir()
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    device = str(_required(cfg, "device"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requests CUDA, but CUDA is unavailable.")
    cache = ShootingEmbeddingCache.load(
        _resolve_path(_required(cfg, "data.embedding_cache"))
    )
    context = ShootingContextTokenCache.load(
        _resolve_path(_required(cfg, "data.context_token_cache"))
    )
    history_path = _resolve_path(_required(cfg, "data.history_embeddings"))
    history = np.load(history_path, mmap_mode="r")
    horizons = [float(value) for value in _required(cfg, "target.horizons_ps")]
    selection_seeds = [
        int(value) for value in _required(cfg, "split.selection_source_velocity_seeds")
    ]
    print("[predictability-map] constructing horizon targets", flush=True)
    distribution_targets = prepare_distributional_target_data(
        cache,
        horizons_ps=horizons,
        change_pca_dim=int(_required(cfg, "target.change_pca_dim")),
        rff_features_per_bandwidth=int(
            _required(cfg, "target.rff_features_per_bandwidth")
        ),
        bandwidth_multipliers=[
            float(value) for value in _required(cfg, "target.bandwidth_multipliers")
        ],
        selection_source_velocity_seeds=selection_seeds,
        seed=int(_required(cfg, "target.seed")),
    )
    save_distributional_preprocessing(
        distribution_targets, output_dir / "horizon_target_preprocessing.npz"
    )
    joint_targets = prepare_joint_path_target_data_from_kernel(
        cache,
        kernel_path=_resolve_path(_required(cfg, "target.joint_path_kernel")),
        selection_source_velocity_seeds=selection_seeds,
        rff_device=device,
        rff_batch_size=int(_required(cfg, "target.rff_batch_size")),
    )
    if not all(
        np.array_equal(distribution_targets.split_rows[name], joint_targets.split_rows[name])
        for name in ("optimization", "selection", "validation")
    ):
        raise RuntimeError("Horizon and joint-path targets resolved different data splits.")

    print("[predictability-map] building controlled input ladder", flush=True)
    inputs = build_input_ladder(
        cache, context, history, seed=int(_required(cfg, "evaluation.seed"))
    )
    metadata = parent_metadata(cache)
    probe_kwargs = _probe_kwargs(cfg)
    ladder_metrics: dict[str, Any] = {}
    fitted_ladder = {}
    for name, values in inputs.items():
        print(f"[predictability-map] fitting input variant {name}", flush=True)
        fitted = fit_dense_probe(
            values,
            joint_targets.target_modes,
            joint_targets.split_rows,
            **probe_kwargs,
        )
        save_fitted_probe(fitted, output_dir / "models" / f"input_{name}.pt")
        selected_metrics = fitted.seed_metrics[fitted.selected_seed]
        ladder_metrics[name] = {
            "input_dim": int(values.shape[1]),
            "selected_seed": int(fitted.selected_seed),
            "selection": selected_metrics["selection"],
            "validation": selected_metrics["validation"],
            "bootstrap": source_bootstrap_r2(
                fitted.prediction,
                joint_targets.target_modes,
                joint_targets.split_rows["validation"],
                metadata["source"],
                samples=int(_required(cfg, "evaluation.bootstrap_samples")),
                seed=int(_required(cfg, "evaluation.seed")),
            ),
            "seed_metrics": fitted.seed_metrics,
        }
        fitted_ladder[name] = fitted
    selected_input = max(
        ladder_metrics,
        key=lambda name: (
            float(ladder_metrics[name]["selection"]["r2"]), name
        ),
    )
    print(f"[predictability-map] selected horizon input={selected_input}", flush=True)

    noise_metrics: dict[str, Any] = {}
    horizon_metrics: dict[str, Any] = {}
    selected_values = inputs[selected_input]
    parent_count, center_count = cache.parent_local_z.shape[:2]
    validation_parents = distribution_targets.parent_splits["validation"]
    for horizon_index, horizon in enumerate(horizons):
        key = f"{horizon:g}ps"
        print(f"[predictability-map] horizon={horizon:g} ps", flush=True)
        projected, branch_rff, branch_parent = branch_features_for_horizon(
            cache, distribution_targets, horizon_index
        )
        parent_mean, parent_variance = aggregate_by_parent(
            projected, branch_parent, parent_count
        )
        mean_target = parent_mean.reshape(parent_count * center_count, -1)
        variance_target = np.log(
            parent_variance.reshape(parent_count * center_count, -1) + 1.0e-6
        ).astype(np.float32)
        law_target = distribution_targets.distribution_signature[:, horizon_index]
        noise_metrics[key] = split_shot_noise_ceiling(
            projected,
            branch_rff,
            branch_parent,
            validation_parents,
            repetitions=int(_required(cfg, "evaluation.shot_split_repetitions")),
            seed=int(_required(cfg, "evaluation.seed")) + horizon_index,
        )
        fitted_targets = {}
        target_metrics: dict[str, Any] = {}
        for target_name, values in (
            ("mean_future", mean_target),
            ("future_law", law_target),
            ("log_variance", variance_target),
        ):
            print(
                f"[predictability-map] fitting horizon={horizon:g} target={target_name}",
                flush=True,
            )
            fitted = fit_dense_probe(
                selected_values,
                values,
                distribution_targets.split_rows,
                **probe_kwargs,
            )
            save_fitted_probe(
                fitted, output_dir / "models" / f"horizon_{horizon:g}ps_{target_name}.pt"
            )
            selected_metrics = fitted.seed_metrics[fitted.selected_seed]
            target_metrics[target_name] = {
                "selected_seed": fitted.selected_seed,
                "selection": selected_metrics["selection"],
                "validation": selected_metrics["validation"],
                "seed_metrics": fitted.seed_metrics,
            }
            fitted_targets[target_name] = fitted
        target_metrics["individual_future"] = _branch_validation_metrics(
            fitted_targets["mean_future"].prediction,
            projected,
            branch_parent,
            validation_parents,
        )
        target_metrics["individual_sibling_reference"] = _leave_one_out_sibling_reference_metrics(
            projected, branch_parent, validation_parents
        )
        horizon_metrics[key] = target_metrics

    model_scores: dict[str, dict[str, float]] = {}
    ceilings: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        key = f"{horizon:g}ps"
        model_scores[key] = {
            "individual_future": float(horizon_metrics[key]["individual_future"]["r2"]),
            "mean_future": float(horizon_metrics[key]["mean_future"]["validation"]["r2"]),
            "future_law": float(horizon_metrics[key]["future_law"]["validation"]["r2"]),
            "log_variance": float(horizon_metrics[key]["log_variance"]["validation"]["r2"]),
        }
        ceilings[key] = {
            "individual_future": float(horizon_metrics[key]["individual_sibling_reference"]["r2"]),
            "mean_future": float(noise_metrics[key]["mean_future"]["estimated_full_shot_reliability_mean"]),
            "future_law": float(noise_metrics[key]["future_law"]["estimated_full_shot_reliability_mean"]),
            "log_variance": float(noise_metrics[key]["log_variance"]["estimated_full_shot_reliability_mean"]),
        }

    shot_count = int(np.bincount(np.asarray(cache.branch_parent_index, dtype=np.int64))[0])
    plot_noise_ceiling(
        noise_metrics,
        horizons,
        output_dir / "plots" / "noise_ceiling.png",
        shot_count=shot_count,
    )
    plot_predictability_heatmap(
        model_scores,
        ceilings,
        horizons,
        output_dir / "plots" / "predictability_heatmap.png",
    )
    plot_input_ladder(ladder_metrics, output_dir / "plots" / "input_ladder.png")
    plot_atlas_metadata(
        _resolve_path(_required(cfg, "accepted_atlas.coordinates")),
        metadata,
        output_dir / "plots" / "accepted_atlas_metadata.png",
    )

    local_fitted = fitted_ladder["local"]
    horizon_indices = [
        int(np.flatnonzero(np.isclose(cache.horizons_ps, horizon))[0])
        for horizon in horizons
    ]
    dynamics = predictive_latent_dynamics(
        cache,
        local_fitted,
        joint_targets.split_rows,
        device=device,
        batch_size=int(_required(cfg, "training.batch_size")),
        horizon_indices=horizon_indices,
    )
    fan_records = plot_shooting_fans(
        dynamics,
        cache,
        horizons,
        output_dir / "plots" / "predictive_latent_shooting_fans.png",
    )
    vector_horizon = float(_required(cfg, "evaluation.vector_field_horizon_ps"))
    vector_position_matches = np.flatnonzero(np.isclose(horizons, vector_horizon))
    if vector_position_matches.size != 1:
        raise ValueError(
            f"Vector-field horizon {vector_horizon:g} ps is not in {horizons}."
        )
    plot_latent_vector_field(
        dynamics,
        cache,
        int(vector_position_matches[0]),
        joint_targets.split_rows["validation"],
        output_dir / "plots" / "predictive_latent_vector_field.png",
    )

    best_fitted = fitted_ladder[selected_input]
    predicted_joint_embedding = (
        best_fitted.prediction * joint_targets.target_scale + joint_targets.target_mean
    ).astype(np.float32)
    disagreement = static_predictive_disagreement(
        cache,
        joint_targets,
        predicted_joint_embedding,
        samples=int(_required(cfg, "evaluation.pair_samples")),
        seed=int(_required(cfg, "evaluation.seed")),
        plot_path=output_dir / "plots" / "static_vs_predictive_disagreement.png",
    )

    accepted_position_metrics = json.loads(
        _resolve_path(_required(cfg, "accepted_atlas.position_metrics")).read_text(
            encoding="utf-8"
        )
    )
    accepted_history_metrics = json.loads(
        _resolve_path(_required(cfg, "accepted_atlas.history_metrics")).read_text(
            encoding="utf-8"
        )
    )
    shots_per_parent = np.bincount(
        np.asarray(cache.branch_parent_index, dtype=np.int64)
    )
    if np.unique(shots_per_parent).size != 1:
        raise RuntimeError(
            f"Predictability map requires a balanced shooting set, counts={shots_per_parent}."
        )
    metrics = {
        "scientific_contract": {
            "encoder": str(cache.manifest["spec"]["checkpoint"]),
            "split": "source-run-held-out optimization/model-selection/final-validation",
            "noise_ceiling": f"repeated random {int(shots_per_parent[0]) // 2}+{int(shots_per_parent[0]) // 2} sibling-shot partitions on final-validation parents",
            "individual_future_reference": f"finite-sample leave-one-shot-out mean of the other {int(shots_per_parent[0]) - 1} sibling futures; not a theoretical upper bound",
            "horizon_input_selection": "input variant chosen only by model-selection R2, then evaluated at every horizon on untouched validation sources",
            "visualization": "local-only predictive map is applied to cached future GeoFrame embeddings; this future-state use is diagnostic and out of the probe training distribution",
        },
        "data": {
            "embedding_cache": str(cache.path),
            "parents": int(parent_count),
            "branches": int(cache.future_z.shape[0]),
            "shots_per_parent": int(shots_per_parent[0]),
            "center_atoms_per_parent": int(center_count),
            "horizons_ps": horizons,
            "optimization_parents": int(distribution_targets.parent_splits["optimization"].size),
            "selection_parents": int(distribution_targets.parent_splits["selection"].size),
            "validation_parents": int(distribution_targets.parent_splits["validation"].size),
            "validation_sources": int(np.unique(metadata["source"][joint_targets.split_rows["validation"]]).size),
        },
        "noise_ceiling": noise_metrics,
        "input_ladder": ladder_metrics,
        "selected_horizon_input": selected_input,
        "predictability_by_horizon": horizon_metrics,
        "model_scores_for_plot": model_scores,
        "ceilings_for_plot": ceilings,
        "static_predictive_disagreement": disagreement,
        "shooting_fans": fan_records,
        "accepted_atlas_reference": {
            "position_joint_law_validation_r2": accepted_position_metrics["atlas_training"]["seed_metrics"][str(accepted_position_metrics["atlas_training"]["selected_seed"])]["validation"]["r2"],
            "position_retrieval_gain_percent": accepted_position_metrics["evaluation"]["retrieval"]["predicted_joint_path_mean_embedding"]["gain_over_static_caliper_baseline"]["gain_percent"],
            "history_joint_law_validation_r2": accepted_history_metrics["training"]["seed_metrics"][str(accepted_history_metrics["training"]["selected_seed"])]["validation"]["r2"],
            "history_retrieval_gain_percent": accepted_history_metrics["evaluation"]["retrieval"]["predicted_joint_path_mean_embedding_history"]["gain_over_static_caliper_baseline"]["gain_percent"],
        },
    }
    np.savez_compressed(
        output_dir / "analysis_arrays.npz",
        parent_index=np.repeat(np.arange(parent_count), center_count),
        atom_id=np.tile(np.asarray(cache.atom_ids), parent_count),
        temperature_K=metadata["temperature"],
        crystalline_fraction=metadata["crystallinity"],
        local_predictive_latent=local_fitted.latent,
        selected_predictive_latent=best_fitted.latent,
        selected_joint_law_prediction=predicted_joint_embedding,
        empirical_joint_law=joint_targets.empirical_mean_embedding,
        predictive_latent_present_2d=dynamics["present_2d"],
        predictive_latent_future_2d=dynamics["future_2d"],
        branch_parent_index=dynamics["branch_parent_index"],
        horizons_ps=np.asarray(horizons),
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(json_ready(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_histories.json").write_text(
        json.dumps(
            json_ready(
                {
                    name: fitted.histories for name, fitted in fitted_ladder.items()
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv_tables(output_dir, horizons, noise_metrics, horizon_metrics, ladder_metrics)
    _write_summary(output_dir, metrics, resolved_config_path)
    print(f"[predictability-map] complete output={output_dir}", flush=True)
    return metrics


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure the target-, horizon-, and input-dependent predictability of shooting dynamics."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
