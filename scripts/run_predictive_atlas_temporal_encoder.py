#!/usr/bin/env python3
"""Evaluate an ordinary-MD temporally trained encoder in the fixed shooting atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")

from src.temporal_vamp.evaluation import CovariancePCA
from src.temporal_vamp.predictive_atlas import (
    FittedPredictiveAtlas,
    _bootstrap_gain,
    build_atlas_baseline_spaces,
    evaluate_predictive_atlas,
    fit_predictive_atlas,
    parent_temperature_conditioning,
    prepare_joint_path_target_data_from_kernel,
    select_atlas_witnesses,
)
from src.temporal_vamp.shooting_context import ShootingContextTokenCache
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_history import ShootingHistoryEmbeddingCache
from src.temporal_vamp.shooting_multiscale import build_multiscale_feature_variants
from src.temporal_vamp.shooting_spatial import build_spatial_token_data


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Temporal-encoder atlas configuration requires {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _assert_aligned(
    student: ShootingEmbeddingCache,
    teacher: ShootingEmbeddingCache,
) -> None:
    array_pairs = {
        "atom_ids": (student.atom_ids, teacher.atom_ids),
        "horizons_ps": (student.horizons_ps, teacher.horizons_ps),
        "branch_parent_index": (
            student.branch_parent_index,
            teacher.branch_parent_index,
        ),
    }
    for name, (left, right) in array_pairs.items():
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            raise RuntimeError(f"Student and teacher shooting caches disagree on {name}.")
    student_parents = student.manifest["snapshot"]["parents"]
    teacher_parents = teacher.manifest["snapshot"]["parents"]
    keys = (
        "parent_id",
        "source_run_id",
        "source_split",
        "source_velocity_seed",
        "source_frame_index",
        "temperature_K",
        "phase",
    )
    if len(student_parents) != len(teacher_parents):
        raise RuntimeError(
            f"Student/teacher parent counts differ: {len(student_parents)} and "
            f"{len(teacher_parents)}."
        )
    for index, (student_parent, teacher_parent) in enumerate(
        zip(student_parents, teacher_parents)
    ):
        changed = [
            key for key in keys if student_parent.get(key) != teacher_parent.get(key)
        ]
        if changed:
            raise RuntimeError(
                f"Student/teacher parent provenance differs at index={index}: {changed}."
            )


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


def _plot_training(fitted: FittedPredictiveAtlas, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed, history in sorted(fitted.histories.items()):
        axes[0].plot(history["optimization"], label=f"seed {seed}")
        axes[1].plot(history["selection"], label=f"seed {seed}")
    for axis, title in zip(axes, ("Optimization", "Source-run selection")):
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel("conditional mean-embedding MSE")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_retrieval(retrieval: dict[str, Any], path: Path) -> None:
    names = (
        "teacher_local_pca_32d",
        "previous_teacher_encoder_atlas_prediction",
        "temporal_encoder_atlas_prediction",
        "empirical_joint_path_mean_embedding_oracle",
    )
    labels = ("static PCA", "previous atlas", "temporal encoder", "oracle")
    values = [
        retrieval[name]["mean_heldout_empirical_mmd_distance"] for name in names
    ]
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.bar(labels, values, color=("#777777", "#4472c4", "#c44e52", "#55a868"))
    axis.set_ylabel("held-out empirical future-path distance")
    axis.set_title("Fixed-teacher temporal-encoder comparison (lower is better)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config_path: str | Path) -> dict[str, Any]:
    cfg: DictConfig = OmegaConf.load(_resolve_path(config_path))
    OmegaConf.resolve(cfg)
    output_dir = _resolve_path(_required(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    device = str(_required(cfg, "device"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requests CUDA, but CUDA is unavailable.")

    student_root = _resolve_path(_required(cfg, "input.student_cache_run"))
    teacher_root = _resolve_path(_required(cfg, "input.teacher_cache_run"))
    history_root = _resolve_path(_required(cfg, "input.student_history_run"))
    student = ShootingEmbeddingCache.load(student_root / "embeddings")
    student_context = ShootingContextTokenCache.load(student_root / "context_tokens")
    teacher = ShootingEmbeddingCache.load(teacher_root / "embeddings")
    teacher_context = ShootingContextTokenCache.load(teacher_root / "context_tokens")
    history = ShootingHistoryEmbeddingCache.load(history_root / "history_embeddings")
    _assert_aligned(student, teacher)
    if not np.array_equal(
        np.asarray(history.context_center_atom_ids),
        np.asarray(
            ShootingHistoryEmbeddingCache.load(
                _resolve_path(_required(cfg, "input.teacher_history_run"))
                / "history_embeddings"
            ).context_center_atom_ids,
        ),
    ):
        raise RuntimeError("Student history satellite atom identities changed.")

    tokens = build_spatial_token_data(student, student_context)
    history_values = np.asarray(history.history_z, dtype=np.float32).reshape(
        tokens.embeddings.shape[0],
        history.history_z.shape[2],
        history.history_z.shape[3],
        history.history_z.shape[4],
    )
    target = prepare_joint_path_target_data_from_kernel(
        teacher,
        kernel_path=_resolve_path(_required(cfg, "input.fixed_path_kernel")),
        selection_source_velocity_seeds=[
            int(value) for value in _required(cfg, "split.selection_source_velocity_seeds")
        ],
        rff_device=device,
        rff_batch_size=int(_required(cfg, "target.rff_batch_size")),
    )
    initial_payload = torch.load(
        _resolve_path(_required(cfg, "initialization.previous_atlas_checkpoint")),
        map_location="cpu",
        weights_only=False,
    )
    fitted = fit_predictive_atlas(
        tokens,
        target,
        parent_temperature_conditioning(student),
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
        history_embeddings=history_values,
        initial_model_state=initial_payload["state_dict"],
    )
    torch.save(
        _checkpoint_payload(fitted, history_spec=history.manifest["spec"]),
        output_dir / "model.pt",
    )

    student_features = build_multiscale_feature_variants(
        student,
        student_context,
        radial_scales_angstrom=[
            float(value) for value in _required(cfg, "context.radial_scales_angstrom")
        ],
    )
    spaces, vamp = build_atlas_baseline_spaces(
        student,
        target,
        student_features,
        fitted,
        marginal_prediction=None,
        static_pca_dim=int(_required(cfg, "evaluation.static_pca_dim")),
        vamp_horizon_ps=float(_required(cfg, "evaluation.vamp.horizon_ps")),
        vamp_dimension=int(_required(cfg, "evaluation.vamp.dimension")),
        vamp_regularization=float(_required(cfg, "evaluation.vamp.regularization")),
        vamp_eigenvalue_cutoff=float(
            _required(cfg, "evaluation.vamp.eigenvalue_cutoff")
        ),
    )
    spaces["student_local_pca_32d"] = spaces.pop("local_pca_32d")
    spaces["student_context_mean_std_pca_32d"] = spaces.pop(
        "context_mean_std_pca_32d"
    )
    spaces["temporal_encoder_atlas_latent_32d"] = spaces.pop("atlas_latent_32d")
    spaces["temporal_encoder_atlas_prediction"] = spaces.pop(
        "predicted_joint_path_mean_embedding"
    )
    teacher_features = build_multiscale_feature_variants(
        teacher,
        teacher_context,
        radial_scales_angstrom=[
            float(value) for value in _required(cfg, "context.radial_scales_angstrom")
        ],
    )
    teacher_local = np.asarray(teacher_features["local"], dtype=np.float64)
    teacher_pca = CovariancePCA.fit(
        teacher_local[target.split_rows["optimization"]],
        dimension=int(_required(cfg, "evaluation.static_pca_dim")),
    )
    spaces["teacher_local_pca_32d"] = teacher_pca.transform(
        teacher_local, dimension=int(_required(cfg, "evaluation.static_pca_dim"))
    )
    previous_archive = _resolve_path(_required(cfg, "input.previous_coordinates"))
    with np.load(previous_archive, allow_pickle=False) as previous:
        previous_prediction = previous["expanded_frozen_prediction"].copy()
    if previous_prediction.shape != spaces["temporal_encoder_atlas_prediction"].shape:
        raise RuntimeError(
            f"Previous/new atlas prediction shapes differ: {previous_prediction.shape} "
            f"and {spaces['temporal_encoder_atlas_prediction'].shape}."
        )
    spaces["previous_teacher_encoder_atlas_prediction"] = previous_prediction

    evaluation, arrays, _ = evaluate_predictive_atlas(
        teacher,
        target,
        spaces,
        static_space_name="teacher_local_pca_32d",
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
    validation_rows = np.asarray(arrays["validation_rows"], dtype=np.int64)
    source_ids = np.repeat(
        np.asarray(
            [str(parent["source_run_id"]) for parent in teacher.manifest["snapshot"]["parents"]]
        ),
        int(teacher.parent_z.shape[1]),
    )[validation_rows]
    direct_gain = _bootstrap_gain(
        np.asarray(arrays["query_future_distance__temporal_encoder_atlas_prediction"]),
        np.asarray(
            arrays["query_future_distance__previous_teacher_encoder_atlas_prediction"]
        ),
        source_ids.tolist(),
        samples=int(_required(cfg, "evaluation.bootstrap_samples")),
        seed=int(_required(cfg, "evaluation.seed")) + 311,
    )
    witnesses = select_atlas_witnesses(
        teacher,
        target,
        spaces,
        arrays,
        static_space_name="teacher_local_pca_32d",
        atlas_space_name="temporal_encoder_atlas_prediction",
        count=int(_required(cfg, "evaluation.witness_count")),
    )
    metrics = {
        "scientific_contract": {
            "present_and_history_encoder": str(student.manifest["spec"]["checkpoint"]),
            "future_teacher_cache": str(teacher.path),
            "fixed_path_kernel": str(_resolve_path(_required(cfg, "input.fixed_path_kernel"))),
            "fixed_static_caliper": "teacher GeoFrame PCA from the previous experiment",
            "fixed_previous_prediction": str(previous_archive),
            "history_lags_ps": [12, 9, 6, 3],
        },
        "selected_seed": fitted.seed,
        "training": {str(seed): value for seed, value in fitted.seed_metrics.items()},
        "evaluation": evaluation,
        "direct_gain_over_previous_atlas": direct_gain,
    }
    vamp.save(output_dir / "vamp_baseline.npz")
    np.savez(
        output_dir / "coordinates_and_predictions.npz",
        atom_ids=np.tile(np.asarray(teacher.atom_ids), int(teacher.parent_z.shape[0])),
        parent_index=np.repeat(
            np.arange(int(teacher.parent_z.shape[0])), int(teacher.parent_z.shape[1])
        ),
        **{f"space__{name}": value for name, value in spaces.items()},
        **arrays,
    )
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "witnesses.json", {"witnesses": witnesses})
    _plot_training(fitted, plot_dir / "training.png")
    _plot_retrieval(evaluation["retrieval"], plot_dir / "retrieval.png")
    print(
        f"[temporal-encoder-atlas] complete output={output_dir} seed={fitted.seed}",
        flush=True,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
