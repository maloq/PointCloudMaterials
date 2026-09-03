#!/usr/bin/env python3
"""Fine-tune the GeoFrameV2 encoder tail on leakage-safe ordinary MD futures."""

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

from src.data_utils.temporal_binary_context_dataset import TemporalBinaryContextDataset
from src.temporal_vamp.embeddings import load_frozen_encoder
from src.temporal_vamp.ordinary_pretraining import (
    OrdinaryContextEmbeddingCache,
    prepare_ordinary_pretraining_targets,
)
from src.temporal_vamp.simulation_catalog import discover_simulation_catalog
from src.temporal_vamp.temporal_encoder_pretraining import (
    OrdinaryGeoFrameActivationCache,
    extract_ordinary_geoframe_activation_cache,
    fit_temporal_geoframe_encoder,
    write_temporal_encoder_checkpoint,
)


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Temporal-encoder configuration requires {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _plot(histories: dict[int, dict[str, list[float]]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed, history in sorted(histories.items()):
        axes[0].plot(history["baseline_selection"], label=f"seed {seed}")
        axes[1].plot(history["joint_selection"], label=f"seed {seed}")
    for axis, title in zip(axes, ("Frozen-encoder head", "Encoder-tail fine-tuning")):
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel("source-run selection MSE")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config_path: str | Path, *, stage: str) -> dict[str, Any]:
    if stage not in {"all", "extract", "train"}:
        raise ValueError(f"stage must be all, extract, or train; got {stage!r}.")
    cfg: DictConfig = OmegaConf.load(_resolve_path(config_path))
    OmegaConf.resolve(cfg)
    output_dir = _resolve_path(_required(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    device = str(_required(cfg, "device"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requests CUDA, but CUDA is unavailable.")

    ordinary_cache = OrdinaryContextEmbeddingCache.load(
        _resolve_path(_required(cfg, "ordinary.embedding_cache"))
    )
    entries = discover_simulation_catalog(
        _resolve_path(_required(cfg, "ordinary.catalog.root")),
        campaign_globs=[
            str(value) for value in _required(cfg, "ordinary.catalog.campaign_globs")
        ],
        cache_root=_resolve_path(_required(cfg, "ordinary.catalog.cache_root")),
        required_atom_count=int(_required(cfg, "ordinary.catalog.required_atom_count")),
        required_potential_parameter_sha256=str(
            _required(cfg, "ordinary.catalog.required_potential_parameter_sha256")
        ),
        required_crystal_seed=OmegaConf.select(
            cfg, "ordinary.catalog.required_crystal_seed", default=None
        ),
        require_periodic=bool(_required(cfg, "ordinary.catalog.require_periodic")),
    )
    included_run_ids = {
        str(value["run_id"]) for value in ordinary_cache.manifest["spec"]["runs"]
    }
    selected_entries = tuple(entry for entry in entries if entry.run_id in included_run_ids)
    if {entry.run_id for entry in selected_entries} != included_run_ids:
        missing = sorted(included_run_ids.difference(entry.run_id for entry in selected_entries))
        raise RuntimeError(f"Ordinary cache source runs are absent from the catalog: {missing}")
    selected_entries = tuple(
        sorted(
            selected_entries,
            key=lambda entry: next(
                index
                for index, item in enumerate(ordinary_cache.manifest["spec"]["runs"])
                if str(item["run_id"]) == entry.run_id
            ),
        )
    )
    spec = ordinary_cache.manifest["spec"]
    dataset = TemporalBinaryContextDataset(
        selected_entries,
        center_atom_ids=np.asarray(spec["center_atom_ids"], dtype=np.int64),
        horizons_ps=[float(value) for value in spec["horizons_ps"]],
        anchor_stride_frames=int(spec["anchor_stride_frames"]),
        num_points=int(spec["num_points"]),
        radius=float(spec["radius"]),
        context_center_count=int(spec["context_center_count"]),
        steinhardt_shell_min_neighbors=int(
            _required(cfg, "ordinary.steinhardt_shell_min_neighbors")
        ),
        steinhardt_shell_max_neighbors=int(
            _required(cfg, "ordinary.steinhardt_shell_max_neighbors")
        ),
        trajectory_cache_size=int(_required(cfg, "ordinary.trajectory_cache_size")),
    )
    expected_rows = len(dataset) * int(dataset.center_atom_ids.size)
    if expected_rows != int(ordinary_cache.manifest["row_count"]):
        raise RuntimeError(
            f"Reconstructed ordinary dataset has {expected_rows} rows, while its "
            f"immutable cache has {ordinary_cache.manifest['row_count']}."
        )

    checkpoint = _resolve_path(_required(cfg, "encoder.checkpoint"))
    frozen = load_frozen_encoder(
        checkpoint,
        device=device,
        repeats=int(_required(cfg, "encoder.repeats")),
        seed=int(_required(cfg, "encoder.seed")),
        representation_source=str(_required(cfg, "encoder.representation_source")),
    )
    activation_path = output_dir / "ordinary_geoframe_activations"
    if stage in {"all", "extract"}:
        activations = extract_ordinary_geoframe_activation_cache(
            dataset,
            ordinary_cache,
            encoder=frozen,
            cache_path=activation_path,
            trainable_tail_blocks=int(_required(cfg, "training.trainable_tail_blocks")),
            point_cloud_batch_size=int(_required(cfg, "encoder.point_cloud_batch_size")),
            environment_batch_size=int(_required(cfg, "encoder.environment_batch_size")),
            environment_num_workers=int(_required(cfg, "encoder.environment_num_workers")),
            force_recompute=bool(_required(cfg, "cache.force_recompute")),
        )
    else:
        activations = OrdinaryGeoFrameActivationCache.load(activation_path)
    extraction = {
        "runs": len(selected_entries),
        "anchors": len(dataset),
        "centers_per_anchor": int(dataset.center_atom_ids.size),
        "rows": expected_rows,
        "horizons_ps": dataset.horizons_ps.tolist(),
        "activation_shapes": activations.manifest["array_shapes"],
    }
    _write_json(output_dir / "extraction_summary.json", extraction)
    if stage == "extract":
        return extraction

    targets = prepare_ordinary_pretraining_targets(
        ordinary_cache,
        optimization_velocity_seeds=[
            int(value) for value in _required(cfg, "split.optimization_velocity_seeds")
        ],
        selection_velocity_seeds=[
            int(value) for value in _required(cfg, "split.selection_velocity_seeds")
        ],
        pca_dim_per_horizon=int(_required(cfg, "target.pca_dim_per_horizon")),
    )
    fitted = fit_temporal_geoframe_encoder(
        activations,
        ordinary_cache,
        targets,
        frozen,
        device=device,
        trainable_tail_blocks=int(_required(cfg, "training.trainable_tail_blocks")),
        hidden_dim=int(_required(cfg, "model.hidden_dim")),
        batch_size=int(_required(cfg, "training.batch_size")),
        head_learning_rate=float(_required(cfg, "training.head_learning_rate")),
        head_maximum_epochs=int(_required(cfg, "training.head_maximum_epochs")),
        head_patience=int(_required(cfg, "training.head_patience")),
        encoder_learning_rate=float(_required(cfg, "training.encoder_learning_rate")),
        joint_head_learning_rate=float(
            _required(cfg, "training.joint_head_learning_rate")
        ),
        joint_maximum_epochs=int(_required(cfg, "training.joint_maximum_epochs")),
        joint_patience=int(_required(cfg, "training.joint_patience")),
        weight_decay=float(_required(cfg, "training.weight_decay")),
        distillation_weight=float(_required(cfg, "training.distillation_weight")),
        gradient_clip_norm=float(_required(cfg, "training.gradient_clip_norm")),
        seeds=[int(value) for value in _required(cfg, "training.seeds")],
    )
    metrics = {
        "scientific_contract": {
            "future_teacher": str(checkpoint),
            "teacher_frozen": True,
            "future_target": "PCA modes of teacher embedding changes at 6/12/24 ps",
            "split_unit": "independent ordinary MD source run via velocity seed",
            "epoch_minus_one_candidate": True,
            "downstream_temporal_context": "four-frame history is retained by the atlas",
        },
        "extraction": extraction,
        "selected_seed": fitted.selected_seed,
        "trainable_parameter_count": fitted.trainable_parameter_count,
        "trainable_parameter_names": list(fitted.trainable_parameter_names),
        "initial_embedding_max_abs_error": fitted.initial_embedding_max_abs_error,
        "seed_metrics": {str(seed): value for seed, value in fitted.metrics.items()},
    }
    selected = fitted.metrics[fitted.selected_seed]
    metrics["selected_selection_gain_percent"] = 100.0 * (
        1.0
        - float(selected["selection"]["mse"])
        / float(selected["baseline_selection_mse"])
    )
    checkpoint_path = output_dir / "temporal_geoframe_v2.ckpt"
    write_temporal_encoder_checkpoint(
        checkpoint,
        checkpoint_path,
        fitted,
        metadata={
            "ordinary_embedding_cache": str(ordinary_cache.path),
            "horizons_ps": dataset.horizons_ps.tolist(),
            "metrics_path": str(output_dir / "metrics.json"),
        },
    )
    _write_json(output_dir / "metrics.json", metrics)
    torch.save(
        {
            "encoder_state": fitted.encoder_state,
            "selected_seed": fitted.selected_seed,
            "histories": fitted.histories,
            "metrics": fitted.metrics,
        },
        output_dir / "temporal_encoder_training.pt",
    )
    _plot(fitted.histories, output_dir / "training.png")
    reloaded = load_frozen_encoder(
        checkpoint_path,
        device=device,
        repeats=int(_required(cfg, "encoder.repeats")),
        seed=int(_required(cfg, "encoder.seed")),
        representation_source=str(_required(cfg, "encoder.representation_source")),
    )
    if reloaded.output_dim != frozen.output_dim:
        raise RuntimeError(
            f"Exported temporal encoder changed output dimension: "
            f"teacher={frozen.output_dim}, student={reloaded.output_dim}."
        )
    print(
        f"[temporal-encoder] complete output={output_dir} "
        f"seed={fitted.selected_seed} checkpoint={checkpoint_path}",
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
