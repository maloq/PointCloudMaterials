from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data_utils.shooting_dataset import (
    load_shooting_campaign_snapshot,
    load_shooting_campaigns_snapshot,
)
from src.temporal_vamp.embeddings import load_frozen_encoder
from src.temporal_vamp.shooting_embeddings import (
    ShootingEmbeddingCache,
    extract_shooting_embedding_cache,
)
from src.temporal_vamp.shooting_predictor import (
    evaluate_shooting_predictor,
    fit_shooting_predictive_bottleneck,
    plot_shooting_neighbor_metrics,
    plot_shooting_training,
    save_shooting_predictor,
    write_shooting_json,
)


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Shooting predictor configuration requires {path!r}.")
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
            f"embedding.device={raw!r} requests CUDA, but torch.cuda.is_available() is false."
        )
    return str(raw)


def run_shooting_predictor(
    config_path: str | Path,
    *,
    stage: str = "all",
) -> dict[str, Any]:
    config_file = _resolve_path(config_path)
    cfg: DictConfig = OmegaConf.load(config_file)
    OmegaConf.resolve(cfg)
    resolved_stage = str(stage).strip().lower()
    if resolved_stage not in {"all", "extract", "train"}:
        raise ValueError(
            f"Shooting predictor stage must be all, extract, or train; got {stage!r}."
        )
    output_dir = _resolve_path(_required(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    temperatures_K = [
        float(value) for value in _required(cfg, "data.temperatures_K")
    ]
    minimum_branches = int(
        _required(cfg, "data.minimum_complete_branches_per_parent")
    )
    configured_roots = OmegaConf.select(cfg, "data.campaign_roots", default=None)
    configured_root = OmegaConf.select(cfg, "data.campaign_root", default=None)
    if configured_roots is not None:
        if configured_root is not None:
            raise ValueError(
                "Configure exactly one of data.campaign_root or data.campaign_roots, "
                "not both."
            )
        snapshot = load_shooting_campaigns_snapshot(
            [_resolve_path(value) for value in configured_roots],
            temperatures_K=temperatures_K,
            minimum_complete_branches_per_parent=minimum_branches,
        )
    else:
        snapshot = load_shooting_campaign_snapshot(
            _resolve_path(_required(cfg, "data.campaign_root")),
            temperatures_K=temperatures_K,
            minimum_complete_branches_per_parent=minimum_branches,
        )
    expected_parent_count = OmegaConf.select(
        cfg, "data.expected_parent_count", default=None
    )
    expected_branch_count = OmegaConf.select(
        cfg, "data.expected_branch_count", default=None
    )
    if expected_parent_count is not None and len(snapshot.parents) != int(
        expected_parent_count
    ):
        raise RuntimeError(
            "Shooting snapshot parent count does not match the configured scientific "
            f"contract: expected={int(expected_parent_count)}, "
            f"observed={len(snapshot.parents)}."
        )
    if expected_branch_count is not None and len(snapshot.branches) != int(
        expected_branch_count
    ):
        raise RuntimeError(
            "Shooting snapshot branch count does not match the configured scientific "
            f"contract: expected={int(expected_branch_count)}, "
            f"observed={len(snapshot.branches)}."
        )
    write_shooting_json(output_dir / "dataset_snapshot.json", snapshot.to_dict())
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    device = _resolve_device(str(_required(cfg, "embedding.device")))
    embedding_path = output_dir / "embeddings"
    if resolved_stage in {"all", "extract"}:
        encoder = load_frozen_encoder(
            _resolve_path(_required(cfg, "encoder.checkpoint")),
            device=device,
            repeats=int(_required(cfg, "embedding.repeats")),
            seed=int(_required(cfg, "embedding.seed")),
            representation_source=str(
                OmegaConf.select(cfg, "encoder.representation_source", default="checkpoint")
            ),
        )
        print(
            f"[shooting] encoder=GeoFrameTransformer checkpoint={encoder.checkpoint_path} "
            f"representation={encoder.representation_source} device={device}"
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
                _required(cfg, "data.spatial_context.center_count")
            ),
            spatial_context_aggregation=str(
                _required(cfg, "data.spatial_context.aggregation")
            ),
            point_cloud_batch_size=int(
                _required(cfg, "embedding.point_cloud_batch_size")
            ),
            environment_batch_size=int(
                _required(cfg, "embedding.environment_batch_size")
            ),
            environment_num_workers=int(
                _required(cfg, "embedding.environment_num_workers")
            ),
            force_recompute=bool(_required(cfg, "cache.force_recompute")),
        )
    else:
        cache = ShootingEmbeddingCache.load(embedding_path)
    extraction_summary = {
        "parents": int(cache.parent_z.shape[0]),
        "branches": int(cache.future_z.shape[0]),
        "center_atoms": int(cache.parent_z.shape[1]),
        "input_embedding_dim": int(cache.parent_z.shape[2]),
        "future_embedding_dim": int(cache.future_z.shape[3]),
        "horizons_ps": np.asarray(cache.horizons_ps).tolist(),
    }
    if resolved_stage == "extract":
        write_shooting_json(output_dir / "extraction_summary.json", extraction_summary)
        return extraction_summary

    fitted = fit_shooting_predictive_bottleneck(
        cache,
        device=device,
        hidden_dim=int(_required(cfg, "model.hidden_dim")),
        bottleneck_dim=int(_required(cfg, "model.bottleneck_dim")),
        target_pca_dim=int(_required(cfg, "model.target_pca_dim")),
        input_pca_dim=int(_required(cfg, "evaluation.input_pca_dim")),
        dropout=float(_required(cfg, "model.dropout")),
        learning_rate=float(_required(cfg, "training.learning_rate")),
        weight_decay=float(_required(cfg, "training.weight_decay")),
        geometry_weight=float(_required(cfg, "training.geometry_weight")),
        batch_size=int(_required(cfg, "training.batch_size")),
        maximum_epochs=int(_required(cfg, "training.maximum_epochs")),
        patience=int(_required(cfg, "training.patience")),
        seeds=[int(value) for value in _required(cfg, "training.seeds")],
        selection_source_velocity_seeds=[
            int(value)
            for value in _required(cfg, "split.selection_source_velocity_seeds")
        ],
    )
    metrics, arrays = evaluate_shooting_predictor(
        fitted,
        cache,
        device=device,
        ridge_alphas=[float(value) for value in _required(cfg, "evaluation.ridge_alphas")],
        neighbors=int(_required(cfg, "evaluation.future_neighbors.k")),
        seed=int(_required(cfg, "evaluation.seed")),
    )
    metrics["extraction"] = extraction_summary
    metrics["scientific_contract"] = {
        "conditioning": "positions and temperature; independent momenta and Langevin noise marginalized by branch averaging",
        "future_target": "mean frozen local GeoFrameTransformer embedding across complete sibling branches",
        "split_unit": "source MD run; siblings and both parent times remain together",
        "metadata_used_as_model_input": False,
    }
    save_shooting_predictor(fitted, output_dir / "shooting_predictor.pt")
    np.savez(output_dir / "coordinates_and_predictions.npz", **arrays)
    write_shooting_json(output_dir / "metrics.json", metrics)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_shooting_training(fitted, plots_dir / "training_curves.png")
    plot_shooting_neighbor_metrics(
        metrics["future_neighbor_consistency"],
        plots_dir / "ensemble_future_neighbor_consistency.png",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a predictive bottleneck on position-conditioned shooting futures."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("all", "extract", "train"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_shooting_predictor(args.config, stage=args.stage)


if __name__ == "__main__":
    main()


__all__ = ["run_shooting_predictor"]
