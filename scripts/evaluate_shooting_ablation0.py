#!/usr/bin/env python
"""Re-score a saved shooting representation against future-change targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.temporal_vamp.shooting_ablation import (
    evaluate_saved_shooting_dynamic_targets,
    load_saved_shooting_arrays,
)
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache
from src.temporal_vamp.shooting_predictor import (
    plot_shooting_neighbor_metrics,
    write_shooting_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved shooting coordinates against mean-delta and "
            "current-residualized future targets."
        )
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache = ShootingEmbeddingCache.load(experiment_dir / "embeddings")
    saved_arrays = load_saved_shooting_arrays(
        experiment_dir / "coordinates_and_predictions.npz"
    )
    with (experiment_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        previous_metrics = json.load(handle)
    metrics, targets = evaluate_saved_shooting_dynamic_targets(
        cache,
        saved_arrays,
        split_parent_indices=previous_metrics["split_parent_indices"],
        ridge_alphas=args.ridge_alphas,
        neighbors=int(args.neighbors),
        seed=int(args.seed),
    )
    metrics["source_experiment"] = str(experiment_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_shooting_json(output_dir / "metrics.json", metrics)
    np.savez(output_dir / "future_change_targets.npz", **targets)
    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    for target_name, target_metrics in metrics["future_neighbor_consistency"].items():
        plot_values = {
            name: value
            for name, value in target_metrics.items()
            if name != "bottleneck_gain_over_context_pca_percent"
        }
        plot_shooting_neighbor_metrics(
            plot_values,
            plots / f"future_neighbor_{target_name}.png",
        )
    print(json.dumps(metrics["future_neighbor_consistency"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
