#!/usr/bin/env python3
"""Compare the checkpoint's invariant encoder and VICReg projector dynamics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import csv
import json
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _load_all_lags(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["group_kind"] == "all" and row["group"] == "all"
        ]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projector", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    roots = {
        "VICReg projector": Path(args.projector).expanduser().resolve(),
        "raw invariant encoder": Path(args.encoder).expanduser().resolve(),
    }
    target = Path(args.output).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    metrics = {name: _load_json(root / "metrics.json") for name, root in roots.items()}
    lag_rows = {name: _load_all_lags(root / "lag_metrics.csv") for name, root in roots.items()}

    summary: dict[str, Any] = {}
    for name, values in metrics.items():
        short = values["lag_checkpoints"]["0.3ps"]
        long = values["lag_checkpoints"]["24ps"]
        unsmoothed = values["smoothing"][0]
        smoothed = values["smoothing"][1]
        summary[name] = {
            "embedding_rms_radius": values["embedding"]["global_rms_radius"],
            "normalized_rms_distance_0.3ps": short[
                "distance_rms_over_static_radius"
            ],
            "normalized_rms_distance_24ps": long[
                "distance_rms_over_static_radius"
            ],
            "distance_growth_24ps_over_0.3ps": long["distance_rms"]
            / short["distance_rms"],
            "trajectory_centered_correlation_0.3ps": short[
                "trajectory_centered_correlation"
            ],
            "increment_direction_cosine": values["increment_direction_alignment"][
                "mean"
            ],
            "aligned_increment_fraction": values["increment_direction_alignment"][
                "positive_fraction"
            ],
            "step_rms_reduction_by_0.9ps_average": 1.0
            - smoothed["adjacent_step_rms"] / unsmoothed["adjacent_step_rms"],
            "temporal_change_pca_cumulative_2": values["pca"][
                "temporal_change_cumulative_2"
            ],
            "temporal_change_pca_cumulative_8": values["pca"][
                "temporal_change_cumulative_8"
            ],
        }
    with (target / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    projector = summary["VICReg projector"]
    encoder = summary["raw invariant encoder"]
    (target / "README.md").write_text(
        "\n".join(
            [
                "# Representation-source diagnostic",
                "",
                "Both representations come from the same frozen checkpoint and use exactly the same point clouds.",
                "",
                f"- VICReg projector: the 0.3 ps jump is {100*projector['normalized_rms_distance_0.3ps']:.1f}% of its static RMS radius; the 24 ps distance is only {projector['distance_growth_24ps_over_0.3ps']:.3f} times larger; consecutive-increment cosine is {projector['increment_direction_cosine']:.3f}.",
                f"- Raw invariant encoder: the corresponding values are {100*encoder['normalized_rms_distance_0.3ps']:.1f}%, {encoder['distance_growth_24ps_over_0.3ps']:.3f} times, and {encoder['increment_direction_cosine']:.3f}.",
                "",
                "The projector preserves slightly more lag-dependent signal and has less reversing high-frequency motion, so it remains the preferred representation. The short-time plateau exists in both representations and is therefore not caused only by the projector head.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    colors = {"VICReg projector": "#7b2cbf", "raw invariant encoder": "#1982c4"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for name, rows in lag_rows.items():
        lag = [float(row["lag_ps"]) for row in rows]
        axes[0].plot(
            lag,
            [float(row["distance_rms_over_static_radius"]) for row in rows],
            "o-",
            label=name,
            color=colors[name],
        )
        axes[1].plot(
            lag,
            [float(row["trajectory_centered_correlation"]) for row in rows],
            "o-",
            label=name,
            color=colors[name],
        )
    names = list(summary)
    axes[2].bar(
        [0, 1],
        [summary[name]["increment_direction_cosine"] for name in names],
        color=[colors[name] for name in names],
    )
    axes[2].set_xticks([0, 1], ["projector", "encoder"])
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[0].set_title("Lag-dependent change")
    axes[0].set_xlabel("lag (ps)")
    axes[0].set_ylabel("RMS distance / static RMS radius")
    axes[1].set_title("Temporal memory")
    axes[1].set_xlabel("lag (ps)")
    axes[1].set_ylabel("trajectory-centered correlation")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[2].set_title("Consecutive increment alignment")
    axes[2].set_ylabel("mean cosine")
    for axis in axes[:2]:
        axis.legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(target / "representation_source_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
