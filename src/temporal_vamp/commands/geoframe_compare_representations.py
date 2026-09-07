#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import csv
import json
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.temporal_vamp.geoframe_stability import row_cosine, same_atom_retrieval


def _load_result(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics_path = path / "metrics.json"
    arrays_path = path / "stability_arrays.npz"
    if not metrics_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(
            f"GeoFrame stability result requires metrics.json and stability_arrays.npz: {path}"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    archive = np.load(arrays_path)
    arrays = {name: archive[name] for name in archive.files}
    return metrics, arrays


def _representation_rows(
    metrics: dict[str, Any], arrays: dict[str, np.ndarray]
) -> list[dict[str, float | str]]:
    embeddings = arrays["embeddings"].astype(np.float64)
    normalized = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
    cross_angular = np.sqrt(
        np.maximum(0.0, 2.0 - 2.0 * arrays["cross_state_cosines"].astype(np.float64))
    )
    cross_angular_median = float(np.median(cross_angular))
    rows: list[dict[str, float | str]] = []
    for time_index, time_ps in enumerate(arrays["times_ps"].tolist()):
        cosine = row_cosine(embeddings[:, time_index], embeddings[:, 0])
        angular_distance = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosine))
        normalized_top1, normalized_rank = same_atom_retrieval(
            normalized[:, time_index], normalized[:, 0]
        )
        lag = metrics["lag_metrics"][time_index]
        rows.append(
            {
                "representation": str(metrics["encoder"]["representation_source"]),
                "time_ps": float(time_ps),
                "raw_distance_relative_to_cross_state": float(
                    lag["relative_to_cross_state_median"]
                ),
                "angular_distance_relative_to_cross_state": float(
                    np.median(angular_distance) / cross_angular_median
                ),
                "cosine_mean": float(np.mean(cosine)),
                "cosine_q05": float(np.quantile(cosine, 0.05)),
                "raw_same_atom_top1": float(lag["same_atom_top1"]),
                "normalized_same_atom_top1": normalized_top1,
                "normalized_same_atom_median_rank": normalized_rank,
            }
        )
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare GeoFrame encoder and VICReg-projector finest-step stability."
    )
    parser.add_argument("--projector-result", type=Path, required=True)
    parser.add_argument("--encoder-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite stability comparison: {output}")
    output.mkdir(parents=True)

    projector_metrics, projector_arrays = _load_result(
        args.projector_result.expanduser().resolve()
    )
    encoder_metrics, encoder_arrays = _load_result(args.encoder_result.expanduser().resolve())
    if not np.array_equal(projector_arrays["times_ps"], encoder_arrays["times_ps"]):
        raise RuntimeError("Representation stability runs used different physical times.")
    if not np.array_equal(
        projector_arrays["center_atom_ids"], encoder_arrays["center_atom_ids"]
    ):
        raise RuntimeError("Representation stability runs used different center atom IDs.")

    rows = _representation_rows(projector_metrics, projector_arrays)
    rows.extend(_representation_rows(encoder_metrics, encoder_arrays))
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for name, label in (
        ("vicreg_projector", "VICReg projector"),
        ("encoder", "GeoFrame encoder output"),
    ):
        selected = [row for row in rows if row["representation"] == name]
        time_ps = np.asarray([row["time_ps"] for row in selected])
        axes[0, 0].plot(
            time_ps,
            [row["raw_distance_relative_to_cross_state"] for row in selected],
            marker="o",
            label=label,
        )
        axes[0, 1].plot(
            time_ps,
            [row["angular_distance_relative_to_cross_state"] for row in selected],
            marker="o",
            label=label,
        )
        axes[1, 0].plot(
            time_ps,
            [row["cosine_mean"] for row in selected],
            marker="o",
            label=label,
        )
        axes[1, 1].plot(
            time_ps,
            [row["normalized_same_atom_top1"] for row in selected],
            marker="o",
            label=label,
        )
    axes[0, 0].set_ylabel("raw drift / cross-state median")
    axes[0, 1].set_ylabel("angular drift / cross-state median")
    axes[1, 0].set_ylabel("same-atom cosine")
    axes[1, 1].set_ylabel("same-atom top-1 retrieval")
    axes[1, 1].axhline(1.0 / 32.0, linestyle="--", color="black", label="chance (1/32)")
    for axis in axes.flat:
        axis.set_xlabel("time from shot start (ps)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Frozen GeoFrame stability at the finest stored 30 fs cadence")
    figure.tight_layout()
    figure.savefig(output / "representation_comparison.png", dpi=180)
    plt.close(figure)

    summary: dict[str, Any] = {}
    for name in ("vicreg_projector", "encoder"):
        selected = [row for row in rows if row["representation"] == name]
        summary[name] = {"at_0.03_ps": selected[1], "at_0.30_ps": selected[-1]}
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        """# GeoFrame finest-timestep stability comparison

This compares the frozen checkpoint's direct GeoFrame encoder output with its
VICReg projection head on exactly the same 50,688 local environments. Each
experiment follows 32 fixed atom IDs through 11 frames at 30 fs spacing in all
144 branches of the 36-parent nested shooting campaign.

Raw distance is normalized by the median distance between local states from
different transition parents at the same temperature. Angular distance is the
Euclidean distance after per-embedding L2 normalization and is normalized by
its corresponding cross-state median. Same-atom retrieval searches among the
32 candidate atoms within the same branch; chance is 1/32.

See `comparison.csv`, `summary.json`, and `representation_comparison.png`.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
