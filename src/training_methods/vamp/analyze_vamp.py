from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.analysis.cluster_geometry import _compute_cluster_representative_indices
from src.analysis.cluster_geometry import _sample_indices_stratified
from src.analysis.cluster_blender import _save_md_cluster_snapshot_raytrace_blender
from src.analysis.cluster_rendering import _save_cluster_representatives_figure
from src.training_methods.vamp.common import (
    TrajectoryEmbeddings,
    ensure_dir,
    log_progress,
    load_local_neighborhoods,
    resolve_frame_window,
)
from src.training_methods.vamp.config import load_vamp_config, resolve_path
from src.training_methods.vamp.vamp import ManualVAMP
from src.vis_tools.latent_analysis_vis import save_md_space_clusters_plot
from src.vis_tools.md_cluster_plot import save_interactive_md_plot
from src.vis_tools.real_md_analysis_vis import (
    save_cluster_proportion_plots,
    save_temporal_embedding_cluster_animation,
    save_temporal_embedding_trajectory_animation,
    save_temporal_spatial_cluster_animation,
)
from src.vis_tools.tsne_vis import compute_tsne, compute_umap


class _RepresentativeNeighborhoodDataset:
    def __init__(
        self,
        embeddings: TrajectoryEmbeddings,
        frame_indices: np.ndarray,
        atom_ids: np.ndarray,
        *,
        cache_dir: str | None,
    ) -> None:
        self._embeddings = embeddings
        self._frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        self._atom_ids = np.asarray(atom_ids, dtype=np.int64).reshape(-1)
        self._cache_dir = cache_dir
        self._cache: dict[int, np.ndarray] = {}

    def __getitem__(self, index: int) -> np.ndarray:
        idx = int(index)
        if idx in self._cache:
            return {"points": self._cache[idx]}
        neighborhoods = load_local_neighborhoods(
            self._embeddings,
            frame_indices=np.asarray([self._frame_indices[idx]], dtype=np.int64),
            atom_ids=np.asarray([self._atom_ids[idx]], dtype=np.int64),
            cache_dir=self._cache_dir,
        )
        points = np.asarray(neighborhoods[0], dtype=np.float32)
        self._cache[idx] = points
        return {"points": points}


def _canonicalize_kmeans(labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    centers_arr = np.asarray(centers, dtype=np.float64)
    order = sorted(
        range(centers_arr.shape[0]),
        key=lambda idx: tuple(float(v) for v in centers_arr[idx].tolist()),
    )
    remap = {int(old): int(new) for new, old in enumerate(order)}
    canonical_labels = np.asarray([remap[int(label)] for label in labels_arr], dtype=np.int64)
    canonical_centers = centers_arr[np.asarray(order, dtype=np.int64)]
    return canonical_labels, canonical_centers


def _run_kmeans(features: np.ndarray, *, k: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    estimator = KMeans(
        n_clusters=int(k),
        random_state=int(random_state),
        n_init=20,
    )
    labels = estimator.fit_predict(features).astype(np.int64, copy=False)
    centers = np.asarray(estimator.cluster_centers_, dtype=np.float64)
    return _canonicalize_kmeans(labels, centers)


def _build_cluster_color_map(cluster_count: int) -> dict[int, str]:
    cmap = plt.get_cmap("tab10")
    return {
        int(cluster_id): str(mcolors.to_hex(cmap(int(cluster_id) % 10)))
        for cluster_id in range(int(cluster_count))
    }


def _plot_cluster_populations(
    frame_axis: np.ndarray,
    populations: np.ndarray,
    *,
    ylabel: str,
    out_path: str | Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=180)
    for cluster_id in range(populations.shape[1]):
        ax.plot(
            frame_axis,
            populations[:, cluster_id],
            label=f"cluster {cluster_id}",
        )
    ax.set_xlabel("timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(Path(out_path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def _plot_spatial_snapshots(
    coords: np.ndarray,
    labels: np.ndarray,
    timesteps: np.ndarray,
    *,
    out_path: str | Path,
    snapshot_count: int,
) -> None:
    frame_count = int(coords.shape[0])
    snapshot_count = min(int(snapshot_count), frame_count)
    snapshot_indices = np.linspace(0, frame_count - 1, num=snapshot_count, dtype=np.int64)
    ncols = min(3, snapshot_count)
    nrows = int(np.ceil(snapshot_count / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.5 * ncols, 4.5 * nrows), dpi=180)
    axes_arr = np.atleast_1d(axes).reshape(-1)
    cmap = plt.cm.tab10
    for ax, frame_idx in zip(axes_arr, snapshot_indices.tolist(), strict=False):
        frame_coords = coords[int(frame_idx)]
        frame_labels = labels[int(frame_idx)]
        colors = cmap(frame_labels % 10)
        ax.scatter(frame_coords[:, 0], frame_coords[:, 1], s=8, c=colors, alpha=0.85, linewidths=0)
        ax.set_title(f"frame {frame_idx} / t={int(timesteps[int(frame_idx)])}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes_arr[snapshot_count:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(out_path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def _plot_representative_neighborhoods(
    neighborhoods: np.ndarray,
    *,
    out_path: str | Path,
) -> None:
    cluster_count = int(neighborhoods.shape[0])
    ncols = min(3, cluster_count)
    nrows = int(np.ceil(cluster_count / ncols))
    fig = plt.figure(figsize=(5.8 * ncols, 4.8 * nrows), dpi=180)
    for cluster_id in range(cluster_count):
        ax = fig.add_subplot(nrows, ncols, cluster_id + 1, projection="3d")
        pts = neighborhoods[cluster_id]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, alpha=0.85)
        ax.set_title(f"cluster {cluster_id}")
        ax.set_xlabel("x / rc")
        ax.set_ylabel("y / rc")
        ax.set_zlabel("z / rc")
    fig.tight_layout()
    fig.savefig(Path(out_path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def _plot_2d_embedding(
    coordinates: np.ndarray,
    labels: np.ndarray,
    frame_values: np.ndarray,
    *,
    out_clusters: str | Path,
    out_time: str | Path,
    title_prefix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=180)
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        s=8,
        alpha=0.85,
        cmap="tab10",
        linewidths=0,
    )
    ax.set_title(f"{title_prefix} colored by cluster")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(Path(out_clusters).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=180)
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=frame_values,
        s=8,
        alpha=0.85,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_title(f"{title_prefix} colored by time")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(scatter, ax=ax, label="frame index")
    fig.tight_layout()
    fig.savefig(Path(out_time).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def _plot_vamp_mode_dynamics(
    frame_axis: np.ndarray,
    coordinates: np.ndarray,
    *,
    out_path: str | Path,
    max_modes: int = 4,
) -> None:
    mode_count = min(int(max_modes), int(coordinates.shape[-1]))
    if mode_count <= 0:
        raise ValueError(f"Expected at least one VAMP mode to plot, got shape {coordinates.shape}.")
    fig, axes = plt.subplots(
        nrows=mode_count,
        ncols=1,
        figsize=(8.0, 2.4 * mode_count),
        dpi=180,
        sharex=True,
    )
    axes_arr = np.atleast_1d(axes).reshape(-1)
    for mode_idx, ax in enumerate(axes_arr):
        values = np.asarray(coordinates[:, :, mode_idx], dtype=np.float64)
        mean = np.mean(values, axis=1)
        q10 = np.quantile(values, 0.10, axis=1)
        q90 = np.quantile(values, 0.90, axis=1)
        color = f"C{mode_idx % 10}"
        ax.fill_between(frame_axis, q10, q90, color=color, alpha=0.22, label="10-90% across atoms")
        ax.plot(frame_axis, mean, color=color, linewidth=1.7, label="frame mean")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_ylabel(f"psi{mode_idx + 1}")
        ax.grid(True, alpha=0.25)
        if mode_idx == 0:
            ax.legend(frameon=False, loc="upper right")
    axes_arr[-1].set_xlabel("timestep")
    fig.suptitle("Leading VAMP coordinates vs time", y=0.995)
    fig.tight_layout()
    fig.savefig(Path(out_path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def _plot_vamp_phase_space(
    coordinates: np.ndarray,
    labels: np.ndarray,
    time_values: np.ndarray,
    cluster_centers: np.ndarray,
    *,
    out_clusters: str | Path,
    out_time: str | Path,
    max_modes: int = 3,
) -> None:
    mode_count = min(int(max_modes), int(coordinates.shape[1]))
    if mode_count < 2:
        return
    mode_pairs = [(0, 1)]
    if mode_count >= 3:
        mode_pairs.extend([(0, 2), (1, 2)])
    ncols = min(3, len(mode_pairs))
    nrows = int(np.ceil(len(mode_pairs) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.8 * ncols, 4.6 * nrows), dpi=180)
    axes_arr = np.atleast_1d(axes).reshape(-1)
    for ax, (left_idx, right_idx) in zip(axes_arr, mode_pairs, strict=False):
        ax.scatter(
            coordinates[:, left_idx],
            coordinates[:, right_idx],
            c=labels,
            s=8,
            alpha=0.82,
            cmap="tab10",
            linewidths=0,
        )
        ax.scatter(
            cluster_centers[:, left_idx],
            cluster_centers[:, right_idx],
            c=np.arange(cluster_centers.shape[0]),
            cmap="tab10",
            marker="X",
            s=120,
            edgecolors="black",
            linewidths=0.6,
        )
        ax.set_xlabel(f"psi{left_idx + 1}")
        ax.set_ylabel(f"psi{right_idx + 1}")
        ax.set_title(f"VAMP phase space: psi{left_idx + 1} vs psi{right_idx + 1}")
        ax.grid(True, alpha=0.2)
    for ax in axes_arr[len(mode_pairs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(out_clusters).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.8 * ncols, 4.6 * nrows), dpi=180)
    axes_arr = np.atleast_1d(axes).reshape(-1)
    scatter = None
    for ax, (left_idx, right_idx) in zip(axes_arr, mode_pairs, strict=False):
        scatter = ax.scatter(
            coordinates[:, left_idx],
            coordinates[:, right_idx],
            c=time_values,
            s=8,
            alpha=0.82,
            cmap="viridis",
            linewidths=0,
        )
        ax.set_xlabel(f"psi{left_idx + 1}")
        ax.set_ylabel(f"psi{right_idx + 1}")
        ax.set_title(f"VAMP phase space over time: psi{left_idx + 1} vs psi{right_idx + 1}")
        ax.grid(True, alpha=0.2)
    for ax in axes_arr[len(mode_pairs):]:
        ax.axis("off")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes_arr[: len(mode_pairs)].tolist(), label="timestep", shrink=0.88)
    fig.tight_layout()
    fig.savefig(Path(out_time).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_center_heatmap(
    cluster_centers: np.ndarray,
    *,
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(1.2 * cluster_centers.shape[1] + 2.0, 0.75 * cluster_centers.shape[0] + 2.0), dpi=180)
    image = ax.imshow(cluster_centers, aspect="auto", cmap="coolwarm")
    ax.set_xlabel("VAMP mode")
    ax.set_ylabel("cluster")
    ax.set_xticks(np.arange(cluster_centers.shape[1]))
    ax.set_xticklabels([f"psi{idx + 1}" for idx in range(cluster_centers.shape[1])])
    ax.set_yticks(np.arange(cluster_centers.shape[0]))
    ax.set_yticklabels([f"cluster {idx}" for idx in range(cluster_centers.shape[0])])
    ax.set_title("KMeans centers in VAMP space")
    fig.colorbar(image, ax=ax, label="center value")
    fig.tight_layout()
    fig.savefig(Path(out_path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def _format_frame_label(*, frame_index: int, timestep: int) -> str:
    return f"frame {int(frame_index)} | t={int(timestep)}"


def _sample_frame_offsets(frame_count: int, snapshot_count: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be > 0, got {frame_count}.")
    if snapshot_count <= 0:
        raise ValueError(f"snapshot_count must be > 0, got {snapshot_count}.")
    return np.unique(
        np.linspace(0, frame_count - 1, num=min(frame_count, snapshot_count), dtype=np.int64)
    )


def _subsample_frame_points(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    max_points: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    value_arr = np.asarray(values)
    label_arr = np.asarray(labels, dtype=int).reshape(-1)
    if value_arr.shape[0] != label_arr.shape[0]:
        raise ValueError(
            "values and labels length mismatch while subsampling frame points: "
            f"{value_arr.shape[0]} vs {label_arr.shape[0]}."
        )
    selected = _sample_indices_stratified(
        label_arr,
        None if max_points is None else int(max_points),
        random_seed=int(random_state),
    )
    return value_arr[selected], label_arr[selected]


def _build_temporal_md_records(
    coords: np.ndarray,
    labels: np.ndarray,
    frame_indices: np.ndarray,
    timesteps: np.ndarray,
    *,
    max_points_per_frame: int | None,
    random_state: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for local_frame_idx in range(coords.shape[0]):
        frame_coords, frame_labels = _subsample_frame_points(
            np.asarray(coords[local_frame_idx], dtype=np.float32),
            np.asarray(labels[local_frame_idx], dtype=int),
            max_points=max_points_per_frame,
            random_state=int(random_state + local_frame_idx),
        )
        records.append(
            {
                "frame_name": str(frame_indices[local_frame_idx]),
                "frame_label": _format_frame_label(
                    frame_index=int(frame_indices[local_frame_idx]),
                    timestep=int(timesteps[local_frame_idx]),
                ),
                "coords": frame_coords.astype(np.float32, copy=False),
                "labels": frame_labels.astype(int, copy=False),
            }
        )
    return records


def _build_temporal_embedding_records(
    embedding: np.ndarray,
    labels: np.ndarray,
    frame_indices: np.ndarray,
    timesteps: np.ndarray,
    *,
    max_points_per_frame: int | None,
    random_state: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for local_frame_idx in range(embedding.shape[0]):
        frame_embedding, frame_labels = _subsample_frame_points(
            np.asarray(embedding[local_frame_idx], dtype=np.float32),
            np.asarray(labels[local_frame_idx], dtype=int),
            max_points=max_points_per_frame,
            random_state=int(random_state + local_frame_idx),
        )
        records.append(
            {
                "frame_name": str(frame_indices[local_frame_idx]),
                "frame_label": _format_frame_label(
                    frame_index=int(frame_indices[local_frame_idx]),
                    timestep=int(timesteps[local_frame_idx]),
                ),
                "embedding": frame_embedding.astype(np.float32, copy=False),
                "labels": frame_labels.astype(int, copy=False),
            }
        )
    return records


def _build_temporal_trajectory_records(
    embedding: np.ndarray,
    labels: np.ndarray,
    atom_ids: np.ndarray,
    frame_indices: np.ndarray,
    timesteps: np.ndarray,
    *,
    max_points: int | None,
    random_state: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    reference_labels = np.asarray(labels[0], dtype=int)
    selected_atom_offsets = _sample_indices_stratified(
        reference_labels,
        None if max_points is None else int(max_points),
        random_seed=int(random_state),
    )
    selected_atom_ids = np.asarray(atom_ids, dtype=np.int64)[selected_atom_offsets]
    records: list[dict[str, Any]] = []
    for local_frame_idx in range(embedding.shape[0]):
        records.append(
            {
                "frame_name": str(frame_indices[local_frame_idx]),
                "frame_label": _format_frame_label(
                    frame_index=int(frame_indices[local_frame_idx]),
                    timestep=int(timesteps[local_frame_idx]),
                ),
                "embedding": np.asarray(
                    embedding[local_frame_idx, selected_atom_offsets],
                    dtype=np.float32,
                ),
                "labels": np.asarray(labels[local_frame_idx, selected_atom_offsets], dtype=int),
                "instance_ids": selected_atom_ids.astype(np.int64, copy=False),
            }
        )
    return records, selected_atom_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster and visualize VAMP-space coordinates using a VAMP config."
    )
    parser.add_argument("config", help="Config name inside configs/vamp/ or a YAML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, config_path, base_dir = load_vamp_config(args.config)
    analyze_cfg = cfg.analyze
    embeddings_path = resolve_path(analyze_cfg.embeddings_path, base_dir=base_dir)
    fit_dir_text = resolve_path(analyze_cfg.fit_dir, base_dir=base_dir)
    if embeddings_path is None:
        raise ValueError("analyze.embeddings_path must be set in the VAMP config.")
    if fit_dir_text is None:
        raise ValueError("analyze.fit_dir must be set in the VAMP config.")
    fit_dir = Path(fit_dir_text).expanduser().resolve()
    output_dir_text = resolve_path(getattr(analyze_cfg, "output_dir", None), base_dir=base_dir)
    output_dir = ensure_dir((fit_dir.parent / "analysis") if output_dir_text is None else output_dir_text)
    figures_dir = ensure_dir(output_dir / "figures")
    artifacts_dir = ensure_dir(output_dir / "artifacts")
    representatives_dir = ensure_dir(output_dir / "representatives")
    time_series_dir = ensure_dir(output_dir / "time_series")
    temporal_dir = ensure_dir(output_dir / "temporal")
    md_space_dir = ensure_dir(output_dir / "md_space")
    analysis_window = str(getattr(analyze_cfg, "analysis_window", "fit"))
    random_state = int(getattr(analyze_cfg, "random_state", 0))
    k = int(getattr(analyze_cfg, "k", 6))
    max_vis_requested = int(getattr(analyze_cfg, "visualization_max_samples", 20000))
    snapshot_count = int(getattr(analyze_cfg, "snapshot_count", 6))
    compare_structural_clustering = bool(getattr(analyze_cfg, "compare_structural_clustering", False))
    time_series_cfg = getattr(analyze_cfg, "time_series", None)
    temporal_cfg = getattr(analyze_cfg, "temporal", None)
    md_space_cfg = getattr(analyze_cfg, "md_space", None)
    start = time.perf_counter()
    log_progress(
        "analyze_vamp",
        f"loading embeddings from {Path(embeddings_path).expanduser().resolve()} using config={config_path.name}",
    )
    embeddings = TrajectoryEmbeddings.load(embeddings_path)
    if not fit_dir.exists():
        raise FileNotFoundError(f"fit_dir does not exist: {fit_dir}")
    summary_candidates = [
        fit_dir / "artifacts" / "summary.json",
        fit_dir / "summary.json",
    ]
    summary_path = None
    for candidate in summary_candidates:
        if candidate.exists():
            summary_path = candidate
            break
    if summary_path is None:
        raise FileNotFoundError(
            "fit summary is missing. "
            f"Tried {[str(candidate) for candidate in summary_candidates]}."
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        fit_summary = json.load(handle)
    model = ManualVAMP.load(fit_summary["selected_model_path"])
    log_progress(
        "analyze_vamp",
        (
            f"loaded fit summary from {summary_path}; selected_lag={fit_summary['selected_lag']}, "
            f"output_dir={output_dir}"
        ),
    )

    if analysis_window == "fit":
        fit_window = fit_summary["window"]
        window = resolve_frame_window(
            embeddings.frame_count,
            window="custom",
            frame_start=int(fit_window["start"]),
            frame_stop=int(fit_window["stop"]),
        )
    else:
        window = resolve_frame_window(
            embeddings.frame_count,
            window=analysis_window,
            frame_start=getattr(analyze_cfg, "frame_start", None),
            frame_stop=getattr(analyze_cfg, "frame_stop", None),
        )

    projection_dim = (
        int(fit_summary["projection_dim"])
        if getattr(analyze_cfg, "projection_dim", None) is None
        else int(analyze_cfg.projection_dim)
    )
    projection_dim = min(projection_dim, int(model.model_dim))
    if projection_dim <= 0:
        raise ValueError(f"projection_dim must be > 0, got {projection_dim}.")

    scaling = fit_summary.get("projection_scaling", None)
    log_progress(
        "analyze_vamp",
        (
            f"projecting embeddings into VAMP space for window={window.to_dict()} "
            f"projection_dim={projection_dim} scaling={scaling}"
        ),
    )
    flat_embeddings = embeddings.invariant_embeddings.reshape(-1, embeddings.latent_dim)
    vamp_coords = model.transform_instantaneous(
        flat_embeddings,
        dim=projection_dim,
        scaling=scaling,
    ).reshape(embeddings.frame_count, embeddings.num_atoms, projection_dim)

    frame_slice = slice(int(window.start), int(window.stop))
    window_coords = np.asarray(vamp_coords[frame_slice], dtype=np.float32)
    window_structural = np.asarray(embeddings.invariant_embeddings[frame_slice], dtype=np.float32)
    window_centers = np.asarray(embeddings.center_positions[frame_slice], dtype=np.float32)
    window_timesteps = np.asarray(embeddings.timesteps[frame_slice], dtype=np.int64)
    window_frame_indices = np.asarray(embeddings.frame_indices[frame_slice], dtype=np.int64)

    flat_vamp = window_coords.reshape(-1, projection_dim)
    flat_frame_indices = np.repeat(window_frame_indices, embeddings.num_atoms)
    flat_atom_ids = np.tile(np.asarray(embeddings.atom_ids, dtype=np.int64), window_coords.shape[0])
    log_progress(
        "analyze_vamp",
        f"running KMeans in VAMP space on {flat_vamp.shape[0]} samples with k={k}",
    )
    vamp_labels_flat, vamp_centers = _run_kmeans(
        flat_vamp,
        k=k,
        random_state=random_state,
    )
    vamp_labels = vamp_labels_flat.reshape(window_coords.shape[:2])
    cluster_color_map = _build_cluster_color_map(k)
    cluster_ids = [int(cluster_id) for cluster_id in range(k)]
    cluster_display_map = {
        int(cluster_id): f"C{int(pos) + 1}"
        for pos, cluster_id in enumerate(cluster_ids)
    }
    cluster_display_labels = [cluster_display_map[int(cluster_id)] for cluster_id in cluster_ids]
    populations = np.stack(
        [
            np.mean(vamp_labels == cluster_id, axis=1)
            for cluster_id in range(k)
        ],
        axis=1,
    )
    counts_matrix = np.stack(
        [
            np.sum(vamp_labels == cluster_id, axis=1)
            for cluster_id in range(k)
        ],
        axis=1,
    ).astype(np.float64, copy=False)
    frame_labels = [
        _format_frame_label(frame_index=int(frame_index), timestep=int(timestep))
        for frame_index, timestep in zip(window_frame_indices.tolist(), window_timesteps.tolist(), strict=False)
    ]

    population_csv_path = artifacts_dir / "cluster_populations.csv"
    with population_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["frame_index", "timestep"] + [f"cluster_{idx}" for idx in range(k)]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for local_frame_idx in range(window_coords.shape[0]):
            row = {
                "frame_index": int(window_frame_indices[local_frame_idx]),
                "timestep": int(window_timesteps[local_frame_idx]),
            }
            for cluster_id in range(k):
                row[f"cluster_{cluster_id}"] = float(populations[local_frame_idx, cluster_id])
            writer.writerow(row)

    np.savez_compressed(
        artifacts_dir / "vamp_coordinates.npz",
        vamp_coordinates=window_coords.astype(np.float32, copy=False),
        cluster_labels=vamp_labels.astype(np.int64, copy=False),
        frame_indices=window_frame_indices,
        timesteps=window_timesteps,
        atom_ids=embeddings.atom_ids,
    )

    time_series_summary = None
    if bool(getattr(time_series_cfg, "enabled", True)):
        log_progress("analyze_vamp", f"writing stacked-area cluster proportions to {time_series_dir}")
        time_series_summary = save_cluster_proportion_plots(
            frame_labels,
            counts_matrix,
            cluster_ids,
            time_series_dir,
            cluster_color_map=cluster_color_map,
            cluster_display_labels=cluster_display_labels,
            save_paper_svg=bool(getattr(time_series_cfg, "paper_enabled", True)),
            stack_alpha=float(getattr(time_series_cfg, "alpha", 0.78)),
            paper_alpha=float(getattr(time_series_cfg, "paper_alpha", 0.72)),
        )

    _plot_cluster_populations(
        window_timesteps,
        populations,
        ylabel="cluster population fraction",
        out_path=figures_dir / "cluster_populations.png",
        title="VAMP-space cluster populations vs time",
    )
    _plot_spatial_snapshots(
        window_centers,
        vamp_labels,
        window_timesteps,
        out_path=figures_dir / "spatial_cluster_snapshots.png",
        snapshot_count=snapshot_count,
    )

    sample_count = flat_vamp.shape[0]
    max_vis = max(2, min(max_vis_requested, sample_count))
    if sample_count > max_vis:
        rng = np.random.default_rng(random_state)
        vis_indices = np.sort(rng.choice(sample_count, size=max_vis, replace=False))
    else:
        vis_indices = np.arange(sample_count, dtype=np.int64)
    vis_features = flat_vamp[vis_indices]
    vis_labels = vamp_labels_flat[vis_indices]
    vis_frame_values = np.repeat(window_frame_indices, embeddings.num_atoms)[vis_indices]
    vis_timesteps = np.repeat(window_timesteps, embeddings.num_atoms)[vis_indices]

    mode_plot_count = min(4, projection_dim)
    if mode_plot_count > 0:
        _plot_vamp_mode_dynamics(
            window_timesteps,
            window_coords,
            out_path=figures_dir / "vamp_mode_dynamics.png",
            max_modes=mode_plot_count,
        )
        _plot_cluster_center_heatmap(
            vamp_centers[:, :mode_plot_count],
            out_path=figures_dir / "vamp_cluster_centers.png",
        )
    if projection_dim >= 2:
        _plot_vamp_phase_space(
            vis_features,
            vis_labels,
            vis_timesteps,
            vamp_centers[:, : min(3, projection_dim)],
            out_clusters=figures_dir / "vamp_phase_space_clusters.png",
            out_time=figures_dir / "vamp_phase_space_time.png",
            max_modes=min(3, projection_dim),
        )

    log_progress(
        "analyze_vamp",
        f"computing t-SNE/UMAP on {vis_features.shape[0]} visualization samples",
    )
    tsne_coords = compute_tsne(vis_features, random_state=random_state, n_iter=1500)
    _plot_2d_embedding(
        tsne_coords,
        vis_labels,
        vis_frame_values,
        out_clusters=figures_dir / "vamp_tsne_clusters.png",
        out_time=figures_dir / "vamp_tsne_time.png",
        title_prefix="VAMP t-SNE",
    )
    umap_coords = compute_umap(
        vis_features,
        random_state=random_state,
        return_info=False,
    )
    _plot_2d_embedding(
        umap_coords,
        vis_labels,
        vis_frame_values,
        out_clusters=figures_dir / "vamp_umap_clusters.png",
        out_time=figures_dir / "vamp_umap_time.png",
        title_prefix="VAMP UMAP",
    )

    temporal_summary = None
    if bool(getattr(temporal_cfg, "enabled", False)):
        log_progress("analyze_vamp", f"building temporal VAMP/MD animations in {temporal_dir}")
        animation_max_points_raw = getattr(temporal_cfg, "animation_max_points", None)
        animation_max_points = (
            None
            if animation_max_points_raw is None or int(animation_max_points_raw) <= 0
            else int(animation_max_points_raw)
        )
        frame_duration_ms = int(getattr(temporal_cfg, "frame_duration_ms", 450))
        total_duration_seconds_raw = getattr(temporal_cfg, "total_duration_seconds", None)
        total_duration_seconds = (
            None if total_duration_seconds_raw is None else float(total_duration_seconds_raw)
        )
        temporal_summary = {
            "root_dir": str(temporal_dir),
            "frame_count": int(window_coords.shape[0]),
            "render_max_points_per_frame": (
                None if animation_max_points is None else int(animation_max_points)
            ),
        }

        temporal_md_cfg = getattr(temporal_cfg, "md_space", None)
        if bool(getattr(temporal_md_cfg, "enabled", True)):
            md_animation_path = temporal_dir / f"md_space_clusters_diagonal_cut_k{k}.gif"
            md_records = _build_temporal_md_records(
                window_centers,
                vamp_labels,
                window_frame_indices,
                window_timesteps,
                max_points_per_frame=animation_max_points,
                random_state=random_state,
            )
            temporal_summary["md_space_animation"] = save_temporal_spatial_cluster_animation(
                md_records,
                md_animation_path,
                cluster_color_map=cluster_color_map,
                cluster_display_map=cluster_display_map,
                point_size=float(getattr(temporal_md_cfg, "point_size", 12.0)),
                alpha=float(getattr(temporal_md_cfg, "alpha", 1.0)),
                saturation_boost=float(getattr(temporal_md_cfg, "saturation_boost", 1.18)),
                view_elev=float(getattr(temporal_md_cfg, "view_elev", 24.0)),
                view_azim=float(getattr(temporal_md_cfg, "view_azim", 35.0)),
                diagonal_visible_depth_fraction=float(
                    getattr(temporal_md_cfg, "diagonal_visible_depth_fraction", 0.10)
                ),
                frame_duration_ms=frame_duration_ms,
                total_duration_seconds=total_duration_seconds,
            )

        temporal_vamp_cfg = getattr(temporal_cfg, "vamp_space", None)
        if projection_dim >= 2 and bool(getattr(temporal_vamp_cfg, "enabled", True)):
            vamp_animation_path = temporal_dir / "vamp_space_clusters.gif"
            vamp_records = _build_temporal_embedding_records(
                np.asarray(window_coords[:, :, :2], dtype=np.float32),
                vamp_labels,
                window_frame_indices,
                window_timesteps,
                max_points_per_frame=animation_max_points,
                random_state=random_state + 17,
            )
            temporal_summary["vamp_space_animation"] = save_temporal_embedding_cluster_animation(
                vamp_records,
                vamp_animation_path,
                cluster_color_map=cluster_color_map,
                cluster_display_map=cluster_display_map,
                point_size=float(getattr(temporal_vamp_cfg, "point_size", 8.0)),
                alpha=float(getattr(temporal_vamp_cfg, "alpha", 0.74)),
                frame_duration_ms=frame_duration_ms,
                total_duration_seconds=total_duration_seconds,
                title="VAMP-space cluster evolution",
                xlabel="psi1",
                ylabel="psi2",
            )
            if bool(getattr(temporal_vamp_cfg, "trajectory_enabled", True)):
                trajectory_max_points_raw = getattr(
                    temporal_vamp_cfg,
                    "trajectory_max_points",
                    animation_max_points,
                )
                trajectory_max_points = (
                    None
                    if trajectory_max_points_raw is None or int(trajectory_max_points_raw) <= 0
                    else int(trajectory_max_points_raw)
                )
                trajectory_records, trajectory_atom_ids = _build_temporal_trajectory_records(
                    np.asarray(window_coords[:, :, :2], dtype=np.float32),
                    vamp_labels,
                    np.asarray(embeddings.atom_ids, dtype=np.int64),
                    window_frame_indices,
                    window_timesteps,
                    max_points=trajectory_max_points,
                    random_state=random_state + 31,
                )
                trajectory_path = temporal_dir / "vamp_space_trajectories.gif"
                temporal_summary["vamp_space_trajectory_animation"] = save_temporal_embedding_trajectory_animation(
                    trajectory_records,
                    trajectory_path,
                    cluster_color_map=cluster_color_map,
                    cluster_display_map=cluster_display_map,
                    line_width=float(getattr(temporal_vamp_cfg, "trajectory_line_width", 0.8)),
                    line_alpha=float(getattr(temporal_vamp_cfg, "trajectory_line_alpha", 0.22)),
                    history_steps=int(getattr(temporal_vamp_cfg, "trajectory_history_steps", 8)),
                    fade_min_alpha_fraction=float(
                        getattr(temporal_vamp_cfg, "trajectory_fade_min_alpha_fraction", 0.18)
                    ),
                    fade_power=float(getattr(temporal_vamp_cfg, "trajectory_fade_power", 1.0)),
                    directional_subsegments=int(
                        getattr(temporal_vamp_cfg, "trajectory_directional_subsegments", 6)
                    ),
                    directional_start_alpha_fraction=float(
                        getattr(temporal_vamp_cfg, "trajectory_directional_start_alpha_fraction", 0.32)
                    ),
                    directional_start_width_fraction=float(
                        getattr(temporal_vamp_cfg, "trajectory_directional_start_width_fraction", 0.60)
                    ),
                    directional_end_width_fraction=float(
                        getattr(temporal_vamp_cfg, "trajectory_directional_end_width_fraction", 1.35)
                    ),
                    endpoint_point_size=float(
                        getattr(temporal_vamp_cfg, "trajectory_endpoint_point_size", 3.0)
                    ),
                    endpoint_point_alpha=float(
                        getattr(temporal_vamp_cfg, "trajectory_endpoint_point_alpha", 0.95)
                    ),
                    frame_duration_ms=frame_duration_ms,
                    total_duration_seconds=total_duration_seconds,
                    title="VAMP-space cluster trajectories",
                    xlabel="psi1",
                    ylabel="psi2",
                )
                temporal_summary["trajectory_sample_count"] = int(trajectory_atom_ids.size)

    md_space_summary = None
    if bool(getattr(md_space_cfg, "enabled", False)):
        log_progress("analyze_vamp", f"writing MD-space outputs to {md_space_dir}")
        flat_centers = window_centers.reshape(-1, 3).astype(np.float32, copy=False)
        md_static_path = md_space_dir / f"md_space_clusters_k{k}.png"
        save_md_space_clusters_plot(
            flat_centers,
            vamp_labels_flat,
            md_static_path,
            cluster_color_map=cluster_color_map,
            max_points=getattr(md_space_cfg, "static_max_points", None),
            title=f"MD-space local-environment clusters (k={k}, n={int(flat_centers.shape[0])})",
        )
        md_space_summary = {
            "root_dir": str(md_space_dir),
            "static_png": str(md_static_path),
        }
        if bool(getattr(md_space_cfg, "interactive_enabled", True)):
            md_interactive_path = md_space_dir / f"md_space_clusters_k{k}.html"
            save_interactive_md_plot(
                flat_centers,
                vamp_labels_flat,
                md_interactive_path,
                palette="tab10",
                cluster_color_map=cluster_color_map,
                max_points=getattr(md_space_cfg, "interactive_max_points", None),
                marker_size=float(getattr(md_space_cfg, "interactive_marker_size", 3.0)),
                marker_line_width=float(
                    getattr(md_space_cfg, "interactive_marker_line_width", 0.0)
                ),
                title=f"MD-space local-environment clusters (k={k}, n={int(flat_centers.shape[0])})",
                aspect_mode=str(getattr(md_space_cfg, "interactive_aspect_mode", "cube")),
            )
            md_space_summary["interactive_html"] = str(md_interactive_path)

        if bool(getattr(md_space_cfg, "raytrace_enabled", False)):
            raytrace_dir = ensure_dir(md_space_dir / "raytrace")
            raytrace_offsets = _sample_frame_offsets(
                window_centers.shape[0],
                int(getattr(md_space_cfg, "raytrace_snapshot_count", snapshot_count)),
            )
            raytrace_records: list[dict[str, Any]] = []
            for local_frame_idx in raytrace_offsets.tolist():
                frame_index = int(window_frame_indices[local_frame_idx])
                timestep = int(window_timesteps[local_frame_idx])
                raytrace_path = raytrace_dir / f"frame_{frame_index:04d}_t{timestep}_clusters_raytrace.png"
                raytrace_info = _save_md_cluster_snapshot_raytrace_blender(
                    np.asarray(window_centers[local_frame_idx], dtype=np.float32),
                    np.asarray(vamp_labels[local_frame_idx], dtype=int),
                    cluster_color_map,
                    raytrace_path,
                    title=f"MD-space local-environment clusters | {_format_frame_label(frame_index=frame_index, timestep=timestep)}",
                    view_elev=float(getattr(md_space_cfg, "view_elev", 24.0)),
                    view_azim=float(getattr(md_space_cfg, "view_azim", 35.0)),
                    max_points=getattr(md_space_cfg, "raytrace_max_points", None),
                    image_width=int(getattr(md_space_cfg, "raytrace_resolution", 1600)),
                    image_height=int(getattr(md_space_cfg, "raytrace_resolution", 1600)),
                    projection=str(getattr(md_space_cfg, "raytrace_projection", "perspective")),
                    perspective_fov_deg=float(getattr(md_space_cfg, "raytrace_fov_deg", 34.0)),
                    camera_distance_factor=float(
                        getattr(md_space_cfg, "raytrace_camera_distance_factor", 2.8)
                    ),
                    sphere_radius_fraction=float(
                        getattr(md_space_cfg, "raytrace_sphere_radius_fraction", 0.0105)
                    ),
                    blender_executable=str(getattr(md_space_cfg, "blender_executable", "blender")),
                    cycles_samples=int(getattr(md_space_cfg, "raytrace_samples", 64)),
                    timeout_seconds=int(getattr(md_space_cfg, "raytrace_timeout_sec", 1200)),
                    use_gpu=bool(getattr(md_space_cfg, "raytrace_use_gpu", False)),
                )
                raytrace_records.append(
                    {
                        "frame_index": frame_index,
                        "timestep": timestep,
                        "render": raytrace_info,
                    }
                )
            md_space_summary["raytrace"] = {
                "root_dir": str(raytrace_dir),
                "frames": raytrace_records,
            }

    representative_index_map = _compute_cluster_representative_indices(flat_vamp, vamp_labels_flat)
    representative_flat_indices = np.asarray(
        [representative_index_map[int(cluster_id)] for cluster_id in sorted(representative_index_map.keys())],
        dtype=np.int64,
    )
    representative_frame_offsets = representative_flat_indices // embeddings.num_atoms
    representative_atom_offsets = representative_flat_indices % embeddings.num_atoms
    representative_frame_indices = flat_frame_indices[representative_flat_indices]
    representative_atom_ids = flat_atom_ids[representative_flat_indices]
    log_progress(
        "analyze_vamp",
        f"loading representative neighborhoods for {k} cluster centers",
    )
    representative_neighborhoods = load_local_neighborhoods(
        embeddings,
        frame_indices=representative_frame_indices,
        atom_ids=representative_atom_ids,
        cache_dir=embeddings.metadata.get("cache_dir", None),
    )
    np.savez_compressed(
        artifacts_dir / "representative_neighborhoods.npz",
        neighborhoods=representative_neighborhoods.astype(np.float32, copy=False),
        cluster_ids=np.arange(k, dtype=np.int64),
        frame_indices=representative_frame_indices.astype(np.int64, copy=False),
        atom_ids=representative_atom_ids.astype(np.int64, copy=False),
        timesteps=window_timesteps[representative_frame_offsets].astype(np.int64, copy=False),
    )
    _plot_representative_neighborhoods(
        representative_neighborhoods,
        out_path=figures_dir / "representative_neighborhoods.png",
    )
    log_progress(
        "analyze_vamp",
        f"rendering analysis-style representative neighborhoods in {representatives_dir}",
    )
    representative_style_summary = _save_cluster_representatives_figure(
        _RepresentativeNeighborhoodDataset(
            embeddings,
            flat_frame_indices,
            flat_atom_ids,
            cache_dir=embeddings.metadata.get("cache_dir", None),
        ),
        flat_vamp,
        vamp_labels_flat,
        cluster_color_map,
        representatives_dir / f"04_cluster_representatives_k{k}.png",
        point_scale=1.0,
        target_points=64,
        orientation_method="pca",
        view_elev=22.0,
        view_azim=38.0,
        projection="ortho",
        representative_ptm_enabled=False,
        representative_cna_enabled=False,
    )

    summary: dict[str, Any] = {
        "window": window.to_dict(),
        "projection_dim": int(projection_dim),
        "cluster_count": k,
        "selected_lag": int(fit_summary["selected_lag"]),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "artifacts_dir": str(artifacts_dir),
        "figures_dir": str(figures_dir),
        "time_series_dir": str(time_series_dir),
        "temporal_dir": str(temporal_dir),
        "md_space_dir": str(md_space_dir),
        "cluster_populations_csv": str(population_csv_path),
        "cluster_display_map": {
            str(cluster_id): str(cluster_display_map[int(cluster_id)])
            for cluster_id in cluster_ids
        },
        "mode_plot_count": int(mode_plot_count),
        "representative_frame_indices": representative_frame_indices.astype(int).tolist(),
        "representative_atom_ids": representative_atom_ids.astype(int).tolist(),
        "representatives": {
            "root_dir": str(representatives_dir),
            "shared_style": representative_style_summary,
            "primary_figure": str(
                representative_style_summary["pca_two_shell_figures"]["spatial_neighbors_paper"]["out_file"]
            ),
            "edge_connected_figure": str(
                representative_style_summary["pca_two_shell_figures"]["knn_edges"]["out_file"]
            ),
        },
    }
    if time_series_summary is not None:
        summary["time_series"] = time_series_summary
    if temporal_summary is not None:
        summary["temporal"] = temporal_summary
    if md_space_summary is not None:
        summary["md_space"] = md_space_summary

    if compare_structural_clustering:
        flat_structural = window_structural.reshape(window_structural.shape[0] * window_structural.shape[1], -1)
        log_progress(
            "analyze_vamp",
            f"running structural-only KMeans comparison on {flat_structural.shape[0]} samples",
        )
        structural_labels_flat, _ = _run_kmeans(
            flat_structural,
            k=k,
            random_state=random_state,
        )
        structural_labels = structural_labels_flat.reshape(window_structural.shape[:2])
        structural_populations = np.stack(
            [
                np.mean(structural_labels == cluster_id, axis=1)
                for cluster_id in range(k)
            ],
            axis=1,
        )
        _plot_cluster_populations(
            window_timesteps,
            structural_populations,
            ylabel="cluster population fraction",
            out_path=figures_dir / "structural_cluster_populations.png",
            title="Structural-only cluster populations vs time",
        )
        summary["structural_comparison"] = {
            "ari": float(adjusted_rand_score(vamp_labels_flat, structural_labels_flat)),
            "nmi": float(normalized_mutual_info_score(vamp_labels_flat, structural_labels_flat)),
        }

    with (artifacts_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    elapsed = time.perf_counter() - start
    log_progress("analyze_vamp", f"wrote analysis artifacts to {output_dir} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
