from __future__ import annotations

import csv
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .config import VisualizationConfig, load_temporal_config

_DIAGONAL_DIRECTION = np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3.0)


@dataclass(frozen=True)
class TemporalVisualizationResult:
    output_dir: Path
    manifest_path: Path
    files: list[Path]


@dataclass
class TemporalFrameStore:
    mode: str
    num_frames: int
    frame_dirs: list[Path] | None = None
    atoms: np.ndarray | None = None
    state_ids: np.ndarray | None = None


def generate_temporal_visualizations(
    dataset_dir: str | Path,
    *,
    visualization_config: VisualizationConfig | None = None,
    output_dir: str | Path | None = None,
) -> TemporalVisualizationResult:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Temporal dataset directory does not exist: {dataset_path}.")
    config_path = dataset_path / "config_snapshot.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Temporal dataset directory is missing config_snapshot.yaml: {config_path}."
        )

    config = load_temporal_config(config_path)
    viz_cfg = visualization_config or config.visualization
    if not viz_cfg.enabled:
        raise ValueError(
            f"Visualization is disabled for dataset {dataset_path}. "
            "Pass an explicit VisualizationConfig(enabled=True, ...) to force rerendering."
        )

    output_path = Path(output_dir) if output_dir is not None else dataset_path / viz_cfg.output_subdir
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[temporal-viz] Writing visualizations to {output_path}", flush=True)

    artifacts = _load_visualization_artifacts(dataset_path)
    frame_store = _open_frame_store(dataset_path)
    state_ids = artifacts["latent"]["state_ids"]
    transition_mask = artifacts["latent"]["transition_mask"]
    grain_ids = artifacts["latent"]["grain_ids"]
    site_centers = artifacts["layout"]["centers"]
    state_names = artifacts["graph"]["state_names"]
    transition_events = artifacts["transition_events"]
    trajectory_pack = artifacts["neighborhoods"]
    selected_frames = _select_frame_indices(state_ids.shape[0], viz_cfg.max_frames_to_plot)
    selected_sites = _select_site_indices(
        state_ids=state_ids,
        transition_mask=transition_mask,
        max_sites=viz_cfg.max_sites_in_gallery,
    )

    color_lookup, listed_cmap, norm = _state_palette(state_names)
    files: list[Path] = []

    try:
        occupancy_path = output_path / "state_occupancy_over_time.png"
        print("[temporal-viz] Rendering state occupancy plot", flush=True)
        _plot_state_occupancy(
            output_path=occupancy_path,
            state_ids=state_ids,
            state_names=state_names,
            color_lookup=color_lookup,
        )
        files.append(occupancy_path)

        raster_path = output_path / "site_state_raster.png"
        print("[temporal-viz] Rendering site-state raster", flush=True)
        _plot_site_state_raster(
            output_path=raster_path,
            state_ids=state_ids,
            transition_mask=transition_mask,
            state_names=state_names,
            cmap=listed_cmap,
            norm=norm,
        )
        files.append(raster_path)

        transition_path = output_path / "transition_matrix.png"
        print("[temporal-viz] Rendering transition matrix", flush=True)
        _plot_transition_matrix(
            output_path=transition_path,
            transition_events=transition_events,
            state_names=state_names,
        )
        files.append(transition_path)

        frame_path = output_path / "frame_snapshots.png"
        print("[temporal-viz] Rendering frame snapshots", flush=True)
        _plot_frame_snapshots(
            output_path=frame_path,
            frame_store=frame_store,
            selected_frames=selected_frames,
            color_lookup=color_lookup,
            box_size=float(config.domain.box_size),
            max_atoms_per_frame=viz_cfg.max_atoms_per_frame,
            slice_axis=viz_cfg.frame_slice_axis,
            slice_relative_thickness=viz_cfg.frame_slice_relative_thickness,
        )
        files.append(frame_path)

        gallery_path = output_path / "local_trajectory_gallery.png"
        print("[temporal-viz] Rendering local trajectory gallery", flush=True)
        _plot_local_trajectory_gallery(
            output_path=gallery_path,
            trajectory_points=trajectory_pack["points"],
            state_ids=state_ids,
            selected_frames=selected_frames,
            selected_sites=selected_sites,
            state_names=state_names,
            color_lookup=color_lookup,
        )
        files.append(gallery_path)

        if viz_cfg.write_interactive_html:
            html_path = output_path / "site_state_evolution.html"
            print("[temporal-viz] Rendering interactive site-state HTML", flush=True)
            _write_interactive_site_evolution(
                output_path=html_path,
                site_centers=site_centers,
                state_ids=state_ids,
                grain_ids=grain_ids,
                state_names=state_names,
                color_lookup=color_lookup,
            )
            files.append(html_path)

        if viz_cfg.write_animations:
            diagonal_gif_path = output_path / "all_phases_diagonal_cut.gif"
            print("[temporal-viz] Rendering diagonal-cut animation", flush=True)
            _write_phase_animation(
                output_path=diagonal_gif_path,
                frame_store=frame_store,
                box_size=float(config.domain.box_size),
                avg_nn_distance=float(config.domain.avg_nn_distance),
                state_names=state_names,
                color_lookup=color_lookup,
                max_atoms_per_frame=viz_cfg.animation_max_atoms_per_frame,
                mode="diagonal_cut_all_phases",
                diagonal_cut_fraction=viz_cfg.animation_diagonal_cut_fraction,
                diagonal_visible_depth_nn=viz_cfg.animation_diagonal_visible_depth_nn,
                diagonal_marker_size=viz_cfg.animation_diagonal_marker_size,
                full_box_marker_size=viz_cfg.animation_full_box_marker_size,
                parallel_workers=viz_cfg.parallel_workers,
            )
            files.append(diagonal_gif_path)

            solid_gif_path = output_path / "solid_only_full_box.gif"
            print("[temporal-viz] Rendering solid-only full-box animation", flush=True)
            _write_phase_animation(
                output_path=solid_gif_path,
                frame_store=frame_store,
                box_size=float(config.domain.box_size),
                avg_nn_distance=float(config.domain.avg_nn_distance),
                state_names=state_names,
                color_lookup=color_lookup,
                max_atoms_per_frame=viz_cfg.animation_max_atoms_per_frame,
                mode="solid_only_full_box",
                diagonal_cut_fraction=viz_cfg.animation_diagonal_cut_fraction,
                diagonal_visible_depth_nn=viz_cfg.animation_diagonal_visible_depth_nn,
                diagonal_marker_size=viz_cfg.animation_diagonal_marker_size,
                full_box_marker_size=viz_cfg.animation_full_box_marker_size,
                parallel_workers=viz_cfg.parallel_workers,
            )
            files.append(solid_gif_path)
    finally:
        _close_frame_store(frame_store)

    manifest = {
        "dataset_dir": str(dataset_path),
        "output_dir": str(output_path),
        "selected_frames": [int(item) for item in selected_frames],
        "selected_sites": [int(item) for item in selected_sites],
        "files": [str(path) for path in files],
    }
    manifest_path = output_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return TemporalVisualizationResult(output_dir=output_path, manifest_path=manifest_path, files=files)


def _load_visualization_artifacts(dataset_dir: Path) -> dict[str, Any]:
    latent_path = dataset_dir / "latent" / "site_latent_trajectories.npz"
    layout_path = dataset_dir / "site_layout.npz"
    graph_path = dataset_dir / "transition_graph.json"
    transitions_path = dataset_dir / "latent" / "transition_events.csv"
    trajectory_path = dataset_dir / "neighborhoods" / "trajectory_pack.npz"

    for path in [latent_path, layout_path, graph_path, transitions_path, trajectory_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Temporal visualization requires artifact {path}, but it does not exist."
            )

    with graph_path.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    if "state_names" not in graph or not isinstance(graph["state_names"], list):
        raise ValueError(
            f"transition_graph.json at {graph_path} must contain a 'state_names' list."
        )

    transition_events = []
    with transitions_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            transition_events.append(
                {
                    "frame_index": int(row["frame_index"]),
                    "site_id": int(row["site_id"]),
                    "source_state": row["source_state"],
                    "target_state": row["target_state"],
                    "source_grain_id": int(row["source_grain_id"]),
                    "target_grain_id": int(row["target_grain_id"]),
                }
            )

    return {
        "latent": dict(np.load(latent_path)),
        "layout": dict(np.load(layout_path)),
        "graph": graph,
        "transition_events": transition_events,
        "neighborhoods": dict(np.load(trajectory_path)),
    }


def _open_frame_store(dataset_dir: Path) -> TemporalFrameStore:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Temporal visualization requires dataset manifest.json, but it does not exist: {manifest_path}."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    frame_storage = str(manifest.get("frame_storage", "frame_dirs"))
    if frame_storage == "frame_dirs":
        frames_dir = dataset_dir / "frames"
        if not frames_dir.exists():
            raise FileNotFoundError(
                f"Dataset manifest declares frame_dirs storage, but {frames_dir} does not exist."
            )
        frame_dirs = sorted(frames_dir.iterdir())
        if not frame_dirs:
            raise ValueError(f"No frame directories found in {frames_dir}.")
        return TemporalFrameStore(
            mode="frame_dirs",
            num_frames=len(frame_dirs),
            frame_dirs=frame_dirs,
        )
    if frame_storage == "single_chunk_npz":
        raw_path = manifest.get("frame_chunk_path")
        frame_chunk_path = Path(raw_path) if raw_path else dataset_dir / "frames_chunk.npz"
        if not frame_chunk_path.exists():
            raise FileNotFoundError(
                "Dataset manifest declares single_chunk_npz frame storage, "
                f"but the chunk file does not exist: {frame_chunk_path}."
            )
        archive = np.load(frame_chunk_path)
        required_arrays = {"atoms", "state_ids"}
        missing = sorted(required_arrays.difference(archive.files))
        if missing:
            archive.close()
            raise KeyError(
                f"Frame chunk store {frame_chunk_path} is missing required arrays: {missing}."
            )
        atoms = archive["atoms"]
        state_ids = archive["state_ids"]
        archive.close()
        if atoms.ndim != 3 or atoms.shape[2] != 3:
            raise ValueError(
                f"Chunked frame atoms must have shape (num_frames, num_atoms, 3), got {atoms.shape}."
            )
        if state_ids.shape[:2] != atoms.shape[:2]:
            raise ValueError(
                "Chunked frame state_ids must align with atoms along frame and atom axes, "
                f"got atoms.shape={atoms.shape}, state_ids.shape={state_ids.shape}."
            )
        return TemporalFrameStore(
            mode="single_chunk_npz",
            num_frames=int(atoms.shape[0]),
            atoms=atoms,
            state_ids=state_ids,
        )
    raise ValueError(
        "Temporal visualization supports only frame storage modes 'frame_dirs' and 'single_chunk_npz', "
        f"got {frame_storage!r} in {manifest_path}."
    )


def _close_frame_store(frame_store: TemporalFrameStore) -> None:
    del frame_store


def _load_frame_atoms_and_states(
    frame_store: TemporalFrameStore,
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    if frame_idx < 0 or frame_idx >= frame_store.num_frames:
        raise IndexError(
            f"Requested frame_idx={frame_idx}, but frame store contains {frame_store.num_frames} frames."
        )
    if frame_store.mode == "frame_dirs":
        if frame_store.frame_dirs is None:
            raise RuntimeError("frame_dirs storage mode is missing frame_dirs list.")
        frame_dir = frame_store.frame_dirs[frame_idx]
        atoms = np.load(frame_dir / "atoms.npy")
        atom_table = np.load(frame_dir / "atom_table.npz")
        return atoms, atom_table["state_ids"]
    if frame_store.mode == "single_chunk_npz":
        if frame_store.atoms is None or frame_store.state_ids is None:
            raise RuntimeError("single_chunk_npz storage mode is missing atoms or state_ids arrays.")
        return frame_store.atoms[frame_idx], frame_store.state_ids[frame_idx]
    raise ValueError(f"Unsupported frame store mode {frame_store.mode!r}.")


def _state_palette(
    state_names: list[str],
) -> tuple[dict[str, str], ListedColormap, BoundaryNorm]:
    base_colors = [
        "#3B82F6",
        "#F59E0B",
        "#EF4444",
        "#10B981",
        "#8B5CF6",
        "#EC4899",
        "#14B8A6",
        "#F97316",
        "#6366F1",
        "#84CC16",
    ]
    if len(state_names) > len(base_colors):
        raise ValueError(
            f"State palette currently supports up to {len(base_colors)} states, got {len(state_names)}."
        )
    color_lookup = {name: base_colors[idx] for idx, name in enumerate(state_names)}
    cmap = ListedColormap([color_lookup[name] for name in state_names], name="temporal_states")
    norm = BoundaryNorm(np.arange(len(state_names) + 1) - 0.5, cmap.N)
    return color_lookup, cmap, norm


def _plot_state_occupancy(
    *,
    output_path: Path,
    state_ids: np.ndarray,
    state_names: list[str],
    color_lookup: dict[str, str],
) -> None:
    frames = np.arange(state_ids.shape[0], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(10, 5))
    for state_idx, state_name in enumerate(state_names):
        counts = np.sum(state_ids == state_idx, axis=1)
        ax.plot(
            frames,
            counts,
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=color_lookup[state_name],
            label=state_name,
        )
    ax.set_title("State Occupancy Over Time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Site Count")
    ax.grid(alpha=0.25)
    ax.legend(ncols=min(4, len(state_names)), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_site_state_raster(
    *,
    output_path: Path,
    state_ids: np.ndarray,
    transition_mask: np.ndarray,
    state_names: list[str],
    cmap: ListedColormap,
    norm: BoundaryNorm,
) -> None:
    transition_counts = np.sum(transition_mask, axis=0)
    final_states = state_ids[-1]
    order = np.lexsort((transition_counts, final_states))
    raster = state_ids[:, order].T

    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(raster, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title("Site-State Raster")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Site (sorted by final state)")
    cbar = fig.colorbar(image, ax=ax, ticks=np.arange(len(state_names)))
    cbar.ax.set_yticklabels(state_names)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_transition_matrix(
    *,
    output_path: Path,
    transition_events: list[dict[str, Any]],
    state_names: list[str],
) -> None:
    index = {name: idx for idx, name in enumerate(state_names)}
    matrix = np.zeros((len(state_names), len(state_names)), dtype=np.int32)
    for event in transition_events:
        matrix[index[event["source_state"]], index[event["target_state"]]] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="magma")
    ax.set_title("Transition Counts")
    ax.set_xlabel("Target State")
    ax.set_ylabel("Source State")
    ax.set_xticks(np.arange(len(state_names)))
    ax.set_yticks(np.arange(len(state_names)))
    ax.set_xticklabels(state_names)
    ax.set_yticklabels(state_names)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            if value > 0:
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_frame_snapshots(
    *,
    output_path: Path,
    frame_store: TemporalFrameStore,
    selected_frames: list[int],
    color_lookup: dict[str, str],
    box_size: float,
    max_atoms_per_frame: int,
    slice_axis: int,
    slice_relative_thickness: float,
) -> None:
    axis_names = ("x", "y", "z")
    if slice_axis not in (0, 1, 2):
        raise ValueError(f"frame_slice_axis must be 0, 1, or 2, got {slice_axis}.")
    if slice_relative_thickness <= 0.0 or slice_relative_thickness > 1.0:
        raise ValueError(
            "frame_slice_relative_thickness must be in (0, 1], "
            f"got {slice_relative_thickness}."
        )

    view_axes = [axis for axis in (0, 1, 2) if axis != slice_axis]
    fig, axes = plt.subplots(
        1,
        len(selected_frames),
        figsize=(4.4 * len(selected_frames), 4.2),
        squeeze=False,
    )

    for col_idx, frame_idx in enumerate(selected_frames):
        atom_ax = axes[0, col_idx]
        atoms, atom_states = _load_frame_atoms_and_states(frame_store, frame_idx)
        sliced_atoms, sliced_states = _select_atoms_for_snapshot(
            atoms=atoms,
            atom_states=atom_states,
            frame_idx=frame_idx,
            max_atoms=max_atoms_per_frame,
            slice_axis=slice_axis,
            slice_relative_thickness=slice_relative_thickness,
        )
        atom_colors = [_color_for_state_index(int(state_idx), color_lookup) for state_idx in sliced_states]
        atom_ax.scatter(
            sliced_atoms[:, view_axes[0]],
            sliced_atoms[:, view_axes[1]],
            c=atom_colors,
            s=5,
            alpha=0.7,
            linewidths=0.0,
        )
        atom_ax.set_title(
            f"Frame {frame_idx}: Atom Slice ({axis_names[slice_axis]} slab)"
        )
        atom_ax.set_xlabel(axis_names[view_axes[0]])
        atom_ax.set_ylabel(axis_names[view_axes[1]])
        atom_ax.set_aspect("equal", adjustable="box")
        atom_ax.set_xlim(0.0, box_size)
        atom_ax.set_ylim(0.0, box_size)
        atom_ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _select_atoms_for_snapshot(
    *,
    atoms: np.ndarray,
    atom_states: np.ndarray,
    frame_idx: int,
    max_atoms: int,
    slice_axis: int,
    slice_relative_thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    axis_values = atoms[:, slice_axis]
    thickness = slice_relative_thickness * float(np.max(axis_values) - np.min(axis_values))
    if thickness <= 0.0:
        raise ValueError(
            f"Atom snapshot slice thickness collapsed to zero for frame {frame_idx} and slice_axis={slice_axis}."
        )
    center = float(np.median(axis_values))
    mask = np.abs(axis_values - center) <= 0.5 * thickness
    if int(np.count_nonzero(mask)) < min(300, max_atoms // 6):
        mask = np.ones(atoms.shape[0], dtype=bool)
    selected_atoms = atoms[mask]
    selected_states = atom_states[mask]
    if selected_atoms.shape[0] > max_atoms:
        rng = np.random.default_rng(19_937 + frame_idx)
        keep = rng.choice(selected_atoms.shape[0], size=max_atoms, replace=False)
        selected_atoms = selected_atoms[keep]
        selected_states = selected_states[keep]
    return selected_atoms.astype(np.float32), selected_states.astype(np.int16)


def _plot_local_trajectory_gallery(
    *,
    output_path: Path,
    trajectory_points: np.ndarray,
    state_ids: np.ndarray,
    selected_frames: list[int],
    selected_sites: list[int],
    state_names: list[str],
    color_lookup: dict[str, str],
) -> None:
    if not selected_sites:
        raise ValueError("Local trajectory gallery requires at least one selected site.")
    if not selected_frames:
        raise ValueError("Local trajectory gallery requires at least one selected frame.")

    fig, axes = plt.subplots(
        len(selected_sites),
        len(selected_frames),
        figsize=(3.0 * len(selected_frames), 3.0 * len(selected_sites)),
        squeeze=False,
    )
    radius = float(np.max(np.abs(trajectory_points[selected_frames][:, selected_sites])))
    if radius <= 0.0:
        radius = 1.0

    for row_idx, site_id in enumerate(selected_sites):
        for col_idx, frame_idx in enumerate(selected_frames):
            ax = axes[row_idx, col_idx]
            points = trajectory_points[frame_idx, site_id]
            state_name = state_names[int(state_ids[frame_idx, site_id])]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=points[:, 2],
                cmap="viridis",
                s=12,
                alpha=0.85,
                linewidths=0.0,
            )
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color_lookup[state_name])
                spine.set_linewidth(2.0)
            ax.set_title(f"site {site_id} | f{frame_idx} | {state_name}", fontsize=9)
    fig.suptitle("Tracked Local Neighborhood Trajectories", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_interactive_site_evolution(
    *,
    output_path: Path,
    site_centers: np.ndarray,
    state_ids: np.ndarray,
    grain_ids: np.ndarray,
    state_names: list[str],
    color_lookup: dict[str, str],
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Interactive temporal visualization requires plotly. "
            "Install the project requirements before using write_interactive_html=true."
        ) from exc

    def trace_for_frame(frame_idx: int) -> go.Scatter3d:
        frame_states = state_ids[frame_idx]
        colors = [_color_for_state_index(int(state_idx), color_lookup) for state_idx in frame_states]
        hover = [
            (
                f"site={site_id}<br>"
                f"state={state_names[int(frame_states[site_id])]}<br>"
                f"grain={int(grain_ids[frame_idx, site_id])}<br>"
                f"frame={frame_idx}"
            )
            for site_id in range(site_centers.shape[0])
        ]
        return go.Scatter3d(
            x=site_centers[:, 0],
            y=site_centers[:, 1],
            z=site_centers[:, 2],
            mode="markers",
            marker={"size": 5, "color": colors, "opacity": 0.9},
            text=hover,
            hoverinfo="text",
        )

    frames = [go.Frame(data=[trace_for_frame(frame_idx)], name=str(frame_idx)) for frame_idx in range(state_ids.shape[0])]
    figure = go.Figure(data=[trace_for_frame(0)], frames=frames)
    figure.update_layout(
        title="Site-State Evolution",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 550, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
                "showactive": False,
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {
                        "label": str(frame_idx),
                        "method": "animate",
                        "args": [
                            [str(frame_idx)],
                            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                        ],
                    }
                    for frame_idx in range(state_ids.shape[0])
                ],
            }
        ],
    )
    figure.write_html(str(output_path), include_plotlyjs="cdn")


def _write_phase_animation(
    *,
    output_path: Path,
    frame_store: TemporalFrameStore,
    box_size: float,
    avg_nn_distance: float,
    state_names: list[str],
    color_lookup: dict[str, str],
    max_atoms_per_frame: int,
    mode: str,
    diagonal_cut_fraction: float,
    diagonal_visible_depth_nn: float,
    diagonal_marker_size: float,
    full_box_marker_size: float,
    parallel_workers: int,
    ) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "GIF animation export requires Pillow. Install the project requirements before using "
            "visualization.write_animations=true."
        ) from exc

    state_index = {name: idx for idx, name in enumerate(state_names)}
    excluded_state_ids = {state_index[name] for name in ("L", "P") if name in state_index}
    if parallel_workers <= 0:
        raise ValueError(f"visualization.parallel_workers must be positive, got {parallel_workers}.")
    if frame_store.mode == "frame_dirs":
        if frame_store.frame_dirs is None or not frame_store.frame_dirs:
            raise ValueError("frame_dirs animation mode requires a non-empty frame directory list.")
        task_payloads = [
            {
                "frame_dir": str(frame_dir),
                "frame_idx": int(frame_dir.name.split("_")[-1]),
                "box_size": box_size,
                "avg_nn_distance": avg_nn_distance,
                "color_lookup": color_lookup,
                "max_atoms": max_atoms_per_frame,
                "mode": mode,
                "excluded_state_ids": excluded_state_ids,
                "diagonal_cut_fraction": diagonal_cut_fraction,
                "diagonal_visible_depth_nn": diagonal_visible_depth_nn,
                "diagonal_marker_size": diagonal_marker_size,
                "full_box_marker_size": full_box_marker_size,
            }
            for frame_dir in frame_store.frame_dirs
        ]
        if len(task_payloads) == 1:
            rendered_frames = [_render_animation_frame_from_disk(task_payloads[0])]
        else:
            worker_count = min(parallel_workers, len(task_payloads), 12)
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=_animation_process_context(),
            ) as executor:
                rendered_frames = list(executor.map(_render_animation_frame_from_disk, task_payloads))
    elif frame_store.mode == "single_chunk_npz":
        rendered_frames = []
        for frame_idx in range(frame_store.num_frames):
            atoms, atom_states = _load_frame_atoms_and_states(frame_store, frame_idx)
            rendered_frames.append(
                _render_animation_frame(
                    atoms=atoms,
                    atom_states=atom_states,
                    frame_idx=frame_idx,
                    box_size=box_size,
                    avg_nn_distance=avg_nn_distance,
                    color_lookup=color_lookup,
                    max_atoms=max_atoms_per_frame,
                    mode=mode,
                    excluded_state_ids=excluded_state_ids,
                    diagonal_cut_fraction=diagonal_cut_fraction,
                    diagonal_visible_depth_nn=diagonal_visible_depth_nn,
                    diagonal_marker_size=diagonal_marker_size,
                    full_box_marker_size=full_box_marker_size,
                )
            )
    else:
        raise ValueError(f"Unsupported frame store mode {frame_store.mode!r}.")
    images = [Image.fromarray(image) for image in rendered_frames]
    if not images:
        raise RuntimeError(f"Animation export created no frames for mode={mode!r}.")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(round(550)),
        loop=0,
    )


def _select_frame_indices(num_frames: int, max_frames_to_plot: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    if max_frames_to_plot <= 0:
        raise ValueError(f"max_frames_to_plot must be positive, got {max_frames_to_plot}.")
    if num_frames <= max_frames_to_plot:
        return list(range(num_frames))
    raw = np.linspace(0, num_frames - 1, num=max_frames_to_plot)
    return [int(round(item)) for item in raw]


def _select_site_indices(
    *,
    state_ids: np.ndarray,
    transition_mask: np.ndarray,
    max_sites: int,
) -> list[int]:
    if max_sites <= 0:
        raise ValueError(f"max_sites must be positive, got {max_sites}.")
    site_count = state_ids.shape[1]
    transition_counts = np.sum(transition_mask, axis=0)
    final_states = state_ids[-1]
    ranked = np.lexsort((np.arange(site_count), final_states, -transition_counts))
    selected: list[int] = []
    covered_states: set[int] = set()
    for site_id in ranked:
        final_state = int(final_states[site_id])
        if final_state in covered_states:
            continue
        selected.append(int(site_id))
        covered_states.add(final_state)
        if len(selected) >= min(max_sites, site_count):
            return selected
    for site_id in ranked:
        if int(site_id) in selected:
            continue
        selected.append(int(site_id))
        if len(selected) >= min(max_sites, site_count):
            break
    return selected


def _color_for_state_index(state_idx: int, color_lookup: dict[str, str]) -> str:
    state_names = list(color_lookup.keys())
    if state_idx < 0 or state_idx >= len(state_names):
        raise IndexError(
            f"State index {state_idx} is out of range for palette of size {len(state_names)}."
        )
    return color_lookup[state_names[state_idx]]


def _render_animation_frame(
    *,
    atoms: np.ndarray,
    atom_states: np.ndarray,
    frame_idx: int,
    box_size: float,
    avg_nn_distance: float,
    color_lookup: dict[str, str],
    max_atoms: int,
    mode: str,
    excluded_state_ids: set[int],
    diagonal_cut_fraction: float,
    diagonal_visible_depth_nn: float,
    diagonal_marker_size: float,
    full_box_marker_size: float,
) -> np.ndarray:
    if max_atoms <= 0:
        raise ValueError(f"Animation max_atoms must be positive, got {max_atoms}.")
    if mode == "diagonal_cut_all_phases":
        selected_atoms, selected_states = _select_diagonal_cut_atoms(
            atoms=atoms,
            atom_states=atom_states,
            frame_idx=frame_idx,
            max_atoms=max_atoms,
            box_size=box_size,
            diagonal_cut_fraction=diagonal_cut_fraction,
            avg_nn_distance=avg_nn_distance,
            visible_depth_nn=diagonal_visible_depth_nn,
        )
        title = f"Frame {frame_idx}: all phases, diagonal cut box"
        view_elev, view_azim = _view_from_vector(_DIAGONAL_DIRECTION)
        marker_size = diagonal_marker_size
    elif mode == "solid_only_full_box":
        mask = np.asarray([int(state_id) not in excluded_state_ids for state_id in atom_states], dtype=bool)
        filtered_atoms = atoms[mask]
        filtered_states = atom_states[mask]
        if filtered_atoms.shape[0] == 0:
            selected_atoms = np.zeros((0, 3), dtype=np.float32)
            selected_states = np.zeros(0, dtype=np.int16)
        else:
            selected_atoms, selected_states = _sample_atoms_for_animation(
                atoms=filtered_atoms,
                atom_states=filtered_states,
                frame_idx=frame_idx,
                max_atoms=max_atoms,
            )
        view_elev, view_azim = 20.0, 36.0
        title = f"Frame {frame_idx}: no L/P, full box"
        marker_size = full_box_marker_size
    else:
        raise ValueError(f"Unsupported animation mode {mode!r}.")

    fig = plt.figure(figsize=(6.4, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    if selected_atoms.shape[0] > 0:
        colors = [_color_for_state_index(int(state_idx), color_lookup) for state_idx in selected_states]
        ax.scatter(
            selected_atoms[:, 0],
            selected_atoms[:, 1],
            selected_atoms[:, 2],
            c=colors,
            s=marker_size,
            alpha=0.94 if mode == "diagonal_cut_all_phases" else 0.72,
            linewidths=0.0,
            depthshade=True,
        )
    else:
        ax.text2D(0.5, 0.5, "No atoms for this phase view", ha="center", va="center", transform=ax.transAxes)
    _draw_box_edges(ax=ax, box_size=box_size)
    if mode == "diagonal_cut_all_phases":
        _draw_diagonal_cut_plane(ax=ax, box_size=box_size)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(0.0, box_size)
    ax.set_ylim(0.0, box_size)
    ax.set_zlim(0.0, box_size)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.grid(False)
    ax.set_xticks([0.0, 0.5 * box_size, box_size])
    ax.set_yticks([0.0, 0.5 * box_size, box_size])
    ax.set_zticks([0.0, 0.5 * box_size, box_size])
    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return image[..., :3].copy()


def _render_animation_frame_from_disk(payload: dict[str, Any]) -> np.ndarray:
    frame_dir = Path(payload["frame_dir"])
    atoms = np.load(frame_dir / "atoms.npy")
    atom_table = np.load(frame_dir / "atom_table.npz")
    atom_states = atom_table["state_ids"].astype(np.int16)
    return _render_animation_frame(
        atoms=atoms,
        atom_states=atom_states,
        frame_idx=int(payload["frame_idx"]),
        box_size=float(payload["box_size"]),
        avg_nn_distance=float(payload["avg_nn_distance"]),
        color_lookup=dict(payload["color_lookup"]),
        max_atoms=int(payload["max_atoms"]),
        mode=str(payload["mode"]),
        excluded_state_ids=set(payload["excluded_state_ids"]),
        diagonal_cut_fraction=float(payload["diagonal_cut_fraction"]),
        diagonal_visible_depth_nn=float(payload["diagonal_visible_depth_nn"]),
        diagonal_marker_size=float(payload["diagonal_marker_size"]),
        full_box_marker_size=float(payload["full_box_marker_size"]),
    )


def _animation_process_context() -> mp.context.BaseContext:
    main_module = sys.modules.get("__main__")
    main_path = getattr(main_module, "__file__", None)
    if not main_path or str(main_path).startswith("<"):
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _select_diagonal_cut_atoms(
    *,
    atoms: np.ndarray,
    atom_states: np.ndarray,
    frame_idx: int,
    max_atoms: int,
    box_size: float,
    diagonal_cut_fraction: float,
    avg_nn_distance: float,
    visible_depth_nn: float,
) -> tuple[np.ndarray, np.ndarray]:
    if diagonal_cut_fraction <= 0.0 or diagonal_cut_fraction >= 3.0:
        raise ValueError(
            "animation_diagonal_cut_fraction must be in (0, 3), "
            f"got {diagonal_cut_fraction}."
        )
    if visible_depth_nn <= 0.0:
        raise ValueError(
            "animation_diagonal_visible_depth_nn must be positive, "
            f"got {visible_depth_nn}."
        )
    half_cube_mask = _diagonal_cut_half_cube_mask(atoms=atoms, box_size=box_size)
    selected_atoms = atoms[half_cube_mask]
    selected_states = atom_states[half_cube_mask]
    if selected_atoms.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int16)
    center = np.full(3, 0.5 * box_size, dtype=np.float32)
    signed_distance = (selected_atoms - center[None, :]) @ _DIAGONAL_DIRECTION
    visible_depth = float(visible_depth_nn) * float(avg_nn_distance)
    near_plane_mask = signed_distance >= -visible_depth
    if int(np.count_nonzero(near_plane_mask)) >= min(1200, max_atoms):
        selected_atoms = selected_atoms[near_plane_mask]
        selected_states = selected_states[near_plane_mask]
    return _sample_atoms_for_animation(
        atoms=selected_atoms,
        atom_states=selected_states,
        frame_idx=frame_idx,
        max_atoms=max_atoms,
    )


def _sample_atoms_for_animation(
    *,
    atoms: np.ndarray,
    atom_states: np.ndarray,
    frame_idx: int,
    max_atoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_atoms = atoms
    selected_states = atom_states
    if selected_atoms.shape[0] > max_atoms:
        rng = np.random.default_rng(104_729 + frame_idx)
        keep = rng.choice(selected_atoms.shape[0], size=max_atoms, replace=False)
        selected_atoms = selected_atoms[keep]
        selected_states = selected_states[keep]
    return selected_atoms.astype(np.float32), selected_states.astype(np.int16)


def _draw_box_edges(ax, *, box_size: float) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [box_size, 0.0, 0.0],
            [0.0, box_size, 0.0],
            [0.0, 0.0, box_size],
            [box_size, box_size, 0.0],
            [box_size, 0.0, box_size],
            [0.0, box_size, box_size],
            [box_size, box_size, box_size],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    for start, stop in edges:
        ax.plot(
            [vertices[start, 0], vertices[stop, 0]],
            [vertices[start, 1], vertices[stop, 1]],
            [vertices[start, 2], vertices[stop, 2]],
            color="#64748B",
            linewidth=0.8,
            alpha=0.35,
        )


def _draw_diagonal_cut_plane(ax, *, box_size: float) -> None:
    polygon = _plane_cube_intersection_polygon(
        box_size=box_size,
        cutoff=1.5 * box_size,
    )
    if polygon.shape[0] < 3:
        return
    poly = Poly3DCollection(
        [polygon],
        facecolor="#CBD5E1",
        edgecolor="#475569",
        linewidths=1.0,
        alpha=0.10,
    )
    ax.add_collection3d(poly)
    wrapped = np.vstack([polygon, polygon[0]])
    ax.plot(wrapped[:, 0], wrapped[:, 1], wrapped[:, 2], color="#475569", linewidth=1.1, alpha=0.55)


def _diagonal_cutoff(*, box_size: float, diagonal_cut_fraction: float) -> float:
    return float((3.0 - diagonal_cut_fraction) * box_size)


def _diagonal_cut_half_cube_mask(*, atoms: np.ndarray, box_size: float) -> np.ndarray:
    if atoms.size == 0:
        return np.zeros((atoms.shape[0],), dtype=bool)
    center = np.full(3, 0.5 * box_size, dtype=np.float32)
    relative = atoms - center[None, :]
    projection = relative @ _DIAGONAL_DIRECTION
    return projection <= 0.0


def _view_from_vector(direction: np.ndarray) -> tuple[float, float]:
    vector = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return 30.0, 45.0
    vector = vector / norm
    azimuth = float(np.degrees(np.arctan2(vector[1], vector[0])))
    elevation = float(np.degrees(np.arcsin(np.clip(vector[2], -1.0, 1.0))))
    return elevation, azimuth


def _plane_cube_intersection_polygon(*, box_size: float, cutoff: float) -> np.ndarray:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [box_size, 0.0, 0.0],
            [0.0, box_size, 0.0],
            [0.0, 0.0, box_size],
            [box_size, box_size, 0.0],
            [box_size, 0.0, box_size],
            [0.0, box_size, box_size],
            [box_size, box_size, box_size],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]

    points: list[np.ndarray] = []
    plane_values = np.sum(vertices, axis=1) - cutoff
    for start, stop in edges:
        start_value = float(plane_values[start])
        stop_value = float(plane_values[stop])
        start_point = vertices[start]
        stop_point = vertices[stop]
        if abs(start_value) < 1e-6:
            points.append(start_point)
        if abs(stop_value) < 1e-6:
            points.append(stop_point)
        if start_value * stop_value < 0.0:
            alpha = start_value / (start_value - stop_value)
            intersection = start_point + alpha * (stop_point - start_point)
            points.append(intersection.astype(np.float32))

    if not points:
        return np.zeros((0, 3), dtype=np.float32)

    unique_points = []
    for point in points:
        if not any(np.linalg.norm(point - existing) < 1e-5 for existing in unique_points):
            unique_points.append(point.astype(np.float32))
    if len(unique_points) < 3:
        return np.zeros((0, 3), dtype=np.float32)

    polygon = np.asarray(unique_points, dtype=np.float32)
    centroid = np.mean(polygon, axis=0)
    normal = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    normal = normal / np.linalg.norm(normal)
    tangent_a = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    tangent_a = tangent_a / np.linalg.norm(tangent_a)
    tangent_b = np.cross(normal, tangent_a).astype(np.float32)
    tangent_b = tangent_b / np.linalg.norm(tangent_b)
    rel = polygon - centroid[None, :]
    u_coord = rel @ tangent_a
    v_coord = rel @ tangent_b
    order = np.argsort(np.arctan2(v_coord, u_coord))
    return polygon[order]
