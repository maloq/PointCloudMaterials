from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.data_utils.shooting_dataset import resolve_shooting_trajectory_path
from src.temporal_vamp.embeddings import FrozenEncoder
from src.temporal_vamp.evaluation import CovariancePCA
from src.temporal_vamp.predictability_map import FittedDenseProbe
from src.temporal_vamp.shooting_embeddings import ShootingEmbeddingCache


@dataclass(frozen=True)
class ShootingGifExample:
    parent_position: int
    parent_id: str
    temperature_K: float
    center_position: int
    atom_id: int
    terminal_embedding_dispersion: float
    branch_positions: np.ndarray


def select_transition_gif_examples(
    cache: ShootingEmbeddingCache,
    *,
    temperatures_K: Sequence[float],
) -> list[ShootingGifExample]:
    parents = cache.manifest["snapshot"]["parents"]
    branch_parent = np.asarray(cache.branch_parent_index, dtype=np.int64)
    examples: list[ShootingGifExample] = []
    for raw_temperature in temperatures_K:
        temperature = float(raw_temperature)
        candidates = [
            position
            for position, parent in enumerate(parents)
            if float(parent["temperature_K"]) == temperature
            and str(parent["source_split"]) == "final_validation"
        ]
        if not candidates:
            raise RuntimeError(
                f"No final-validation shooting parent exists at {temperature:g} K."
            )
        candidate_scores: list[tuple[float, int, int]] = []
        for parent_position in candidates:
            branches = np.flatnonzero(branch_parent == parent_position)
            terminal = np.asarray(cache.future_z[branches, -1], dtype=np.float32)
            per_center = np.mean(
                np.sum(
                    np.square(terminal - terminal.mean(axis=0, keepdims=True)),
                    axis=-1,
                ),
                axis=0,
            )
            center_position = int(np.argmax(per_center))
            candidate_scores.append(
                (float(per_center[center_position]), parent_position, center_position)
            )
        score, parent_position, center_position = max(candidate_scores)
        branches = np.flatnonzero(branch_parent == parent_position)
        examples.append(
            ShootingGifExample(
                parent_position=int(parent_position),
                parent_id=str(parents[parent_position]["parent_id"]),
                temperature_K=temperature,
                center_position=int(center_position),
                atom_id=int(cache.atom_ids[center_position]),
                terminal_embedding_dispersion=float(score),
                branch_positions=branches,
            )
        )
    return examples


def _source_outcome(branch: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(branch["source_outcome_path"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Nested shooting source outcome is missing for GIF branch "
            f"{branch['branch_id']}: {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        outcome = json.load(handle)
    if outcome.get("state") != "complete":
        raise RuntimeError(
            f"GIF source branch is not complete: branch={branch['branch_id']}, "
            f"state={outcome.get('state')!r}, outcome={path}."
        )
    return outcome


def _local_cloud(
    positions: np.ndarray,
    box_lengths: np.ndarray,
    *,
    atom_id: int,
    num_points: int,
) -> np.ndarray:
    center = np.asarray(positions[int(atom_id) - 1], dtype=np.float32)
    delta = np.asarray(positions, dtype=np.float32) - center[None, :]
    delta -= box_lengths[None, :] * np.round(delta / box_lengths[None, :])
    distance_squared = np.sum(np.square(delta), axis=1)
    selected = np.argpartition(distance_squared, int(num_points) - 1)[: int(num_points)]
    selected = selected[np.argsort(distance_squared[selected], kind="stable")]
    return delta[selected].astype(np.float32, copy=False)


def extract_example_clouds_and_embeddings(
    cache: ShootingEmbeddingCache,
    example: ShootingGifExample,
    *,
    encoder: FrozenEncoder,
    frame_stride: int,
    point_batch_size: int,
) -> dict[str, Any]:
    snapshot = cache.manifest["snapshot"]
    branches = snapshot["branches"]
    campaign_root = Path(str(snapshot["campaign_root"])).expanduser().resolve()
    num_points = int(cache.manifest["spec"]["num_points"])
    radius = float(cache.manifest["spec"]["radius"])
    branch_records = [branches[int(position)] for position in example.branch_positions]
    trajectories = [
        ShootingBinaryTrajectory.load(
            resolve_shooting_trajectory_path(campaign_root, branch)
        )
        for branch in branch_records
    ]
    first_timesteps = np.asarray(trajectories[0].timesteps, dtype=np.int64)
    for trajectory in trajectories[1:]:
        if not np.array_equal(trajectory.timesteps, first_timesteps):
            raise RuntimeError(
                f"Sibling GIF branches have different timesteps for parent={example.parent_id}."
            )
    stride = int(frame_stride)
    if stride <= 0:
        raise ValueError(f"GIF frame_stride must be positive, got {stride}.")
    frame_indices = np.arange(0, first_timesteps.size, stride, dtype=np.int64)
    if frame_indices[-1] != first_timesteps.size - 1:
        frame_indices = np.concatenate(
            [frame_indices, np.asarray([first_timesteps.size - 1], dtype=np.int64)]
        )
    selected_timesteps = first_timesteps[frame_indices]
    branch_clouds = np.empty(
        (len(trajectories), frame_indices.size, num_points, 3), dtype=np.float32
    )
    for branch_index, trajectory in enumerate(trajectories):
        for local_frame, source_frame in enumerate(frame_indices.tolist()):
            positions = np.asarray(trajectory.positions[source_frame], dtype=np.float32)
            box_lengths = (
                np.asarray(trajectory.box_high[source_frame], dtype=np.float32)
                - np.asarray(trajectory.box_low[source_frame], dtype=np.float32)
            )
            branch_clouds[branch_index, local_frame] = _local_cloud(
                positions,
                box_lengths,
                atom_id=example.atom_id,
                num_points=num_points,
            )
    normalized = torch.from_numpy(
        (branch_clouds / np.float32(radius)).reshape(-1, num_points, 3)
    )
    blocks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, normalized.shape[0], int(point_batch_size)):
            blocks.append(
                encoder.encode(normalized[start : start + int(point_batch_size)])
                .cpu()
                .numpy()
            )
    embeddings = np.concatenate(blocks).reshape(
        len(trajectories), frame_indices.size, -1
    ).astype(np.float32, copy=False)
    timestep_ps = float(snapshot["branches"][0]["timestep_fs"]) / 1000.0
    relative_times_ps = selected_timesteps.astype(np.float64) * timestep_ps
    outcomes = [_source_outcome(branch) for branch in branch_records]
    return {
        "local_clouds_A": branch_clouds,
        "embeddings": embeddings,
        "timesteps": selected_timesteps,
        "relative_times_ps": relative_times_ps,
        "branches": branch_records,
        "outcomes": outcomes,
        "radius_A": radius,
    }


def embedding_trajectory_coordinates(
    cache: ShootingEmbeddingCache,
    extracted: dict[str, Any],
    local_probe: FittedDenseProbe,
    *,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    parents = cache.manifest["snapshot"]["parents"]
    parent_count, center_count = cache.parent_local_z.shape[:2]
    parent_temperature = np.asarray(
        [float(parent["temperature_K"]) for parent in parents], dtype=np.float32
    )
    parent_local = np.asarray(cache.parent_local_z, dtype=np.float32).reshape(-1, 128)
    parent_inputs = np.concatenate(
        [np.repeat(parent_temperature, center_count)[:, None], parent_local], axis=1
    )
    parent_latent, _ = local_probe.transform(
        parent_inputs, device=device, batch_size=int(batch_size)
    )
    optimization_parents = np.asarray(
        [
            index
            for index, parent in enumerate(parents)
            if str(parent["source_split"]) == "optimization"
        ],
        dtype=np.int64,
    )
    optimization_rows = np.concatenate(
        [
            np.arange(index * center_count, (index + 1) * center_count, dtype=np.int64)
            for index in optimization_parents.tolist()
        ]
    )
    embedding_pca = CovariancePCA.fit(parent_local[optimization_rows], dimension=2)
    latent_pca = CovariancePCA.fit(parent_latent[optimization_rows], dimension=2)
    embeddings = np.asarray(extracted["embeddings"], dtype=np.float32)
    branch_count, frame_count, embedding_dim = embeddings.shape
    temperature = float(extracted["branches"][0]["temperature_K"])
    trajectory_inputs = np.concatenate(
        [
            np.full((branch_count * frame_count, 1), temperature, dtype=np.float32),
            embeddings.reshape(-1, embedding_dim),
        ],
        axis=1,
    )
    trajectory_latent, _ = local_probe.transform(
        trajectory_inputs, device=device, batch_size=int(batch_size)
    )
    return {
        "parent_embedding_2d": embedding_pca.transform(parent_local, dimension=2).astype(
            np.float32
        ),
        "trajectory_embedding_2d": embedding_pca.transform(
            embeddings.reshape(-1, embedding_dim), dimension=2
        )
        .reshape(branch_count, frame_count, 2)
        .astype(np.float32),
        "parent_predictive_2d": latent_pca.transform(parent_latent, dimension=2).astype(
            np.float32
        ),
        "trajectory_predictive_2d": latent_pca.transform(
            trajectory_latent, dimension=2
        )
        .reshape(branch_count, frame_count, 2)
        .astype(np.float32),
    }


def _outcome_label(outcome: dict[str, Any]) -> str:
    label = str(outcome["first_passage_outcome"])
    if bool(outcome["censored"]):
        return "censored"
    time_ps = float(outcome["first_passage_time_ps"])
    if label == "basin_A_liquid":
        return f"liquid @ {time_ps:g} ps"
    if label == "basin_B_crystal":
        return f"crystal @ {time_ps:g} ps"
    raise ValueError(f"Unexpected completed nested-shooting outcome={label!r}.")


def render_structure_gif(
    example: ShootingGifExample,
    extracted: dict[str, Any],
    path: str | Path,
    *,
    fps: int,
) -> None:
    clouds = np.asarray(extracted["local_clouds_A"], dtype=np.float32)
    times = np.asarray(extracted["relative_times_ps"], dtype=np.float64)
    outcomes = extracted["outcomes"]
    radius = float(extracted["radius_A"])
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), sharex=True, sharey=True)
    scatters = []
    for branch_index, axis in enumerate(axes.flat):
        points = clouds[branch_index, 0]
        scatter = axis.scatter(
            points[:, 0],
            points[:, 1],
            c=points[:, 2],
            cmap="coolwarm",
            vmin=-radius,
            vmax=radius,
            s=18,
            alpha=0.82,
            edgecolors="none",
        )
        axis.scatter([0.0], [0.0], marker="*", s=110, color="black", zorder=5)
        axis.set_xlim(-radius, radius)
        axis.set_ylim(-radius, radius)
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
        axis.set_title(f"shot {branch_index}: {_outcome_label(outcomes[branch_index])}")
        axis.set_xlabel("x relative to atom (Å)")
        axis.set_ylabel("y relative to atom (Å)")
        scatters.append(scatter)
    title = figure.suptitle("")
    colorbar = figure.colorbar(scatters[0], ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("z relative to atom (Å)")

    def update(frame: int) -> tuple[Any, ...]:
        for branch_index, scatter in enumerate(scatters):
            points = clouds[branch_index, frame]
            scatter.set_offsets(points[:, :2])
            scatter.set_array(points[:, 2])
        title.set_text(
            f"Local shooting evolution: {example.temperature_K:g} K, "
            f"atom {example.atom_id}, t={times[frame]:.1f} ps"
        )
        return (*scatters, title)

    animation = FuncAnimation(
        figure, update, frames=times.size, interval=1000.0 / int(fps), blit=False
    )
    animation.save(Path(path), writer=PillowWriter(fps=int(fps)), dpi=105)
    plt.close(figure)


def render_embedding_gif(
    example: ShootingGifExample,
    extracted: dict[str, Any],
    coordinates: dict[str, np.ndarray],
    path: str | Path,
    *,
    fps: int,
) -> None:
    times = np.asarray(extracted["relative_times_ps"], dtype=np.float64)
    outcomes = extracted["outcomes"]
    parent_spaces = [
        coordinates["parent_embedding_2d"],
        coordinates["parent_predictive_2d"],
    ]
    trajectories = [
        coordinates["trajectory_embedding_2d"],
        coordinates["trajectory_predictive_2d"],
    ]
    titles = ["Frozen GeoFrame PCA", "Predictive latent PCA"]
    colors = plt.cm.tab10(np.arange(4))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    line_groups: list[list[Any]] = []
    marker_groups: list[list[Any]] = []
    for axis, background, trajectory, axis_title in zip(
        axes, parent_spaces, trajectories, titles, strict=True
    ):
        axis.scatter(background[:, 0], background[:, 1], s=4, color="0.75", alpha=0.20)
        combined = np.concatenate([background, trajectory.reshape(-1, 2)], axis=0)
        low = np.quantile(combined, 0.005, axis=0)
        high = np.quantile(combined, 0.995, axis=0)
        margin = 0.08 * np.maximum(high - low, 1.0)
        axis.set_xlim(low[0] - margin[0], high[0] + margin[0])
        axis.set_ylim(low[1] - margin[1], high[1] + margin[1])
        axis.set_title(axis_title)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.18)
        lines: list[Any] = []
        markers: list[Any] = []
        for branch_index in range(trajectory.shape[0]):
            label = f"shot {branch_index}: {_outcome_label(outcomes[branch_index])}"
            line, = axis.plot([], [], color=colors[branch_index], linewidth=1.5, alpha=0.82, label=label)
            marker, = axis.plot([], [], "o", color=colors[branch_index], markersize=6)
            lines.append(line)
            markers.append(marker)
        axis.legend(frameon=False, fontsize=7, loc="best")
        line_groups.append(lines)
        marker_groups.append(markers)
    title = figure.suptitle("")

    def update(frame: int) -> tuple[Any, ...]:
        artists: list[Any] = []
        for trajectory, lines, markers in zip(
            trajectories, line_groups, marker_groups, strict=True
        ):
            for branch_index, (line, marker) in enumerate(zip(lines, markers, strict=True)):
                trail = trajectory[branch_index, : frame + 1]
                line.set_data(trail[:, 0], trail[:, 1])
                marker.set_data([trail[-1, 0]], [trail[-1, 1]])
                artists.extend([line, marker])
        title.set_text(
            f"Shooting embedding trajectories: {example.temperature_K:g} K, "
            f"atom {example.atom_id}, t={times[frame]:.1f} ps"
        )
        artists.append(title)
        return tuple(artists)

    animation = FuncAnimation(
        figure, update, frames=times.size, interval=1000.0 / int(fps), blit=False
    )
    animation.save(Path(path), writer=PillowWriter(fps=int(fps)), dpi=110)
    plt.close(figure)


__all__ = [
    "ShootingGifExample",
    "embedding_trajectory_coordinates",
    "extract_example_clouds_and_embeddings",
    "render_embedding_gif",
    "render_structure_gif",
    "select_transition_gif_examples",
]
