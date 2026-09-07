#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.data_utils.shooting_dataset import (
    build_periodic_environment_batch,
    resolve_shooting_trajectory_path,
)
from src.temporal_vamp.embeddings import FrozenEncoder, load_frozen_encoder
from src.temporal_vamp.geoframe_stability import (
    plot_distance_distributions,
    plot_example_trajectories,
    plot_sibling_divergence,
    plot_stability_summary,
    row_cosine,
    stratified_stability_rows,
    temporal_stability_table,
    write_csv,
)
from src.temporal_vamp.commands.common import (
    required,
    resolve_path,
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required GeoFrame stability JSON is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def _encode_chunks(
    encoder: FrozenEncoder,
    points: torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, points.shape[0], int(batch_size)):
            blocks.append(
                encoder.encode(points[start : start + int(batch_size)]).cpu().numpy()
            )
    return np.concatenate(blocks).astype(np.float32, copy=False)


def _neighbor_retention(current: np.ndarray, initial: np.ndarray) -> np.ndarray:
    return np.mean(
        np.any(current[:, :, None] == initial[:, None, :], axis=2), axis=1
    ).astype(np.float32)


def _initial_neighbor_rms(
    trajectory: ShootingBinaryTrajectory,
    frame_index: int,
    center_atom_ids: np.ndarray,
    initial_neighbor_atom_ids: np.ndarray,
    initial_relative_A: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(trajectory.positions[int(frame_index)], dtype=np.float32)
    box_lengths = (
        np.asarray(trajectory.box_high[int(frame_index)], dtype=np.float32)
        - np.asarray(trajectory.box_low[int(frame_index)], dtype=np.float32)
    )
    centers = positions[center_atom_ids - 1]
    current_relative = positions[initial_neighbor_atom_ids - 1] - centers[:, None, :]
    current_relative -= box_lengths[None, None, :] * np.round(
        current_relative / box_lengths[None, None, :]
    )
    displacement = current_relative - initial_relative_A
    displacement -= box_lengths[None, None, :] * np.round(
        displacement / box_lengths[None, None, :]
    )
    return np.sqrt(np.mean(np.sum(np.square(displacement), axis=-1), axis=-1)).astype(
        np.float32
    )


def _cross_state_distances(
    embeddings: np.ndarray,
    branch_parent: np.ndarray,
    parent_temperature_K: np.ndarray,
    parent_role: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    representative_branch = np.empty(parent_temperature_K.size, dtype=np.int64)
    for parent in range(parent_temperature_K.size):
        branches = np.flatnonzero(branch_parent == parent)
        if branches.size != 4:
            raise RuntimeError(
                f"Cross-state reference expected four branches for parent={parent}, "
                f"observed={branches.size}."
            )
        representative_branch[parent] = branches[0]
    pools = {
        float(temperature): np.flatnonzero(
            (parent_temperature_K == temperature)
            & (parent_role == "transition_candidate")
        )
        for temperature in np.unique(parent_temperature_K)
    }
    for temperature, parents in pools.items():
        if parents.size < 2:
            raise RuntimeError(
                f"Cross-state reference needs two transition parents at {temperature:g} K."
            )
    temperatures = np.asarray(sorted(pools), dtype=np.float64)
    distances = np.empty(int(samples), dtype=np.float32)
    cosines = np.empty(int(samples), dtype=np.float32)
    center_count = embeddings.shape[2]
    for index in range(int(samples)):
        temperature = float(rng.choice(temperatures))
        left_parent, right_parent = rng.choice(pools[temperature], size=2, replace=False)
        left_center = int(rng.integers(0, center_count))
        right_center = int(rng.integers(0, center_count))
        left = embeddings[representative_branch[left_parent], 0, left_center]
        right = embeddings[representative_branch[right_parent], 0, right_center]
        distances[index] = np.linalg.norm(left.astype(np.float64) - right.astype(np.float64))
        cosines[index] = row_cosine(left[None, :], right[None, :])[0]
    return distances, cosines


def _rotation_matrices(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    matrices = np.empty((int(count), 3, 3), dtype=np.float32)
    for index in range(int(count)):
        matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(matrix) < 0.0:
            matrix[:, 0] *= -1.0
        matrices[index] = matrix.astype(np.float32)
    return matrices


def _invariance_controls(
    encoder: FrozenEncoder,
    points: torch.Tensor,
    *,
    batch_size: int,
    cross_state_scale: float,
    seed: int,
) -> dict[str, Any]:
    original = _encode_chunks(encoder, points, batch_size=batch_size)
    repeated = _encode_chunks(encoder, points, batch_size=batch_size)
    numpy_points = points.numpy()
    rotations = _rotation_matrices(points.shape[0], seed)
    rotated_points = torch.from_numpy(
        np.einsum("npi,nij->npj", numpy_points, rotations).astype(np.float32)
    )
    rotated = _encode_chunks(encoder, rotated_points, batch_size=batch_size)
    rng = np.random.default_rng(int(seed) + 1)
    permutations = np.argsort(rng.random((points.shape[0], points.shape[1])), axis=1)
    permuted_points = torch.from_numpy(
        np.take_along_axis(numpy_points, permutations[:, :, None], axis=1).astype(
            np.float32
        )
    )
    permuted = _encode_chunks(encoder, permuted_points, batch_size=batch_size)

    output: dict[str, Any] = {"cloud_count": int(points.shape[0])}
    for name, transformed in (
        ("repeat", repeated),
        ("rotation", rotated),
        ("point_permutation", permuted),
    ):
        distance = np.linalg.norm(
            transformed.astype(np.float64) - original.astype(np.float64), axis=1
        )
        cosine = row_cosine(transformed, original)
        output[name] = {
            "distance_mean": float(np.mean(distance)),
            "distance_max": float(np.max(distance)),
            "distance_q95": float(np.quantile(distance, 0.95)),
            "distance_mean_relative_to_cross_state": float(
                np.mean(distance) / float(cross_state_scale)
            ),
            "cosine_mean": float(np.mean(cosine)),
            "cosine_min": float(np.min(cosine)),
            "max_absolute_component_difference": float(
                np.max(np.abs(transformed.astype(np.float64) - original.astype(np.float64)))
            ),
        }
    return output


def _write_readme(output_dir: Path, metrics: dict[str, Any]) -> None:
    finest = metrics["lag_metrics"][1]
    terminal = metrics["lag_metrics"][-1]
    controls = metrics["invariance_controls"]
    representation = str(metrics["encoder"]["representation_source"]).replace("_", " ")
    text = f"""# GeoFrame temporal stability at 30 fs resolution

This experiment follows identical atom IDs through the first 0.30 ps of all 144
completed branches in the transition-balanced nested shooting campaign. It uses
36 parents, four siblings per parent, 32 deterministic atom IDs, 11 exact frames
at 0.03 ps spacing, and the frozen GeoFrameV2 {representation} representation.

## Main results

| Quantity | 0.03 ps | 0.30 ps |
|---|---:|---:|
| Mean same-atom cosine | {finest['cosine_mean']:.6f} | {terminal['cosine_mean']:.6f} |
| Median drift / cross-state median | {finest['relative_to_cross_state_median']:.6f} | {terminal['relative_to_cross_state_median']:.6f} |
| Same-atom top-1 retrieval | {finest['same_atom_top1']:.4f} | {terminal['same_atom_top1']:.4f} |
| Initial-neighbor retention | {finest['neighbor_retention_mean']:.4f} | {terminal['neighbor_retention_mean']:.4f} |
| Atom-matched local RMS (A) | {finest['cloud_rms_A_mean']:.4f} | {terminal['cloud_rms_A_mean']:.4f} |

Deterministic repeat maximum embedding difference is
{controls['repeat']['distance_max']:.3e}. Mean rotation and input-permutation
changes are {controls['rotation']['distance_mean_relative_to_cross_state']:.3e}
and {controls['point_permutation']['distance_mean_relative_to_cross_state']:.3e}
of the median cross-state distance.

## Interpretation

The 30 fs result measures physical sensitivity plus neighborhood-selection
sensitivity; deterministic encoder noise is separately bounded by the repeat
control. Sibling trajectories share positions at time zero and diverge only after
their momentum/noise initialization, providing an additional physical control.

The local RMS follows the initial 160 atom IDs, whereas neighbor retention reports
changes in the actual nearest-160 set used by the encoder. Stability therefore is
not inferred from embedding distance alone.

## Files

- `metrics.json`: complete metrics and scientific contract.
- `lag_metrics.csv`: stability versus every 30 fs lag.
- `stratified_metrics.csv`: temperature/parent-role breakdown.
- `stability_arrays.npz`: embeddings and physical diagnostics.
- `plots/`: temporal stability, distance distributions, sibling divergence, and
  example PCA paths.
- `resolved_config.yaml`: exact run configuration.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure frozen GeoFrame stability at the finest stored shooting timestep."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = resolve_path(args.config)
    cfg: DictConfig = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)

    output_dir = resolve_path(required(cfg, "output_dir"))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite GeoFrame stability output: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "plots").mkdir()
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    campaign_root = resolve_path(required(cfg, "data.campaign_root"))
    manifest = _load_json(campaign_root / "manifest.json")
    if str(manifest["campaign_type"]) != "transition_balanced_nested_langevin_nvt_shooting_pilot":
        raise ValueError(
            "Finest-step experiment requires the original nested shooting producer, "
            f"observed={manifest['campaign_type']!r}."
        )
    if int(manifest["atom_count"]) != 70304:
        raise RuntimeError(
            f"Finest-step campaign atom count changed: {manifest['atom_count']}."
        )
    parents = sorted(manifest["parents"], key=lambda value: int(value["parent_index"]))
    branches = sorted(manifest["branches"], key=lambda value: int(value["branch_index"]))
    if len(parents) != 36 or len(branches) != 144:
        raise RuntimeError(
            f"Expected 36 parents and 144 branches, got {len(parents)} and {len(branches)}."
        )
    timesteps = np.asarray([int(value) for value in required(cfg, "data.timesteps")], dtype=np.int64)
    expected = np.arange(0, 101, 10, dtype=np.int64)
    if not np.array_equal(timesteps, expected):
        raise ValueError(
            f"Finest stored common schedule must be {expected.tolist()}, got {timesteps.tolist()}."
        )
    timestep_fs = 3.0
    times_ps = timesteps.astype(np.float64) * timestep_fs / 1000.0

    center_count = int(required(cfg, "data.center_atom_count"))
    center_seed = int(required(cfg, "data.center_selection_seed"))
    center_ids = np.sort(
        np.random.default_rng(center_seed).choice(
            np.arange(1, 70305, dtype=np.int64), size=center_count, replace=False
        )
    )
    num_points = int(required(cfg, "data.num_points"))
    radius = float(required(cfg, "data.radius_A"))
    device = str(required(cfg, "encoder.device"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"encoder.device={device!r} requests CUDA, but CUDA is unavailable.")
    encoder = load_frozen_encoder(
        resolve_path(required(cfg, "encoder.checkpoint")),
        device=device,
        repeats=int(required(cfg, "encoder.repeats")),
        seed=int(required(cfg, "encoder.seed")),
        representation_source=str(required(cfg, "encoder.representation_source")),
    )
    embedding_dim = encoder.output_dim
    batch_size = int(required(cfg, "encoder.batch_size"))

    branch_parent = np.asarray([int(value["parent_index"]) for value in branches], dtype=np.int64)
    if not np.array_equal(np.bincount(branch_parent), np.full(36, 4, dtype=np.int64)):
        raise RuntimeError(
            f"Expected four branches per parent, got {np.bincount(branch_parent).tolist()}."
        )
    parent_temperature = np.asarray([float(value["temperature_K"]) for value in parents])
    parent_role = np.asarray([str(value["basin_role"]) for value in parents])
    branch_temperature = parent_temperature[branch_parent]
    branch_role = parent_role[branch_parent]

    embeddings = np.empty(
        (len(branches), timesteps.size, center_count, embedding_dim), dtype=np.float32
    )
    retention = np.empty((len(branches), timesteps.size, center_count), dtype=np.float32)
    cloud_rms = np.empty_like(retention)
    control_count = int(required(cfg, "evaluation.control_cloud_count"))
    total_clouds = len(branches) * timesteps.size * center_count
    control_global = np.sort(
        np.random.default_rng(int(required(cfg, "evaluation.seed"))).choice(
            total_clouds, size=control_count, replace=False
        )
    )
    control_points: list[torch.Tensor] = []
    started = time.perf_counter()
    for branch_position, branch in enumerate(branches):
        trajectory_path = resolve_shooting_trajectory_path(campaign_root, branch)
        trajectory = ShootingBinaryTrajectory.load(trajectory_path)
        available = set(int(value) for value in trajectory.timesteps.tolist())
        missing = [int(value) for value in timesteps.tolist() if int(value) not in available]
        if missing:
            raise RuntimeError(
                f"Branch {branch['branch_id']} lacks finest common timesteps {missing}: {trajectory_path}."
            )
        frames = trajectory.load_position_frames(timesteps.tolist())
        environments = [
            build_periodic_environment_batch(
                frames[int(timestep)],
                center_atom_ids=center_ids,
                num_points=num_points,
                radius=radius,
                spatial_context_center_count=0,
            )
            for timestep in timesteps.tolist()
        ]
        branch_points = torch.stack([value.points for value in environments], dim=0)
        encoded = _encode_chunks(
            encoder, branch_points.reshape(-1, num_points, 3), batch_size=batch_size
        )
        embeddings[branch_position] = encoded.reshape(timesteps.size, center_count, embedding_dim)
        initial_neighbors = environments[0].neighbor_atom_ids
        initial_relative = environments[0].points.numpy() * np.float32(radius)
        timestep_to_frame = {
            int(value): index for index, value in enumerate(trajectory.timesteps.tolist())
        }
        for time_index, timestep in enumerate(timesteps.tolist()):
            retention[branch_position, time_index] = _neighbor_retention(
                environments[time_index].neighbor_atom_ids, initial_neighbors
            )
            cloud_rms[branch_position, time_index] = _initial_neighbor_rms(
                trajectory,
                timestep_to_frame[int(timestep)],
                center_ids,
                initial_neighbors,
                initial_relative,
            )
        branch_start = branch_position * timesteps.size * center_count
        local_control = control_global[
            (control_global >= branch_start)
            & (control_global < branch_start + timesteps.size * center_count)
        ] - branch_start
        if local_control.size:
            control_points.append(
                branch_points.reshape(-1, num_points, 3)[
                    torch.from_numpy(local_control.astype(np.int64))
                ].clone()
            )
        if (branch_position + 1) % 12 == 0 or branch_position + 1 == len(branches):
            print(
                f"[geoframe-stability] encoded {branch_position + 1}/{len(branches)} "
                f"branches elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    selected_control_points = torch.cat(control_points, dim=0)
    if selected_control_points.shape[0] != control_count:
        raise RuntimeError(
            f"Control cloud collection failed: expected={control_count}, "
            f"observed={selected_control_points.shape[0]}."
        )

    cross_distances, cross_cosines = _cross_state_distances(
        embeddings,
        branch_parent,
        parent_temperature,
        parent_role,
        samples=int(required(cfg, "evaluation.cross_state_samples")),
        seed=int(required(cfg, "evaluation.seed")),
    )
    cross_scale = float(np.median(cross_distances))
    lag_rows, temporal_distances, sibling = temporal_stability_table(
        embeddings,
        times_ps,
        retention,
        cloud_rms,
        branch_parent,
        cross_distances,
        seed=int(required(cfg, "evaluation.seed")),
    )
    strata = stratified_stability_rows(
        temporal_distances,
        embeddings,
        retention,
        cloud_rms,
        times_ps,
        branch_temperature,
        branch_role,
        cross_scale,
    )
    controls = _invariance_controls(
        encoder,
        selected_control_points,
        batch_size=batch_size,
        cross_state_scale=cross_scale,
        seed=int(required(cfg, "evaluation.seed")),
    )

    plot_stability_summary(lag_rows, output_dir / "plots" / "stability_vs_time.png")
    plot_distance_distributions(
        temporal_distances,
        cross_distances,
        times_ps,
        output_dir / "plots" / "distance_distributions.png",
    )
    plot_sibling_divergence(
        lag_rows, cross_scale, output_dir / "plots" / "sibling_divergence.png"
    )
    examples = plot_example_trajectories(
        embeddings,
        times_ps,
        branch_parent,
        parent_temperature,
        parent_role,
        center_ids,
        output_dir / "plots" / "example_embedding_trajectories.png",
    )
    write_csv(output_dir / "lag_metrics.csv", lag_rows)
    write_csv(output_dir / "stratified_metrics.csv", strata)
    np.savez_compressed(
        output_dir / "stability_arrays.npz",
        embeddings=embeddings,
        temporal_distances=temporal_distances,
        neighbor_retention=retention,
        cloud_rms_A=cloud_rms,
        sibling_distance_mean=sibling,
        cross_state_distances=cross_distances,
        cross_state_cosines=cross_cosines,
        timesteps=timesteps,
        times_ps=times_ps,
        center_atom_ids=center_ids,
        branch_parent_index=branch_parent,
        branch_temperature_K=branch_temperature,
        branch_role=branch_role.astype("U32"),
    )
    metrics: dict[str, Any] = {
        "scientific_contract": {
            "question": "stability of the same-atom frozen GeoFrameV2 representation under the finest stored physical evolution",
            "smallest_stored_interval_ps": 0.03,
            "interval_basis": "10 LAMMPS steps times 3 fs/step",
            "encoder_noise_control": "exact repeated deterministic encoding of identical point clouds",
            "rotation_control": "independent proper rotation of each normalized local cloud",
            "permutation_control": "independent permutation of the 160 input points",
            "physical_control": "atom-matched local RMS and nearest-160 membership retention",
            "cross_state_reference": "different transition parents at identical temperature, one representative branch per parent",
        },
        "data": {
            "campaign_root": str(campaign_root),
            "parents": len(parents),
            "branches": len(branches),
            "branches_per_parent": 4,
            "center_atom_count": center_count,
            "encoded_environment_count": int(embeddings.shape[0] * embeddings.shape[1] * embeddings.shape[2]),
            "timesteps": timesteps.tolist(),
            "times_ps": times_ps.tolist(),
            "temperatures_K": sorted(np.unique(parent_temperature).tolist()),
            "parent_roles": {
                role: int(np.sum(parent_role == role)) for role in np.unique(parent_role)
            },
            "num_points": num_points,
            "radius_A": radius,
        },
        "encoder": {
            "checkpoint": str(encoder.checkpoint_path),
            "representation_source": encoder.representation_source,
            "embedding_dim": embedding_dim,
            "deterministic_fps": encoder.deterministic,
            "device": device,
        },
        "cross_state_reference": {
            "samples": int(cross_distances.size),
            "distance_mean": float(np.mean(cross_distances)),
            "distance_median": cross_scale,
            "distance_q05": float(np.quantile(cross_distances, 0.05)),
            "distance_q95": float(np.quantile(cross_distances, 0.95)),
            "cosine_mean": float(np.mean(cross_cosines)),
        },
        "invariance_controls": controls,
        "lag_metrics": lag_rows,
        "stratified_metrics": strata,
        "example_trajectories": examples,
        "runtime_seconds": float(time.perf_counter() - started),
        "host": os.uname().nodename,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_readme(output_dir, metrics)
    print(
        f"[geoframe-stability] complete output={output_dir} "
        f"runtime={metrics['runtime_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
