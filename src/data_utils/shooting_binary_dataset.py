"""Parallel local-environment loading from binary shooting trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.data_utils.shooting_dataset import (
    ShootingCampaignSnapshot,
    ShootingPositionFrame,
    build_periodic_environment_batch,
    resolve_shooting_trajectory_path,
)


def propagate_ballistic_positions(
    positions: np.ndarray,
    velocities: np.ndarray,
    box_lengths: np.ndarray,
    horizon_ps: float,
) -> np.ndarray:
    """Force-free LAMMPS-metal position update with periodic wrapping."""

    propagated = np.mod(
        positions + velocities * np.float32(horizon_ps), box_lengths[None, :]
    ).astype(np.float32, copy=False)
    return np.minimum(
        propagated,
        np.nextafter(box_lengths, np.zeros_like(box_lengths))[None, :],
    )


class ShootingBinaryEnvironmentDataset(Dataset[dict[str, Any]]):
    """Build fixed-center periodic point clouds from validated binary memory maps.

    One dataset item is one shooting branch at all requested timesteps. Opening the
    memory maps and constructing the periodic neighbor trees happens in ``__getitem__``
    so a PyTorch ``DataLoader`` can overlap this CPU work with GPU encoding.
    Velocities are deliberately not read: the frozen point-cloud encoder consumes
    positions only, while ``ShootingBinaryTrajectory`` remains the explicit API for
    consumers that need the stored velocity field.
    """

    def __init__(
        self,
        snapshot: ShootingCampaignSnapshot,
        *,
        branches: Sequence[dict[str, Any]],
        timesteps: Sequence[int],
        center_atom_ids: np.ndarray,
        num_points: int,
        radius: float,
        spatial_context_center_count: int,
    ) -> None:
        self.snapshot = snapshot
        self.branches = tuple(branches)
        if not self.branches:
            raise ValueError("Shooting binary environment dataset requires at least one branch.")
        self.timesteps = tuple(int(value) for value in timesteps)
        if not self.timesteps or tuple(sorted(set(self.timesteps))) != self.timesteps:
            raise ValueError(
                "Shooting environment timesteps must be nonempty, unique, and increasing; "
                f"got {self.timesteps}."
            )
        self.center_atom_ids = np.asarray(center_atom_ids, dtype=np.int64)
        atom_count = int(snapshot.manifest["atom_count"])
        if (
            self.center_atom_ids.ndim != 1
            or self.center_atom_ids.size == 0
            or not np.array_equal(
                self.center_atom_ids, np.unique(self.center_atom_ids)
            )
            or int(self.center_atom_ids[0]) < 1
            or int(self.center_atom_ids[-1]) > atom_count
        ):
            raise ValueError(
                "center_atom_ids must be a nonempty sorted unique int64 array within "
                f"[1, {atom_count}], got shape={self.center_atom_ids.shape}."
            )
        self.num_points = int(num_points)
        self.radius = float(radius)
        self.spatial_context_center_count = int(spatial_context_center_count)
        if self.num_points <= 0 or self.num_points > atom_count:
            raise ValueError(
                f"num_points must be in [1, {atom_count}], got {self.num_points}."
            )
        if self.radius <= 0.0:
            raise ValueError(f"radius must be positive, got {self.radius}.")
        if not 0 <= self.spatial_context_center_count < self.num_points:
            raise ValueError(
                "spatial_context_center_count must be in [0, num_points); "
                f"got {self.spatial_context_center_count} and {self.num_points}."
            )

        paths: list[Path] = []
        for branch in self.branches:
            root = (
                snapshot.root
                if len(snapshot.campaign_roots) == 1
                else Path(str(branch["campaign_root"]))
            )
            path = resolve_shooting_trajectory_path(root, branch)
            if not path.is_dir():
                raise RuntimeError(
                    "Shooting training requires a completed binary trajectory artifact. "
                    f"branch={branch['branch_id']}, resolved_path={path}. Run "
                    "scripts/migrate_lammps_shooting_float32.py for this campaign first."
                )
            paths.append(path)
        self.trajectory_paths = tuple(paths)
        parent_by_id = {
            str(parent["parent_id"]): parent for parent in snapshot.parents
        }
        self.parents = tuple(
            parent_by_id[str(branch["parent_id"])] for branch in self.branches
        )

    def __len__(self) -> int:
        return len(self.branches)

    def __getitem__(self, index: int) -> dict[str, Any]:
        branch = self.branches[index]
        parent = self.parents[index]
        trajectory = ShootingBinaryTrajectory.load(self.trajectory_paths[index])
        expected_atom_count = int(self.snapshot.manifest["atom_count"])
        if trajectory.atom_count != expected_atom_count:
            raise RuntimeError(
                f"Shooting branch atom count changed: branch={branch['branch_id']}, "
                f"expected={expected_atom_count}, observed={trajectory.atom_count}."
            )
        frames = trajectory.load_position_frames(self.timesteps)
        environments = [
            build_periodic_environment_batch(
                frames[timestep],
                center_atom_ids=self.center_atom_ids,
                num_points=self.num_points,
                radius=self.radius,
                spatial_context_center_count=self.spatial_context_center_count,
            )
            for timestep in self.timesteps
        ]
        if len(self.snapshot.campaign_roots) == 1:
            campaign_root = self.snapshot.root
            campaign_index = 0
            branch_uid = str(branch["branch_id"])
        else:
            campaign_root = Path(str(branch["campaign_root"]))
            campaign_index = int(branch["campaign_index"])
            branch_uid = str(branch["branch_uid"])
        timestep_ps = float(self.snapshot.manifest["protocol"]["timestep_fs"]) / 1000.0
        relative_times_ps = np.asarray(self.timesteps, dtype=np.float64) * timestep_ps
        absolute_times_ps = relative_times_ps + float(parent["source_frame_time_ps"])
        sample: dict[str, Any] = {
            "dataset_index": index,
            "branch_id": str(branch["branch_id"]),
            "branch_uid": branch_uid,
            "branch_index": int(branch["branch_index"]),
            "campaign_root": str(campaign_root),
            "campaign_index": campaign_index,
            "trajectory_path": str(self.trajectory_paths[index]),
            "parent_id": str(branch["parent_id"]),
            "parent_index": int(branch["parent_index"]),
            "source_index": int(parent["source_index"]),
            "source_run_id": str(branch["source_run_id"]),
            "source_split": str(branch["source_split"]),
            "source_velocity_seed": int(branch["source_velocity_seed"]),
            "temperature_K": float(branch["temperature_K"]),
            "phase": str(branch["phase"]),
            "shot_index": int(branch["shot_index"]),
            "velocity_seed": int(branch["velocity_seed"]),
            "thermostat_seed": int(branch["thermostat_seed"]),
            "nucleation_time_ps": float(parent["nucleation_time_ps"]),
            "parent_offset_ps": float(parent["parent_offset_ps"]),
            "source_frame_index": int(parent["source_frame_index"]),
            "source_frame_step": int(parent["source_frame_step"]),
            "source_frame_time_ps": float(parent["source_frame_time_ps"]),
            "source_crystalline_fraction": float(
                parent["source_crystalline_fraction"]
            ),
            "source_largest_crystalline_cluster_atoms": int(
                parent["source_largest_crystalline_cluster_atoms"]
            ),
            "parent_data_sha256": str(parent["data_sha256"]),
            "parent_data_file": str(parent["data_file"]),
            "timesteps": torch.tensor(self.timesteps, dtype=torch.int64),
            "relative_times_ps": torch.from_numpy(relative_times_ps),
            "absolute_times_ps": torch.from_numpy(absolute_times_ps),
            "box_low": torch.from_numpy(
                np.stack([frames[value].box_low for value in self.timesteps], axis=0)
            ),
            "box_high": torch.from_numpy(
                np.stack([frames[value].box_high for value in self.timesteps], axis=0)
            ),
            "box_lengths": torch.from_numpy(
                np.stack([frames[value].box_lengths for value in self.timesteps], axis=0)
            ),
            "atom_ids": torch.from_numpy(self.center_atom_ids),
            "atom_types": torch.from_numpy(
                np.asarray(trajectory.atom_types[self.center_atom_ids - 1])
            ),
            "points": torch.stack([value.points for value in environments], dim=0),
            "center_positions": torch.from_numpy(
                np.stack([value.center_positions for value in environments], axis=0)
            ),
        }
        if self.spatial_context_center_count > 0:
            context_points = [value.context_points for value in environments]
            context_offsets = [value.context_center_offsets for value in environments]
            context_atom_ids = [value.context_center_atom_ids for value in environments]
            if (
                any(value is None for value in context_points)
                or any(value is None for value in context_offsets)
                or any(value is None for value in context_atom_ids)
            ):
                raise RuntimeError(
                    f"Context construction returned missing arrays for branch={branch['branch_id']}."
                )
            sample["context_points"] = torch.stack(context_points, dim=0)  # type: ignore[arg-type]
            sample["context_center_offsets"] = torch.from_numpy(
                np.stack(context_offsets, axis=0)  # type: ignore[arg-type]
            )
            sample["context_center_atom_ids"] = torch.from_numpy(
                np.stack(context_atom_ids, axis=0)  # type: ignore[arg-type]
            )
        return sample


class ShootingBallisticEnvironmentDataset(ShootingBinaryEnvironmentDataset):
    """Build local clouds after a force-free rollout of the time-zero velocities."""

    def __init__(
        self,
        snapshot: ShootingCampaignSnapshot,
        *,
        branches: Sequence[dict[str, Any]],
        horizons_ps: Sequence[float],
        center_atom_ids: np.ndarray,
        num_points: int,
        radius: float,
    ) -> None:
        super().__init__(
            snapshot,
            branches=branches,
            timesteps=[0],
            center_atom_ids=center_atom_ids,
            num_points=num_points,
            radius=radius,
            spatial_context_center_count=0,
        )
        self.horizons_ps = tuple(float(value) for value in horizons_ps)
        if (
            not self.horizons_ps
            or any(value <= 0.0 for value in self.horizons_ps)
            or tuple(sorted(set(self.horizons_ps))) != self.horizons_ps
        ):
            raise ValueError(
                "Ballistic horizons must be positive, unique, and increasing; "
                f"got {self.horizons_ps}."
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory = ShootingBinaryTrajectory.load(self.trajectory_paths[index])
        frame = trajectory.load_frames([0])[0]
        positions = np.asarray(frame.positions, dtype=np.float32)
        velocities = np.asarray(frame.velocities, dtype=np.float32)
        box_lengths = np.asarray(frame.box_lengths, dtype=np.float32)
        environments = []
        for horizon in self.horizons_ps:
            propagated = propagate_ballistic_positions(
                positions, velocities, box_lengths, horizon
            )
            ballistic_frame = ShootingPositionFrame(
                timestep=0,
                atom_ids=frame.atom_ids,
                atom_types=frame.atom_types,
                positions=propagated,
                box_low=frame.box_low,
                box_high=frame.box_high,
            )
            environments.append(
                build_periodic_environment_batch(
                    ballistic_frame,
                    center_atom_ids=self.center_atom_ids,
                    num_points=self.num_points,
                    radius=self.radius,
                    spatial_context_center_count=0,
                )
            )
        return {
            "dataset_index": index,
            "points": torch.stack([value.points for value in environments], dim=0),
        }


def make_shooting_environment_loader(
    dataset: ShootingBinaryEnvironmentDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[dict[str, Any]]:
    """Create an ordered loader with worker-side memory mapping and prefetch."""

    resolved_batch_size = int(batch_size)
    resolved_workers = int(num_workers)
    if resolved_batch_size <= 0 or resolved_workers < 0:
        raise ValueError(
            "Shooting environment loader requires batch_size>0 and num_workers>=0; "
            f"got batch_size={resolved_batch_size}, num_workers={resolved_workers}."
        )
    kwargs: dict[str, Any] = {}
    if resolved_workers > 0:
        kwargs.update(
            prefetch_factor=2,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=False,
        num_workers=resolved_workers,
        pin_memory=bool(pin_memory),
        **kwargs,
    )


__all__ = [
    "ShootingBallisticEnvironmentDataset",
    "ShootingBinaryEnvironmentDataset",
    "make_shooting_environment_loader",
    "propagate_ballistic_positions",
]
