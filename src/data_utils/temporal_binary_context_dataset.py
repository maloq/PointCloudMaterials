"""Multi-horizon local environments from repository temporal float32 binaries.

This loader is deliberately specific to the completed 70,304-atom simulation
catalog.  One item is an anchor frame containing every configured center atom,
its spatial-context tokens, and the same central atoms at all requested future
horizons.  The raw trajectories remain memory mapped in CPU worker processes.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.baselines.descriptor_baselines import SteinhardtDescriptorBaseline
from src.data_utils.shooting_dataset import (
    ShootingPositionFrame,
    build_periodic_environment_batch,
)
from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory
from src.temporal_vamp.simulation_catalog import CatalogEntry


@dataclass(frozen=True)
class TemporalBinaryAnchor:
    run_index: int
    anchor_frame: int


class TemporalBinaryContextDataset(Dataset[dict[str, object]]):
    """Produce present context and multiple future environments per anchor.

    Point clouds are returned in the encoder's normalized units.  Context
    offsets remain in Angstrom, matching the shooting context-token cache.
    """

    def __init__(
        self,
        entries: Sequence[CatalogEntry],
        *,
        center_atom_ids: Sequence[int],
        horizons_ps: Sequence[float],
        anchor_stride_frames: int,
        num_points: int,
        radius: float,
        context_center_count: int,
        steinhardt_shell_min_neighbors: int,
        steinhardt_shell_max_neighbors: int,
        trajectory_cache_size: int = 2,
    ) -> None:
        super().__init__()
        self.entries = tuple(entries)
        self.center_atom_ids = np.asarray(center_atom_ids, dtype=np.int64)
        self.horizons_ps = np.asarray(horizons_ps, dtype=np.float64)
        self.anchor_stride_frames = int(anchor_stride_frames)
        self.num_points = int(num_points)
        self.radius = float(radius)
        self.context_center_count = int(context_center_count)
        self.trajectory_cache_size = int(trajectory_cache_size)
        if not self.entries:
            raise ValueError("Temporal binary context dataset requires at least one run.")
        if self.center_atom_ids.ndim != 1 or self.center_atom_ids.size == 0:
            raise ValueError("center_atom_ids must be a nonempty one-dimensional sequence.")
        if np.unique(self.center_atom_ids).size != self.center_atom_ids.size:
            raise ValueError("center_atom_ids contains duplicates.")
        if np.any(self.horizons_ps <= 0.0) or np.any(np.diff(self.horizons_ps) <= 0.0):
            raise ValueError(
                f"horizons_ps must be positive and strictly increasing, got "
                f"{self.horizons_ps.tolist()}."
            )
        if self.anchor_stride_frames <= 0:
            raise ValueError(
                f"anchor_stride_frames must be positive, got {self.anchor_stride_frames}."
            )
        if self.num_points <= 0 or self.radius <= 0.0:
            raise ValueError(
                f"num_points and radius must be positive, got {self.num_points}, {self.radius}."
            )
        if not 0 <= self.context_center_count < self.num_points:
            raise ValueError(
                "context_center_count must be in [0, num_points), got "
                f"{self.context_center_count} and num_points={self.num_points}."
            )
        if self.trajectory_cache_size <= 0:
            raise ValueError(
                f"trajectory_cache_size must be positive, got {self.trajectory_cache_size}."
            )

        self._lag_frames: list[np.ndarray] = []
        records: list[TemporalBinaryAnchor] = []
        for run_index, entry in enumerate(self.entries):
            if not entry.trajectory_path.is_dir():
                raise ValueError(
                    "Ablation 5 accepts only migrated temporal float32 binaries; "
                    f"run={entry.run_id}, path={entry.trajectory_path}."
                )
            trajectory = TemporalLAMMPSBinaryTrajectory.load(entry.trajectory_path)
            if trajectory.atom_count != entry.metadata.atom_count:
                raise RuntimeError(
                    f"Binary/catalog atom-count mismatch for run={entry.run_id}: "
                    f"binary={trajectory.atom_count}, catalog={entry.metadata.atom_count}."
                )
            interval_ps = float(entry.metadata.sample_interval_ps)
            raw_lags = self.horizons_ps / interval_ps
            lag_frames = np.rint(raw_lags).astype(np.int64)
            if not np.allclose(raw_lags, lag_frames, atol=1.0e-8, rtol=0.0):
                raise ValueError(
                    f"Requested horizons do not align with frames for run={entry.run_id}: "
                    f"horizons_ps={self.horizons_ps.tolist()}, interval_ps={interval_ps}."
                )
            self._lag_frames.append(lag_frames)
            final_anchor = trajectory.frame_count - int(lag_frames[-1])
            if final_anchor <= 0:
                raise ValueError(
                    f"Run={entry.run_id} is too short for horizon={self.horizons_ps[-1]} ps."
                )
            records.extend(
                TemporalBinaryAnchor(run_index=run_index, anchor_frame=anchor)
                for anchor in range(0, final_anchor, self.anchor_stride_frames)
            )
        self.records = tuple(records)
        self._trajectory_cache: OrderedDict[int, TemporalLAMMPSBinaryTrajectory] = (
            OrderedDict()
        )
        self._descriptor = SteinhardtDescriptorBaseline(
            l_values=[4, 6],
            center_atom_tolerance=1.0e-6,
            shell_min_neighbors=int(steinhardt_shell_min_neighbors),
            shell_max_neighbors=int(steinhardt_shell_max_neighbors),
            append_shell_size=True,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _trajectory(self, run_index: int) -> TemporalLAMMPSBinaryTrajectory:
        cached = self._trajectory_cache.pop(int(run_index), None)
        if cached is None:
            cached = TemporalLAMMPSBinaryTrajectory.load(
                self.entries[int(run_index)].trajectory_path
            )
        self._trajectory_cache[int(run_index)] = cached
        while len(self._trajectory_cache) > self.trajectory_cache_size:
            self._trajectory_cache.popitem(last=False)
        return cached

    @staticmethod
    def _frame(
        trajectory: TemporalLAMMPSBinaryTrajectory, frame_index: int
    ) -> ShootingPositionFrame:
        return ShootingPositionFrame(
            timestep=int(trajectory.timesteps[int(frame_index)]),
            atom_ids=trajectory.atom_ids,
            atom_types=trajectory.atom_types,
            positions=trajectory.positions[int(frame_index)],
            box_low=trajectory.box_low[int(frame_index)],
            box_high=trajectory.box_high[int(frame_index)],
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[int(index)]
        entry = self.entries[record.run_index]
        trajectory = self._trajectory(record.run_index)
        present = build_periodic_environment_batch(
            self._frame(trajectory, record.anchor_frame),
            center_atom_ids=self.center_atom_ids,
            num_points=self.num_points,
            radius=self.radius,
            spatial_context_center_count=self.context_center_count,
        )
        if self.context_center_count == 0:
            token_points = present.points[:, None]
        else:
            if present.context_points is None or present.context_center_offsets is None:
                raise RuntimeError("Present environment builder did not return context tokens.")
            token_points = torch.cat(
                [present.points[:, None], present.context_points], dim=1
            )
        token_count = self.context_center_count + 1
        descriptors = self._descriptor.transform(
            token_points.reshape(-1, self.num_points, 3).numpy()
        ).reshape(self.center_atom_ids.size, token_count, 3)
        offsets = np.zeros(
            (self.center_atom_ids.size, token_count, 3), dtype=np.float32
        )
        if self.context_center_count > 0:
            assert present.context_center_offsets is not None
            offsets[:, 1:] = present.context_center_offsets
        future_points: list[torch.Tensor] = []
        future_frames: list[int] = []
        future_timesteps: list[int] = []
        for lag_frames in self._lag_frames[record.run_index]:
            frame_index = record.anchor_frame + int(lag_frames)
            future = build_periodic_environment_batch(
                self._frame(trajectory, frame_index),
                center_atom_ids=self.center_atom_ids,
                num_points=self.num_points,
                radius=self.radius,
                spatial_context_center_count=0,
            )
            future_points.append(future.points)
            future_frames.append(frame_index)
            future_timesteps.append(int(trajectory.timesteps[frame_index]))
        return {
            "dataset_index": np.int64(index),
            "run_index": np.int32(record.run_index),
            "anchor_frame": np.int64(record.anchor_frame),
            "future_frames": np.asarray(future_frames, dtype=np.int64),
            "anchor_timestep": np.int64(trajectory.timesteps[record.anchor_frame]),
            "future_timesteps": np.asarray(future_timesteps, dtype=np.int64),
            "temperature_K": np.float32(entry.metadata.temperature_K),
            "velocity_seed": np.int64(entry.metadata.velocity_seed),
            "token_points": token_points,
            "token_descriptors": torch.from_numpy(descriptors),
            "token_offsets": torch.from_numpy(offsets),
            "future_points": torch.stack(future_points, dim=0),
        }


def make_temporal_binary_context_loader(
    dataset: TemporalBinaryContextDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    if int(batch_size) <= 0 or int(num_workers) < 0:
        raise ValueError(
            f"Invalid DataLoader batch_size={batch_size}, num_workers={num_workers}."
        )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0,
        multiprocessing_context="spawn" if int(num_workers) > 0 else None,
    )


__all__ = [
    "TemporalBinaryAnchor",
    "TemporalBinaryContextDataset",
    "make_temporal_binary_context_loader",
]
