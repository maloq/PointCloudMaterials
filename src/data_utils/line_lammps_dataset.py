from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import get_worker_info

from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset, logger


class LineLAMMPSDataset(TemporalLAMMPSDumpDataset):
    """Sample line-ordered local structures from cached LAMMPS frames.

    Each item chooses a tracked anchor atom, draws a random 3D line through it,
    selects ``line_atoms`` atoms closest to that line from a local candidate
    pool, optionally enforces a minimum center separation, sorts them by their
    coordinate along the line, and returns the local neighborhood around every
    selected atom.
    """

    def __init__(
        self,
        *args,
        line_atoms: int,
        line_candidate_atoms: int,
        line_min_separation_radius_factor: float = 0.0,
        line_anchor_views_enabled: bool = False,
        line_anchor_view_min_radius_factor: float = 0.0,
        line_anchor_view_max_radius_factor: float | None = None,
        line_anchor_view_selection: str = "closest",
        line_seed: int = 0,
        deterministic_lines: bool = False,
        exact_line_diagnostic_samples: int = 0,
        **kwargs,
    ) -> None:
        self.line_atoms = int(line_atoms)
        self.line_candidate_atoms = int(line_candidate_atoms)
        self.line_min_separation_radius_factor = float(line_min_separation_radius_factor)
        self.line_anchor_views_enabled = bool(line_anchor_views_enabled)
        self.line_anchor_view_min_radius_factor = float(line_anchor_view_min_radius_factor)
        self.line_anchor_view_max_radius_factor = (
            self.line_min_separation_radius_factor
            if line_anchor_view_max_radius_factor is None
            else float(line_anchor_view_max_radius_factor)
        )
        self.line_anchor_view_selection = str(line_anchor_view_selection).strip().lower()
        self.line_seed = int(line_seed)
        self.deterministic_lines = bool(deterministic_lines)
        self.exact_line_diagnostic_samples = int(exact_line_diagnostic_samples)
        self._worker_rngs: dict[int, np.random.Generator] = {}

        if self.line_atoms <= 1 or self.line_atoms % 2 != 1:
            raise ValueError(
                "line_atoms must be an odd integer > 1 so the masked target atom is unambiguous. "
                f"Got line_atoms={self.line_atoms}."
            )
        if self.line_candidate_atoms < self.line_atoms:
            raise ValueError(
                "line_candidate_atoms must be >= line_atoms. "
                f"Got line_candidate_atoms={self.line_candidate_atoms}, line_atoms={self.line_atoms}."
            )
        if self.line_min_separation_radius_factor < 0.0:
            raise ValueError(
                "line_min_separation_radius_factor must be >= 0. "
                f"Got {self.line_min_separation_radius_factor}."
            )
        if self.line_anchor_views_enabled and self.line_min_separation_radius_factor <= 0.0:
            raise ValueError(
                "line_anchor_views_enabled requires line_min_separation_radius_factor > 0 "
                "so near-anchor views are bounded inside the excluded prediction-context radius."
            )
        if self.line_anchor_view_min_radius_factor < 0.0:
            raise ValueError(
                "line_anchor_view_min_radius_factor must be >= 0. "
                f"Got {self.line_anchor_view_min_radius_factor}."
            )
        if self.line_anchor_views_enabled and self.line_anchor_view_max_radius_factor <= 0.0:
            raise ValueError(
                "line_anchor_view_max_radius_factor must be > 0. "
                f"Got {self.line_anchor_view_max_radius_factor}."
            )
        if self.line_anchor_views_enabled and (
            self.line_anchor_view_min_radius_factor >= self.line_anchor_view_max_radius_factor
        ):
            raise ValueError(
                "line_anchor_view_min_radius_factor must be smaller than "
                "line_anchor_view_max_radius_factor when anchor views are enabled. "
                f"Got min={self.line_anchor_view_min_radius_factor}, "
                f"max={self.line_anchor_view_max_radius_factor}."
            )
        if self.line_anchor_view_selection not in {"closest", "outer"}:
            raise ValueError(
                "line_anchor_view_selection must be 'closest' or 'outer', "
                f"got {line_anchor_view_selection!r}."
            )

        super().__init__(*args, sequence_length=1, precompute_neighbor_indices=False, **kwargs)
        if self.line_candidate_atoms > self.num_atoms:
            raise ValueError(
                "line_candidate_atoms cannot exceed the number of atoms in the dump. "
                f"Got line_candidate_atoms={self.line_candidate_atoms}, num_atoms={self.num_atoms}, "
                f"source_path={self.dump_file}."
            )
        if self.exact_line_diagnostic_samples > 0:
            self._run_exact_line_diagnostic(self.exact_line_diagnostic_samples)

    @property
    def target_line_index(self) -> int:
        return self.line_atoms // 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        batch = self._build_line_batch_from_indices(np.asarray([index], dtype=np.int64))
        sample = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                sample[key] = value[0]
            elif isinstance(value, list):
                sample[key] = value[0]
            else:
                sample[key] = value
        return sample

    def __getitems__(self, indices: Sequence[int]) -> dict[str, Any]:
        index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        return self._build_line_batch_from_indices(index_array)

    def _rng_for_index(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.line_seed + int(index) * 104_729)

    def _rng_for_worker(self) -> np.random.Generator:
        worker = get_worker_info()
        if worker is None:
            key = -1
            seed = self.line_seed
        else:
            key = int(worker.id)
            seed = int(worker.seed) + self.line_seed
        rng = self._worker_rngs.get(key)
        if rng is None:
            rng = np.random.default_rng(seed)
            self._worker_rngs[key] = rng
        return rng

    def _sample_directions(self, count: int, *, rng: np.random.Generator) -> np.ndarray:
        directions = rng.normal(size=(int(count), 3)).astype(np.float32)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise RuntimeError(
                "Random line direction sampler produced a zero-length direction. "
                f"count={count}, source_path={self.dump_file}."
            )
        return directions / norms

    @staticmethod
    def _minimum_image_delta(delta: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
        box_shape = (1,) * (delta.ndim - 1) + (3,)
        box = box_lengths.reshape(box_shape)
        return delta - box * np.round(delta / box)

    def _select_line_atoms_from_candidates(
        self,
        *,
        frame_points: np.ndarray,
        box_lengths: np.ndarray,
        anchor_positions: np.ndarray,
        anchor_indices: np.ndarray,
        candidate_indices: np.ndarray,
        directions: np.ndarray,
        frame_idx_info: str = "unknown",
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        candidate_points = np.asarray(frame_points[candidate_indices], dtype=np.float32)
        delta = candidate_points - anchor_positions[:, None, :]
        delta = self._minimum_image_delta(delta, box_lengths.astype(np.float32, copy=False))

        line_t = np.sum(delta * directions[:, None, :], axis=-1)
        dist2 = np.sum(delta * delta, axis=-1)
        perp2 = np.maximum(dist2 - line_t * line_t, 0.0)

        selected_indices = np.empty((candidate_indices.shape[0], self.line_atoms), dtype=np.int64)
        selected_t = np.empty((candidate_indices.shape[0], self.line_atoms), dtype=np.float32)
        selected_perp = np.empty((candidate_indices.shape[0], self.line_atoms), dtype=np.float32)
        anchor_view_indices = (
            np.empty((candidate_indices.shape[0], 2), dtype=np.int64)
            if self.line_anchor_views_enabled
            else None
        )
        anchor_view_t = (
            np.empty((candidate_indices.shape[0], 2), dtype=np.float32)
            if self.line_anchor_views_enabled
            else None
        )
        anchor_view_perp = (
            np.empty((candidate_indices.shape[0], 2), dtype=np.float32)
            if self.line_anchor_views_enabled
            else None
        )
        for row_idx in range(candidate_indices.shape[0]):
            if self.line_min_separation_radius_factor > 0.0:
                closest_slots = self._select_separated_line_slots(
                    frame_idx_info=frame_idx_info,
                    row_idx=row_idx,
                    box_lengths=box_lengths,
                    candidate_indices=candidate_indices,
                    candidate_points=candidate_points,
                    anchor_indices=anchor_indices,
                    line_t=line_t,
                    perp2=perp2,
                )
            else:
                closest_slots = np.argpartition(perp2[row_idx], self.line_atoms - 1)[: self.line_atoms]
            order = np.argsort(line_t[row_idx, closest_slots], kind="mergesort")
            ordered_slots = closest_slots[order]
            selected_indices[row_idx] = candidate_indices[row_idx, ordered_slots]
            selected_t[row_idx] = line_t[row_idx, ordered_slots].astype(np.float32, copy=False)
            selected_perp[row_idx] = np.sqrt(perp2[row_idx, ordered_slots]).astype(np.float32, copy=False)
            if self.line_anchor_views_enabled:
                anchor_slots = self._select_anchor_view_slots(
                    frame_idx_info=frame_idx_info,
                    row_idx=row_idx,
                    candidate_indices=candidate_indices,
                    anchor_indices=anchor_indices,
                    dist2=dist2,
                    line_t=line_t,
                    perp2=perp2,
                )
                if anchor_view_indices is None or anchor_view_t is None or anchor_view_perp is None:
                    raise RuntimeError("LAMMPS line anchor-view arrays were not initialized.")
                anchor_view_indices[row_idx] = candidate_indices[row_idx, anchor_slots]
                anchor_view_t[row_idx] = line_t[row_idx, anchor_slots].astype(np.float32, copy=False)
                anchor_view_perp[row_idx] = np.sqrt(perp2[row_idx, anchor_slots]).astype(
                    np.float32,
                    copy=False,
                )
        return (
            selected_indices,
            selected_t,
            selected_perp,
            anchor_view_indices,
            anchor_view_t,
            anchor_view_perp,
        )

    def _select_anchor_view_slots(
        self,
        *,
        frame_idx_info: str,
        row_idx: int,
        candidate_indices: np.ndarray,
        anchor_indices: np.ndarray,
        dist2: np.ndarray,
        line_t: np.ndarray,
        perp2: np.ndarray,
    ) -> np.ndarray:
        max_distance = float(self.radius) * self.line_anchor_view_max_radius_factor
        max_dist2 = max_distance * max_distance
        min_distance = float(self.radius) * self.line_anchor_view_min_radius_factor
        min_dist2 = min_distance * min_distance
        anchor_index = int(anchor_indices[row_idx])
        negative = self._select_anchor_view_side(
            frame_idx_info=frame_idx_info,
            row_idx=row_idx,
            side_name="negative",
            candidate_indices=candidate_indices,
            anchor_index=anchor_index,
            dist2=dist2,
            line_t=line_t,
            perp2=perp2,
            min_dist2=min_dist2,
            max_dist2=max_dist2,
            side_mask=line_t[row_idx] < 0.0,
        )
        positive = self._select_anchor_view_side(
            frame_idx_info=frame_idx_info,
            row_idx=row_idx,
            side_name="positive",
            candidate_indices=candidate_indices,
            anchor_index=anchor_index,
            dist2=dist2,
            line_t=line_t,
            perp2=perp2,
            min_dist2=min_dist2,
            max_dist2=max_dist2,
            side_mask=line_t[row_idx] > 0.0,
        )
        return np.asarray([negative, positive], dtype=np.int64)

    def _select_anchor_view_side(
        self,
        *,
        frame_idx_info: str,
        row_idx: int,
        side_name: str,
        candidate_indices: np.ndarray,
        anchor_index: int,
        dist2: np.ndarray,
        line_t: np.ndarray,
        perp2: np.ndarray,
        min_dist2: float,
        max_dist2: float,
        side_mask: np.ndarray,
    ) -> int:
        allowed = (
            side_mask
            & (candidate_indices[row_idx] != int(anchor_index))
            & (dist2[row_idx] >= float(min_dist2))
            & (dist2[row_idx] < float(max_dist2))
        )
        side_slots = np.flatnonzero(allowed)
        if side_slots.size == 0:
            raise RuntimeError(
                "LAMMPS anchor-view line selection could not find a near-line atom "
                f"on the {side_name} side of the target. frame={frame_idx_info}, row_idx={row_idx}, "
                f"line_candidate_atoms={self.line_candidate_atoms}, "
                f"min_center_distance={float(min_dist2) ** 0.5:.6f}, "
                f"max_center_distance={float(max_dist2) ** 0.5:.6f}, radius={float(self.radius):.6f}, "
                f"source_path={self.dump_file}. Increase data.line_candidate_atoms or "
                "reduce data.line_anchor_view_min_radius_factor, or increase "
                "data.line_anchor_view_max_radius_factor."
            )
        if self.line_anchor_view_selection == "closest":
            ordered = side_slots[
                np.lexsort((np.abs(line_t[row_idx, side_slots]), perp2[row_idx, side_slots]))
            ]
        elif self.line_anchor_view_selection == "outer":
            line_ordered = side_slots[np.argsort(perp2[row_idx, side_slots], kind="mergesort")]
            line_pool = line_ordered[: min(64, line_ordered.size)]
            ordered = line_pool[
                np.lexsort((perp2[row_idx, line_pool], -dist2[row_idx, line_pool]))
            ]
        else:
            raise RuntimeError(
                "Unsupported line_anchor_view_selection at runtime: "
                f"{self.line_anchor_view_selection!r}."
            )
        return int(ordered[0])

    def _select_separated_line_slots(
        self,
        *,
        frame_idx_info: str,
        row_idx: int,
        box_lengths: np.ndarray,
        candidate_indices: np.ndarray,
        candidate_points: np.ndarray,
        anchor_indices: np.ndarray,
        line_t: np.ndarray,
        perp2: np.ndarray,
    ) -> np.ndarray:
        min_distance = float(self.radius) * self.line_min_separation_radius_factor
        anchor_matches = np.flatnonzero(candidate_indices[row_idx] == int(anchor_indices[row_idx]))
        if anchor_matches.size != 1:
            raise RuntimeError(
                "LAMMPS separated line selection expected the anchor atom to appear exactly once "
                "in the candidate pool. "
                f"frame={frame_idx_info}, row_idx={row_idx}, matches={anchor_matches.size}, "
                f"anchor_index={int(anchor_indices[row_idx])}, "
                f"line_candidate_atoms={self.line_candidate_atoms}, source_path={self.dump_file}."
            )

        anchor_slot = int(anchor_matches[0])
        selected_slots: list[int] = [anchor_slot]
        selected_points = [candidate_points[row_idx, anchor_slot]]
        side_count = self.line_atoms // 2
        negative_slots = self._select_separated_line_side(
            frame_idx_info=frame_idx_info,
            row_idx=row_idx,
            side_name="negative",
            box_lengths=box_lengths,
            candidate_points=candidate_points,
            line_t=line_t,
            perp2=perp2,
            selected_slots=selected_slots,
            selected_points=selected_points,
            min_distance=min_distance,
            required=side_count,
            side_mask=line_t[row_idx] < 0.0,
        )
        positive_slots = self._select_separated_line_side(
            frame_idx_info=frame_idx_info,
            row_idx=row_idx,
            side_name="positive",
            box_lengths=box_lengths,
            candidate_points=candidate_points,
            line_t=line_t,
            perp2=perp2,
            selected_slots=selected_slots,
            selected_points=selected_points,
            min_distance=min_distance,
            required=side_count,
            side_mask=line_t[row_idx] > 0.0,
        )
        return np.asarray(negative_slots + [anchor_slot] + positive_slots, dtype=np.int64)

    def _select_separated_line_side(
        self,
        *,
        frame_idx_info: str,
        row_idx: int,
        side_name: str,
        box_lengths: np.ndarray,
        candidate_points: np.ndarray,
        line_t: np.ndarray,
        perp2: np.ndarray,
        selected_slots: list[int],
        selected_points: list[np.ndarray],
        min_distance: float,
        required: int,
        side_mask: np.ndarray,
    ) -> list[int]:
        side_slots = np.flatnonzero(side_mask)
        side_slots = side_slots[
            np.lexsort((np.abs(line_t[row_idx, side_slots]), perp2[row_idx, side_slots]))
        ]
        chosen: list[int] = []
        for slot_raw in side_slots.tolist():
            slot = int(slot_raw)
            if slot in selected_slots:
                continue
            point = candidate_points[row_idx, slot]
            delta = point[None, :] - np.asarray(selected_points)
            distances = np.linalg.norm(self._minimum_image_delta(delta, box_lengths), axis=1)
            if np.any(distances < min_distance):
                continue
            selected_slots.append(slot)
            selected_points.append(point)
            chosen.append(slot)
            if len(chosen) == required:
                return chosen
        raise RuntimeError(
            "LAMMPS separated line selection could not place enough non-overlapping centers "
            f"on the {side_name} side of the target. frame={frame_idx_info}, row_idx={row_idx}, "
            f"required_side_centers={required}, selected_side_centers={len(chosen)}, "
            f"line_atoms={self.line_atoms}, line_candidate_atoms={self.line_candidate_atoms}, "
            f"min_center_distance={min_distance:.6f}, radius={float(self.radius):.6f}, "
            f"source_path={self.dump_file}. Increase data.line_candidate_atoms or reduce "
            "data.line_min_separation_radius_factor."
        )

    def _build_line_batch_from_indices(self, indices: np.ndarray) -> dict[str, Any]:
        index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        if index_array.size == 0:
            raise ValueError("Line-JEPA batch indices must be non-empty.")
        if np.any(index_array < 0) or np.any(index_array >= len(self)):
            raise IndexError(
                "Line-JEPA batch indices are out of range. "
                f"min_index={int(index_array.min())}, max_index={int(index_array.max())}, len={len(self)}."
            )

        batch_size = int(index_array.size)
        line_points = np.empty((batch_size, self.line_atoms, self.num_points, 3), dtype=np.float32)
        line_atom_ids = np.empty((batch_size, self.line_atoms), dtype=np.int64)
        line_t = np.empty((batch_size, self.line_atoms), dtype=np.float32)
        line_perp = np.empty((batch_size, self.line_atoms), dtype=np.float32)
        line_directions = np.empty((batch_size, 3), dtype=np.float32)
        anchor_view_points = (
            np.empty((batch_size, 2, self.num_points, 3), dtype=np.float32)
            if self.line_anchor_views_enabled
            else None
        )
        anchor_view_atom_ids = (
            np.empty((batch_size, 2), dtype=np.int64)
            if self.line_anchor_views_enabled
            else None
        )
        anchor_view_t = (
            np.empty((batch_size, 2), dtype=np.float32)
            if self.line_anchor_views_enabled
            else None
        )
        anchor_view_perp = (
            np.empty((batch_size, 2), dtype=np.float32)
            if self.line_anchor_views_enabled
            else None
        )
        target_atom_ids = np.empty((batch_size,), dtype=np.int64)
        anchor_atom_ids = np.empty((batch_size,), dtype=np.int64)
        frame_indices_batch = np.empty((batch_size,), dtype=np.int64)
        timesteps_batch = np.empty((batch_size,), dtype=np.int64)
        target_positions = np.empty((batch_size, 3), dtype=np.float32)
        anchor_positions_batch = np.empty((batch_size, 3), dtype=np.float32)

        window_slots = (index_array // self.center_count).astype(np.int64, copy=False)
        center_slots = (index_array % self.center_count).astype(np.int64, copy=False)

        grouped_positions: dict[int, list[int]] = {}
        for batch_pos, window_slot in enumerate(window_slots.tolist()):
            grouped_positions.setdefault(int(window_slot), []).append(int(batch_pos))

        worker_rng = None if self.deterministic_lines else self._rng_for_worker()
        for window_slot, batch_positions_list in grouped_positions.items():
            batch_positions = np.asarray(batch_positions_list, dtype=np.int64)
            frame_idx = int(self._window_start_frames[window_slot])
            frame_points = np.asarray(self.positions[frame_idx], dtype=np.float32)
            box_lengths = np.asarray(self.box_lengths[frame_idx], dtype=np.float32)
            center_atom_indices = np.asarray(
                self._center_atom_indices[center_slots[batch_positions]],
                dtype=np.int64,
            )
            anchors = np.asarray(frame_points[center_atom_indices], dtype=np.float32)
            if self.deterministic_lines:
                directions = np.concatenate(
                    [
                        self._sample_directions(1, rng=self._rng_for_index(int(index_array[pos])))
                        for pos in batch_positions.tolist()
                    ],
                    axis=0,
                )
            else:
                if worker_rng is None:
                    raise RuntimeError(
                        "Line-JEPA worker RNG was not initialized for stochastic line sampling. "
                        f"deterministic_lines={self.deterministic_lines}, source_path={self.dump_file}."
                    )
                directions = self._sample_directions(len(batch_positions), rng=worker_rng)

            tree = self._get_tree(frame_idx)
            _, candidates = tree.query(anchors, k=self.line_candidate_atoms)
            candidates = np.asarray(candidates, dtype=np.int64)
            if candidates.ndim == 1:
                candidates = candidates.reshape(1, -1)
            if candidates.shape != (len(batch_positions), self.line_candidate_atoms):
                raise RuntimeError(
                    "Line candidate KDTree query returned an unexpected shape. "
                    f"frame_idx={frame_idx}, expected_shape={(len(batch_positions), self.line_candidate_atoms)}, "
                    f"got_shape={tuple(candidates.shape)}, source_path={self.dump_file}."
                )

            (
                selected,
                selected_t,
                selected_perp,
                selected_anchor_views,
                selected_anchor_view_t,
                selected_anchor_view_perp,
            ) = self._select_line_atoms_from_candidates(
                frame_points=frame_points,
                box_lengths=box_lengths,
                anchor_positions=anchors,
                anchor_indices=center_atom_indices,
                candidate_indices=candidates,
                directions=directions,
                frame_idx_info=str(frame_idx),
            )
            flat_centers = np.asarray(frame_points[selected.reshape(-1)], dtype=np.float32)
            local_indices = self._query_local_structures(frame_idx=frame_idx, centers=flat_centers)
            local_points = np.asarray(frame_points[local_indices], dtype=np.float32)
            local_points = self._to_local_coordinates_batch(
                frame_idx=frame_idx,
                points=local_points,
                centers=flat_centers,
            )
            if self.normalize:
                local_points = self._normalize_point_cloud_batch(local_points).astype(np.float32, copy=False)
                selected_t = (selected_t / float(self.radius)).astype(np.float32, copy=False)
                selected_perp = (selected_perp / float(self.radius)).astype(np.float32, copy=False)

            local_points = local_points.reshape(len(batch_positions), self.line_atoms, self.num_points, 3)
            line_points[batch_positions] = local_points
            line_atom_ids[batch_positions] = np.asarray(self.atom_ids[selected], dtype=np.int64)
            line_t[batch_positions] = selected_t
            line_perp[batch_positions] = selected_perp
            line_directions[batch_positions] = directions
            target_slot_atoms = selected[:, self.target_line_index]
            target_atom_ids[batch_positions] = np.asarray(self.atom_ids[target_slot_atoms], dtype=np.int64)
            anchor_atom_ids[batch_positions] = np.asarray(self.atom_ids[center_atom_indices], dtype=np.int64)
            frame_indices_batch[batch_positions] = frame_idx
            timesteps_batch[batch_positions] = int(self.timesteps[frame_idx])
            target_positions[batch_positions] = frame_points[target_slot_atoms] + self.box_low[frame_idx]
            anchor_positions_batch[batch_positions] = anchors + self.box_low[frame_idx]

            if self.line_anchor_views_enabled:
                if (
                    selected_anchor_views is None
                    or selected_anchor_view_t is None
                    or selected_anchor_view_perp is None
                    or anchor_view_points is None
                    or anchor_view_atom_ids is None
                    or anchor_view_t is None
                    or anchor_view_perp is None
                ):
                    raise RuntimeError("LAMMPS line anchor-view arrays were not initialized.")
                anchor_view_centers = np.asarray(
                    frame_points[selected_anchor_views.reshape(-1)],
                    dtype=np.float32,
                )
                anchor_view_local_indices = self._query_local_structures(
                    frame_idx=frame_idx,
                    centers=anchor_view_centers,
                )
                anchor_view_local_points = np.asarray(
                    frame_points[anchor_view_local_indices],
                    dtype=np.float32,
                )
                anchor_view_local_points = self._to_local_coordinates_batch(
                    frame_idx=frame_idx,
                    points=anchor_view_local_points,
                    centers=anchor_view_centers,
                )
                if self.normalize:
                    anchor_view_local_points = self._normalize_point_cloud_batch(
                        anchor_view_local_points
                    ).astype(np.float32, copy=False)
                    selected_anchor_view_t = (
                        selected_anchor_view_t / float(self.radius)
                    ).astype(np.float32, copy=False)
                    selected_anchor_view_perp = (
                        selected_anchor_view_perp / float(self.radius)
                    ).astype(np.float32, copy=False)
                anchor_view_points[batch_positions] = anchor_view_local_points.reshape(
                    len(batch_positions),
                    2,
                    self.num_points,
                    3,
                )
                anchor_view_atom_ids[batch_positions] = np.asarray(
                    self.atom_ids[selected_anchor_views],
                    dtype=np.int64,
                )
                anchor_view_t[batch_positions] = selected_anchor_view_t
                anchor_view_perp[batch_positions] = selected_anchor_view_perp

        result = {
            "points": torch.from_numpy(line_points),
            "line_atom_ids": torch.from_numpy(line_atom_ids),
            "line_t": torch.from_numpy(line_t),
            "line_perp": torch.from_numpy(line_perp),
            "line_direction": torch.from_numpy(line_directions),
            "target_line_index": torch.full((batch_size,), self.target_line_index, dtype=torch.long),
            "target_atom_id": torch.from_numpy(target_atom_ids),
            "anchor_atom_id": torch.from_numpy(anchor_atom_ids),
            "instance_id": torch.from_numpy(target_atom_ids.copy()),
            "frame_indices": torch.from_numpy(frame_indices_batch),
            "timesteps": torch.from_numpy(timesteps_batch),
            "coords": torch.from_numpy(target_positions),
            "anchor_positions": torch.from_numpy(anchor_positions_batch),
            "source_path": [str(self.dump_file)] * batch_size,
        }
        if self.line_anchor_views_enabled:
            if (
                anchor_view_points is None
                or anchor_view_atom_ids is None
                or anchor_view_t is None
                or anchor_view_perp is None
            ):
                raise RuntimeError("LAMMPS line anchor-view result arrays were not initialized.")
            result.update(
                {
                    "line_anchor_view_points": torch.from_numpy(anchor_view_points),
                    "line_anchor_view_atom_ids": torch.from_numpy(anchor_view_atom_ids),
                    "line_anchor_view_t": torch.from_numpy(anchor_view_t),
                    "line_anchor_view_perp": torch.from_numpy(anchor_view_perp),
                }
            )
        return result

    def _run_exact_line_diagnostic(self, sample_count: int) -> None:
        rng = np.random.default_rng(self.line_seed)
        frame_idx = int(self._window_start_frames[0])
        frame_points = np.asarray(self.positions[frame_idx], dtype=np.float32)
        box_lengths = np.asarray(self.box_lengths[frame_idx], dtype=np.float32)
        anchors_idx = self._center_atom_indices[: min(int(sample_count), self.center_count)]
        anchors = np.asarray(frame_points[anchors_idx], dtype=np.float32)
        directions = self._sample_directions(len(anchors), rng=rng)
        tree = self._get_tree(frame_idx)
        _, candidates = tree.query(anchors, k=self.line_candidate_atoms)
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        approx, _, _, _, _, _ = self._select_line_atoms_from_candidates(
            frame_points=frame_points,
            box_lengths=box_lengths,
            anchor_positions=anchors,
            anchor_indices=anchors_idx,
            candidate_indices=candidates,
            directions=directions,
            frame_idx_info=str(frame_idx),
        )

        all_indices = np.arange(self.num_atoms, dtype=np.int64).reshape(1, -1)
        exact_rows = []
        for row_idx in range(len(anchors)):
            exact_row, _, _, _, _, _ = self._select_line_atoms_from_candidates(
                frame_points=frame_points,
                box_lengths=box_lengths,
                anchor_positions=anchors[row_idx : row_idx + 1],
                anchor_indices=anchors_idx[row_idx : row_idx + 1],
                candidate_indices=all_indices,
                directions=directions[row_idx : row_idx + 1],
                frame_idx_info=str(frame_idx),
            )
            exact_rows.append(exact_row[0])
        overlaps = []
        for row_idx in range(len(anchors)):
            overlaps.append(len(set(approx[row_idx].tolist()) & set(exact_rows[row_idx].tolist())) / self.line_atoms)
        mean_overlap = float(np.mean(overlaps)) if overlaps else math.nan
        logger.print(
            "[line-jepa] exact line diagnostic: "
            f"samples={len(overlaps)}, line_atoms={self.line_atoms}, "
            f"candidate_atoms={self.line_candidate_atoms}, mean_exact_overlap={mean_overlap:.4f}."
        )


__all__ = ["LineLAMMPSDataset"]
