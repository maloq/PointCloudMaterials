from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset, get_worker_info

from src.data_utils.data_load import PointCloudDataset, _load_points
from src.utils.logging_config import setup_logging


logger = setup_logging()


class FixedSlotDirectionError(RuntimeError):
    """A sampled ray cannot satisfy fixed-slot geometry for its anchor batch."""


@dataclass
class _LineStaticSource:
    name: str
    group_name: str
    path: str
    points: np.ndarray
    tree: cKDTree
    radius: float
    center_indices: np.ndarray


class LineStaticPointCloudDataset(Dataset):
    """Sample line-ordered local structures from static point-cloud files."""

    def __init__(
        self,
        *,
        root: str = "",
        data_files: list[str] | None = None,
        data_sources: list[dict] | None = None,
        radius: float,
        num_points: int,
        line_atoms: int,
        line_candidate_atoms: int,
        line_samples_per_file: int,
        sample_indices: Sequence[int] | None = None,
        normalize: bool = True,
        center_neighborhoods: bool = True,
        drop_edge_samples: bool = True,
        edge_drop_layers: int | None = None,
        line_selection_method: str = "closest",
        line_min_separation_radius_factor: float = 0.0,
        line_slot_spacing_radius_factor: float | None = None,
        line_fixed_slot_max_deviation_radius_factor: float | None = None,
        line_direction_max_retries: int = 8,
        line_seed: int = 0,
        deterministic_lines: bool = False,
        auto_cutoff_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.radius = float(radius)
        self.num_points = int(num_points)
        self.line_atoms = int(line_atoms)
        self.line_candidate_atoms = int(line_candidate_atoms)
        self.line_samples_per_file = int(line_samples_per_file)
        self.normalize = bool(normalize)
        self.center_neighborhoods = bool(center_neighborhoods)
        self.drop_edge_samples = bool(drop_edge_samples)
        self.edge_drop_layers = None if edge_drop_layers is None else int(edge_drop_layers)
        self.line_selection_method = str(line_selection_method).strip().lower()
        self.line_min_separation_radius_factor = float(line_min_separation_radius_factor)
        self.line_slot_spacing_radius_factor = (
            None
            if line_slot_spacing_radius_factor is None
            else float(line_slot_spacing_radius_factor)
        )
        self.line_fixed_slot_max_deviation_radius_factor = (
            None
            if line_fixed_slot_max_deviation_radius_factor is None
            else float(line_fixed_slot_max_deviation_radius_factor)
        )
        self.line_direction_max_retries = int(line_direction_max_retries)
        self.line_seed = int(line_seed)
        self.deterministic_lines = bool(deterministic_lines)
        self._worker_rngs: dict[int, np.random.Generator] = {}

        if self.radius <= 0.0:
            raise ValueError(f"LineStaticPointCloudDataset requires radius > 0, got {self.radius}.")
        if self.num_points <= 0:
            raise ValueError(f"num_points must be > 0, got {self.num_points}.")
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
        if self.line_samples_per_file <= 0:
            raise ValueError(
                f"line_samples_per_file must be > 0, got {self.line_samples_per_file}."
            )
        if self.line_min_separation_radius_factor < 0.0:
            raise ValueError(
                "line_min_separation_radius_factor must be >= 0. "
                f"Got {self.line_min_separation_radius_factor}."
            )
        if self.line_selection_method not in {"closest", "radius_then_closest", "fixed_slots"}:
            raise ValueError(
                "line_selection_method must be 'closest', 'radius_then_closest', or 'fixed_slots', "
                f"got {line_selection_method!r}."
            )
        if self.line_selection_method == "fixed_slots":
            if self.line_slot_spacing_radius_factor is None or self.line_slot_spacing_radius_factor <= 0.0:
                raise ValueError(
                    "line_selection_method='fixed_slots' requires "
                    "line_slot_spacing_radius_factor > 0."
                )
            if self.line_min_separation_radius_factor != 0.0:
                raise ValueError(
                    "fixed_slots defines its own spacing; set "
                    "line_min_separation_radius_factor=0 to avoid conflicting geometry controls."
                )
        if (
            self.line_fixed_slot_max_deviation_radius_factor is not None
            and self.line_fixed_slot_max_deviation_radius_factor <= 0.0
        ):
            raise ValueError(
                "line_fixed_slot_max_deviation_radius_factor must be > 0 when provided, "
                f"got {self.line_fixed_slot_max_deviation_radius_factor}."
            )
        if self.line_direction_max_retries < 0:
            raise ValueError(
                "line_direction_max_retries must be >= 0, "
                f"got {self.line_direction_max_retries}."
            )

        self.sources = self._load_sources(
            root=root,
            data_files=data_files,
            data_sources=data_sources,
            auto_cutoff_config=auto_cutoff_config,
        )
        self.total_base_samples = len(self.sources) * self.line_samples_per_file
        if sample_indices is None:
            self.sample_indices = np.arange(self.total_base_samples, dtype=np.int64)
        else:
            self.sample_indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
            if self.sample_indices.size == 0:
                raise ValueError("LineStaticPointCloudDataset sample_indices must be non-empty.")
            if np.any(self.sample_indices < 0) or np.any(self.sample_indices >= self.total_base_samples):
                raise IndexError(
                    "LineStaticPointCloudDataset sample_indices contain out-of-range values. "
                    f"min={int(self.sample_indices.min())}, max={int(self.sample_indices.max())}, "
                    f"total_base_samples={self.total_base_samples}."
                )

    @property
    def target_line_index(self) -> int:
        return self.line_atoms // 2

    def __len__(self) -> int:
        return int(self.sample_indices.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        batch = self._build_batch_from_indices(np.asarray([index], dtype=np.int64))
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
        return self._build_batch_from_indices(index_array)

    def _resolve_radius_for_source(
        self,
        *,
        source: dict[str, Any],
        auto_cutoff_config: dict[str, Any] | None,
    ) -> float:
        radius_override = source.get("radius_override", None)
        if radius_override is not None:
            return float(radius_override)
        if auto_cutoff_config is None:
            return self.radius

        target_points = max(int(auto_cutoff_config["target_points"]), self.num_points)
        seed = int(auto_cutoff_config["seed"]) + int(source["index"])
        estimated_radius, coverage = PointCloudDataset._estimate_source_cutoff_radius(
            source_root=str(source["root"]),
            source_files=list(source["files"]),
            target_points=target_points,
            quantile=float(auto_cutoff_config["quantile"]),
            estimation_samples_per_file=int(auto_cutoff_config["estimation_samples_per_file"]),
            seed=seed,
            safety_factor=float(auto_cutoff_config["safety_factor"]),
            boundary_margin=auto_cutoff_config["boundary_margin"],
        )
        logger.print(
            "[line-static/auto_cutoff] "
            f"source={str(source['name'])!r}, target_points={target_points}, "
            f"quantile={float(auto_cutoff_config['quantile']):.4f}, "
            f"coverage~{coverage * 100.0:.2f}%, radius={estimated_radius:.4f}."
        )
        return float(estimated_radius)

    def _load_sources(
        self,
        *,
        root: str,
        data_files: list[str] | None,
        data_sources: list[dict] | None,
        auto_cutoff_config: dict[str, Any] | None,
    ) -> list[_LineStaticSource]:
        sources_cfg = PointCloudDataset._resolve_sources(root, data_files, data_sources)
        resolved_sources: list[_LineStaticSource] = []
        auto_cfg = PointCloudDataset._resolve_auto_cutoff_config(
            auto_cutoff_config,
            default_target_points=self.num_points,
            default_radius=self.radius,
        )
        for source in sources_cfg:
            source_radius = self._resolve_radius_for_source(
                source=source,
                auto_cutoff_config=auto_cfg,
            )
            for file_name in source["files"]:
                path = os.path.join(str(source["root"]), str(file_name))
                points = _load_points(path)
                if points.shape[0] < max(self.num_points, self.line_candidate_atoms):
                    raise ValueError(
                        "Static line source contains too few atoms for the requested local/line sampling. "
                        f"path={path}, atoms={points.shape[0]}, num_points={self.num_points}, "
                        f"line_candidate_atoms={self.line_candidate_atoms}."
                    )
                tree = cKDTree(points)
                center_indices = self._resolve_center_indices(points, radius=source_radius, path=path)
                resolved_sources.append(
                    _LineStaticSource(
                        name=f"{source['name']}:{Path(path).name}",
                        group_name=str(source["name"]),
                        path=str(Path(path).expanduser().resolve()),
                        points=points,
                        tree=tree,
                        radius=float(source_radius),
                        center_indices=center_indices,
                    )
                )
        if not resolved_sources:
            raise ValueError("LineStaticPointCloudDataset resolved zero static source files.")
        logger.print(
            "[line-static] Loaded "
            f"{len(resolved_sources)} source files, line_samples_per_file={self.line_samples_per_file}."
        )
        return resolved_sources

    def _resolve_center_indices(self, points: np.ndarray, *, radius: float, path: str) -> np.ndarray:
        if not self.drop_edge_samples:
            return np.arange(points.shape[0], dtype=np.int64)
        margin = float(radius)
        if self.edge_drop_layers is not None and self.edge_drop_layers > 1:
            margin *= float(self.edge_drop_layers)
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        mask = np.all(
            (points >= (min_coords + margin))
            & (points <= (max_coords - margin)),
            axis=1,
        )
        center_indices = np.flatnonzero(mask).astype(np.int64, copy=False)
        if center_indices.size == 0:
            raise ValueError(
                "drop_edge_samples removed every possible static line anchor. "
                f"path={path}, margin={margin}, radius={radius}, edge_drop_layers={self.edge_drop_layers}."
            )
        return center_indices

    def _rng_for_base_index(self, base_index: int) -> np.random.Generator:
        return np.random.default_rng(self.line_seed + int(base_index) * 104_729)

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

    @staticmethod
    def _sample_directions(count: int, *, rng: np.random.Generator) -> np.ndarray:
        directions = rng.normal(size=(int(count), 3)).astype(np.float32)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise RuntimeError(f"Random line direction sampler produced a zero-length direction. count={count}.")
        return directions / norms

    def _select_line_atoms_from_candidates(
        self,
        *,
        source: _LineStaticSource,
        anchor_positions: np.ndarray,
        anchor_indices: np.ndarray,
        candidate_indices: np.ndarray,
        directions: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if self.line_selection_method == "fixed_slots":
            return self._select_fixed_slot_atoms_from_candidates(
                source=source,
                anchor_positions=anchor_positions,
                anchor_indices=anchor_indices,
                candidate_indices=candidate_indices,
                directions=directions,
                target_index=self.target_line_index,
            )
        candidate_points = np.asarray(source.points[candidate_indices], dtype=np.float32)
        delta = candidate_points - anchor_positions[:, None, :]
        line_t = np.einsum("bkc,bc->bk", delta, directions, optimize=True)
        dist2 = np.einsum("bkc,bkc->bk", delta, delta, optimize=True)
        perp2 = np.maximum(dist2 - line_t * line_t, 0.0)

        if self.line_min_separation_radius_factor > 0.0:
            return self._select_separated_line_vectorized(
                source=source,
                candidate_indices=candidate_indices,
                anchor_indices=anchor_indices,
                delta=delta,
                line_t=line_t,
                dist2=dist2,
                perp2=perp2,
            )
        if self.line_min_separation_radius_factor <= 0.0:
            closest_slots = np.argpartition(
                perp2,
                self.line_atoms - 1,
                axis=1,
            )[:, : self.line_atoms]
            order = np.argsort(
                np.take_along_axis(line_t, closest_slots, axis=1),
                axis=1,
                kind="mergesort",
            )
            return self._gather_line_selection(
                candidate_indices=candidate_indices,
                line_t=line_t,
                perp2=perp2,
                selected_slots=np.take_along_axis(closest_slots, order, axis=1),
            )

    def _select_fixed_slot_atoms_from_candidates(
        self,
        *,
        source: _LineStaticSource,
        anchor_positions: np.ndarray,
        anchor_indices: np.ndarray,
        candidate_indices: np.ndarray,
        directions: np.ndarray,
        target_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assign unique atoms to equally spaced ideal points on a directed line."""
        if self.line_slot_spacing_radius_factor is None:
            raise RuntimeError("Fixed-slot selection has no configured slot spacing.")
        target_index = int(target_index)
        if target_index not in {0, self.target_line_index}:
            raise ValueError(
                f"Fixed-slot target_index must be 0 or {self.target_line_index}, got {target_index}."
            )
        delta, line_t, _, perp2, _ = self._explicit_line_geometry(
            source, anchor_positions, candidate_indices, directions
        )
        del delta
        anchor_matches = candidate_indices == anchor_indices[:, None]
        match_counts = anchor_matches.sum(axis=1)
        if np.any(match_counts != 1):
            first = int(np.flatnonzero(match_counts != 1)[0])
            raise RuntimeError(
                "Fixed-slot line selection expected the anchor exactly once in the candidate pool. "
                f"source={source.path}, row={first}, matches={int(match_counts[first])}."
            )
        batch_size, candidate_count = candidate_indices.shape
        anchor_slots = np.argmax(anchor_matches, axis=1)
        spacing = float(source.radius) * self.line_slot_spacing_radius_factor
        desired_t = (
            np.arange(self.line_atoms, dtype=np.float32) - float(target_index)
        ) * spacing
        selected_slots = np.empty((batch_size, self.line_atoms), dtype=np.int64)
        selected_slots[:, target_index] = anchor_slots
        used = np.zeros((batch_size, candidate_count), dtype=bool)
        rows = np.arange(batch_size)
        used[rows, anchor_slots] = True

        slot_order = sorted(
            (slot for slot in range(self.line_atoms) if slot != target_index),
            key=lambda slot: abs(slot - target_index),
        )
        for slot in slot_order:
            score = perp2 + (line_t - float(desired_t[slot])) ** 2
            score = np.where(used, np.inf, score)
            chosen = np.argmin(score, axis=1)
            chosen_score = score[rows, chosen]
            if not np.isfinite(chosen_score).all():
                first = int(np.flatnonzero(~np.isfinite(chosen_score))[0])
                raise RuntimeError(
                    "Fixed-slot line selection exhausted candidate atoms. "
                    f"source={source.path}, row={first}, slot={slot}, "
                    f"candidate_count={candidate_count}."
                )
            if self.line_fixed_slot_max_deviation_radius_factor is not None:
                max_deviation = (
                    float(source.radius)
                    * self.line_fixed_slot_max_deviation_radius_factor
                )
                deviation = np.sqrt(chosen_score)
                if np.any(deviation > max_deviation):
                    first = int(np.flatnonzero(deviation > max_deviation)[0])
                    raise FixedSlotDirectionError(
                        "Fixed-slot line selection could not find an atom close enough to an ideal slot. "
                        f"source={source.path}, row={first}, slot={slot}, "
                        f"desired_t={float(desired_t[slot]):.6f}, "
                        f"deviation={float(deviation[first]):.6f}, "
                        f"max_deviation={max_deviation:.6f}. Increase "
                        "line_fixed_slot_max_deviation_radius_factor or line_candidate_atoms."
                    )
            selected_slots[:, slot] = chosen
            used[rows, chosen] = True

        selected, selected_t, selected_perp = self._gather_line_selection(
            candidate_indices=candidate_indices,
            line_t=line_t,
            perp2=perp2,
            selected_slots=selected_slots,
        )
        if not np.array_equal(selected[:, target_index], anchor_indices):
            raise RuntimeError(
                f"Fixed-slot line selection lost its anchor in {source.path}."
            )
        return selected, selected_t, selected_perp

    @staticmethod
    def _gather_line_selection(
        *,
        candidate_indices: np.ndarray,
        line_t: np.ndarray,
        perp2: np.ndarray,
        selected_slots: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected_indices = np.take_along_axis(candidate_indices, selected_slots, axis=1)
        selected_t = np.take_along_axis(line_t, selected_slots, axis=1).astype(np.float32, copy=False)
        selected_perp = np.sqrt(
            np.take_along_axis(perp2, selected_slots, axis=1)
        ).astype(np.float32, copy=False)
        return selected_indices, selected_t, selected_perp

    def _select_separated_line_vectorized(
        self,
        *,
        source: _LineStaticSource,
        candidate_indices: np.ndarray,
        anchor_indices: np.ndarray,
        delta: np.ndarray,
        line_t: np.ndarray,
        dist2: np.ndarray,
        perp2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_distance = float(source.radius) * self.line_min_separation_radius_factor
        min_dist2 = min_distance * min_distance

        anchor_matches = candidate_indices == anchor_indices[:, None]
        anchor_match_counts = anchor_matches.sum(axis=1)
        bad_anchor_rows = np.flatnonzero(anchor_match_counts != 1)
        if bad_anchor_rows.size > 0:
            first = int(bad_anchor_rows[0])
            raise RuntimeError(
                "Static separated line selection expected the anchor atom to appear exactly once "
                "in the candidate pool. "
                f"source={source.path}, row_idx={first}, matches={int(anchor_match_counts[first])}, "
                f"anchor_index={int(anchor_indices[first])}, "
                f"line_candidate_atoms={self.line_candidate_atoms}."
            )

        anchor_slots = np.argmax(anchor_matches, axis=1).astype(np.int64, copy=False)
        side_count = self.line_atoms // 2
        negative_slots_list = self._select_separated_side_vectorized(
            source=source,
            side_name="negative",
            base_allowed=(line_t < 0.0) & (dist2 >= min_dist2),
            existing_selected_deltas=[],
            required=side_count,
            delta=delta,
            dist2=dist2,
            line_t=line_t,
            perp2=perp2,
            min_distance=min_distance,
            min_dist2=min_dist2,
        )
        row_indices = np.arange(candidate_indices.shape[0])
        negative_deltas = [
            delta[row_indices, negative_slots]
            for negative_slots in negative_slots_list
        ]
        positive_slots_list = self._select_separated_side_vectorized(
            source=source,
            side_name="positive",
            base_allowed=(line_t > 0.0) & (dist2 >= min_dist2),
            existing_selected_deltas=negative_deltas,
            required=side_count,
            delta=delta,
            dist2=dist2,
            line_t=line_t,
            perp2=perp2,
            min_distance=min_distance,
            min_dist2=min_dist2,
        )

        selected_slots = np.stack(
            negative_slots_list + [anchor_slots] + positive_slots_list,
            axis=1,
        )
        selected_order = np.argsort(
            np.take_along_axis(line_t, selected_slots, axis=1),
            axis=1,
            kind="mergesort",
        )
        return self._gather_line_selection(
            candidate_indices=candidate_indices,
            line_t=line_t,
            perp2=perp2,
            selected_slots=np.take_along_axis(selected_slots, selected_order, axis=1),
        )

    def _select_separated_side_vectorized(
        self,
        *,
        source: _LineStaticSource,
        side_name: str,
        base_allowed: np.ndarray,
        existing_selected_deltas: list[np.ndarray],
        required: int,
        delta: np.ndarray,
        dist2: np.ndarray,
        line_t: np.ndarray,
        perp2: np.ndarray,
        min_distance: float,
        min_dist2: float,
    ) -> list[np.ndarray]:
        allowed = np.array(base_allowed, dtype=bool, copy=True)
        for selected_delta in existing_selected_deltas:
            allowed &= self._candidate_separation_mask(
                delta=delta,
                dist2=dist2,
                selected_delta=selected_delta,
                min_dist2=min_dist2,
            )

        selected_slots_list: list[np.ndarray] = []
        row_indices = np.arange(delta.shape[0])
        for selected_count in range(int(required)):
            selected_slots = self._select_best_separated_candidate(
                source=source,
                side_name=side_name,
                allowed=allowed,
                line_t=line_t,
                perp2=perp2,
                min_distance=min_distance,
                required=int(required),
                selected_count=selected_count,
            )
            selected_slots_list.append(selected_slots)
            selected_delta = delta[row_indices, selected_slots]
            allowed &= self._candidate_separation_mask(
                delta=delta,
                dist2=dist2,
                selected_delta=selected_delta,
                min_dist2=min_dist2,
            )
        return selected_slots_list

    @staticmethod
    def _candidate_separation_mask(
        *,
        delta: np.ndarray,
        dist2: np.ndarray,
        selected_delta: np.ndarray,
        min_dist2: float,
    ) -> np.ndarray:
        selected_dist2 = np.einsum("bc,bc->b", selected_delta, selected_delta, optimize=True)
        candidate_selected_dot = np.einsum("bkc,bc->bk", delta, selected_delta, optimize=True)
        candidate_selected_dist2 = dist2 + selected_dist2[:, None] - 2.0 * candidate_selected_dot
        return candidate_selected_dist2 >= float(min_dist2)

    def _select_best_separated_candidate(
        self,
        *,
        source: _LineStaticSource,
        side_name: str,
        allowed: np.ndarray,
        line_t: np.ndarray,
        perp2: np.ndarray,
        min_distance: float,
        required: int,
        selected_count: int,
    ) -> np.ndarray:
        masked_perp2 = np.where(allowed, perp2, np.inf)
        selected_slots = np.argmin(masked_perp2, axis=1).astype(np.int64, copy=False)
        row_indices = np.arange(masked_perp2.shape[0])
        selected_perp2 = masked_perp2[row_indices, selected_slots]
        missing_rows = np.flatnonzero(~np.isfinite(selected_perp2))
        if missing_rows.size > 0:
            first = int(missing_rows[0])
            raise RuntimeError(
                "Static separated line selection could not place enough non-overlapping centers "
                f"on the {side_name} side of the target. source={source.path}, row_idx={first}, "
                f"required_side_centers={int(required)}, selected_side_centers={int(selected_count)}, "
                f"line_atoms={self.line_atoms}, line_candidate_atoms={self.line_candidate_atoms}, "
                f"min_center_distance={min_distance:.6f}, source_radius={source.radius:.6f}. "
                "Increase data.line_candidate_atoms or reduce data.line_min_separation_radius_factor."
            )

        tie_mask = allowed & (perp2 == selected_perp2[:, None])
        tie_counts = tie_mask.sum(axis=1)
        tie_rows = np.flatnonzero(tie_counts > 1)
        for row_raw in tie_rows.tolist():
            row = int(row_raw)
            tied_slots = np.flatnonzero(tie_mask[row])
            best_tie = np.argmin(np.abs(line_t[row, tied_slots]))
            selected_slots[row] = int(tied_slots[int(best_tie)])
        return selected_slots

    def _select_separated_line_slots(
        self,
        *,
        source: _LineStaticSource,
        row_idx: int,
        candidate_indices: np.ndarray,
        candidate_points: np.ndarray,
        anchor_indices: np.ndarray,
        line_t: np.ndarray,
        perp2: np.ndarray,
    ) -> np.ndarray:
        min_distance = float(source.radius) * self.line_min_separation_radius_factor
        anchor_matches = np.flatnonzero(candidate_indices[row_idx] == int(anchor_indices[row_idx]))
        if anchor_matches.size != 1:
            raise RuntimeError(
                "Static separated line selection expected the anchor atom to appear exactly once "
                "in the candidate pool. "
                f"source={source.path}, row_idx={row_idx}, matches={anchor_matches.size}, "
                f"anchor_index={int(anchor_indices[row_idx])}, "
                f"line_candidate_atoms={self.line_candidate_atoms}."
            )

        anchor_slot = int(anchor_matches[0])
        selected_slots: list[int] = [anchor_slot]
        selected_points = [candidate_points[row_idx, anchor_slot]]
        side_count = self.line_atoms // 2
        negative_slots = self._select_separated_line_side(
            source=source,
            row_idx=row_idx,
            side_name="negative",
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
            source=source,
            row_idx=row_idx,
            side_name="positive",
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
        source: _LineStaticSource,
        row_idx: int,
        side_name: str,
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
            distances = np.linalg.norm(np.asarray(selected_points) - point[None, :], axis=1)
            if np.any(distances < min_distance):
                continue
            selected_slots.append(slot)
            selected_points.append(point)
            chosen.append(slot)
            if len(chosen) == required:
                return chosen
        raise RuntimeError(
            "Static separated line selection could not place enough non-overlapping centers "
            f"on the {side_name} side of the target. source={source.path}, row_idx={row_idx}, "
            f"required_side_centers={required}, selected_side_centers={len(chosen)}, "
            f"line_atoms={self.line_atoms}, line_candidate_atoms={self.line_candidate_atoms}, "
            f"min_center_distance={min_distance:.6f}, source_radius={source.radius:.6f}. "
            "Increase data.line_candidate_atoms or reduce data.line_min_separation_radius_factor."
        )

    def _query_local_structures(
        self,
        *,
        source: _LineStaticSource,
        centers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = source.tree.query(centers, k=self.num_points)
        distances = np.asarray(distances, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim == 1:
            indices = indices.reshape(1, -1)
            distances = distances.reshape(1, -1)
        expected_shape = (int(centers.shape[0]), self.num_points)
        if indices.shape != expected_shape:
            raise RuntimeError(
                "Static line local-neighborhood KDTree query returned an unexpected shape. "
                f"source={source.path}, expected_shape={expected_shape}, got_shape={tuple(indices.shape)}."
            )

        if self.line_selection_method == "radius_then_closest":
            within_counts = np.sum(distances <= source.radius, axis=1)
            shortfall_rows = np.where(within_counts < self.num_points)[0]
            if shortfall_rows.size > 0:
                first = int(shortfall_rows[0])
                raise RuntimeError(
                    "line_selection_method='radius_then_closest' found fewer atoms within the cutoff "
                    "radius than required. "
                    f"source={source.path}, shortfall_rows={shortfall_rows.size}, first_row={first}, "
                    f"within_radius={int(within_counts[first])}, required={self.num_points}, "
                    f"radius={source.radius}, max_distance_row={float(distances[first].max()):.6f}."
                )
        return indices, distances

    def _build_batch_from_indices(self, indices: np.ndarray) -> dict[str, Any]:
        public_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if public_indices.size == 0:
            raise ValueError("Static Line-JEPA batch indices must be non-empty.")
        if np.any(public_indices < 0) or np.any(public_indices >= len(self)):
            raise IndexError(
                "Static Line-JEPA batch indices are out of range. "
                f"min_index={int(public_indices.min())}, max_index={int(public_indices.max())}, len={len(self)}."
            )

        base_indices = self.sample_indices[public_indices]
        source_slots = (base_indices // self.line_samples_per_file).astype(np.int64, copy=False)
        batch_size = int(public_indices.size)
        line_points = np.empty((batch_size, self.line_atoms, self.num_points, 3), dtype=np.float32)
        line_atom_ids = np.empty((batch_size, self.line_atoms), dtype=np.int64)
        line_t = np.empty((batch_size, self.line_atoms), dtype=np.float32)
        line_perp = np.empty((batch_size, self.line_atoms), dtype=np.float32)
        line_directions = np.empty((batch_size, 3), dtype=np.float32)
        target_atom_ids = np.empty((batch_size,), dtype=np.int64)
        anchor_atom_ids = np.empty((batch_size,), dtype=np.int64)
        coords = np.empty((batch_size, 3), dtype=np.float32)
        source_names: list[str | None] = [None] * batch_size
        source_groups: list[str | None] = [None] * batch_size
        source_paths: list[str | None] = [None] * batch_size

        grouped_positions: dict[int, list[int]] = {}
        for batch_pos, source_slot in enumerate(source_slots.tolist()):
            grouped_positions.setdefault(int(source_slot), []).append(int(batch_pos))

        worker_rng = None if self.deterministic_lines else self._rng_for_worker()
        for source_slot, batch_positions_list in grouped_positions.items():
            source = self.sources[source_slot]
            batch_positions = np.asarray(batch_positions_list, dtype=np.int64)
            direction_rngs: list[np.random.Generator] | None = None
            if self.deterministic_lines:
                anchor_indices = []
                directions_rows = []
                direction_rngs = []
                for pos in batch_positions.tolist():
                    rng = self._rng_for_base_index(int(base_indices[pos]))
                    anchor_indices.append(int(rng.choice(source.center_indices)))
                    directions_rows.append(self._sample_directions(1, rng=rng)[0])
                    direction_rngs.append(rng)
                anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
                directions = np.asarray(directions_rows, dtype=np.float32)
            else:
                if worker_rng is None:
                    raise RuntimeError(
                        "Static Line-JEPA worker RNG was not initialized for stochastic line sampling. "
                        f"deterministic_lines={self.deterministic_lines}, source={source.path}."
                    )
                anchor_indices = worker_rng.choice(
                    source.center_indices,
                    size=len(batch_positions),
                    replace=True,
                ).astype(np.int64, copy=False)
                directions = self._sample_directions(len(batch_positions), rng=worker_rng)

            anchors = np.asarray(source.points[anchor_indices], dtype=np.float32)
            _, candidates = source.tree.query(anchors, k=self.line_candidate_atoms)
            candidates = np.asarray(candidates, dtype=np.int64)
            if candidates.ndim == 1:
                candidates = candidates.reshape(1, -1)
            expected_candidates_shape = (len(batch_positions), self.line_candidate_atoms)
            if candidates.shape != expected_candidates_shape:
                raise RuntimeError(
                    "Static line candidate KDTree query returned an unexpected shape. "
                    f"source={source.path}, expected_shape={expected_candidates_shape}, "
                    f"got_shape={tuple(candidates.shape)}."
                )

            direction_attempt = 0
            first_direction_error: FixedSlotDirectionError | None = None
            while True:
                try:
                    (
                        selected,
                        selected_t,
                        selected_perp,
                    ) = self._select_line_atoms_from_candidates(
                        source=source,
                        anchor_positions=anchors,
                        anchor_indices=anchor_indices,
                        candidate_indices=candidates,
                        directions=directions,
                    )
                    break
                except FixedSlotDirectionError as exc:
                    if first_direction_error is None:
                        first_direction_error = exc
                    if direction_attempt >= self.line_direction_max_retries:
                        raise RuntimeError(
                            "Fixed-slot line selection exhausted random direction retries. "
                            f"source={source.path}, batch_rows={len(batch_positions)}, "
                            f"retries={self.line_direction_max_retries}, "
                            f"first_error={first_direction_error}, last_error={exc}. "
                            "Increase data.line_direction_max_retries only if additional random "
                            "rays are scientifically acceptable; otherwise relax the explicit "
                            "slot geometry configuration."
                        ) from exc
                    direction_attempt += 1
                    if direction_rngs is not None:
                        directions = np.asarray(
                            [self._sample_directions(1, rng=rng)[0] for rng in direction_rngs],
                            dtype=np.float32,
                        )
                    else:
                        if worker_rng is None:
                            raise RuntimeError(
                                "Fixed-slot retry has neither deterministic per-row RNGs nor a "
                                f"worker RNG. source={source.path}."
                            ) from exc
                        directions = self._sample_directions(
                            len(batch_positions),
                            rng=worker_rng,
                        )
            if direction_attempt > 0:
                logger.warning(
                    "[line-static/fixed-slots] Resampled line directions after an invalid "
                    "fixed-slot ray: source=%s, attempts=%d, batch_rows=%d, first_error=%s",
                    source.path,
                    direction_attempt,
                    len(batch_positions),
                    first_direction_error,
                )
            flat_centers = np.asarray(source.points[selected.reshape(-1)], dtype=np.float32)
            local_indices, _ = self._query_local_structures(source=source, centers=flat_centers)
            local_points = np.asarray(source.points[local_indices], dtype=np.float32)
            if self.center_neighborhoods:
                local_points = local_points - flat_centers[:, None, :]
            if self.normalize:
                local_points = (local_points / float(source.radius)).astype(np.float32, copy=False)
                selected_t = (selected_t / float(source.radius)).astype(np.float32, copy=False)
                selected_perp = (selected_perp / float(source.radius)).astype(np.float32, copy=False)

            line_points[batch_positions] = local_points.reshape(
                len(batch_positions),
                self.line_atoms,
                self.num_points,
                3,
            )
            line_atom_ids[batch_positions] = selected
            line_t[batch_positions] = selected_t
            line_perp[batch_positions] = selected_perp
            line_directions[batch_positions] = directions
            target_indices = selected[:, self.target_line_index]
            target_atom_ids[batch_positions] = target_indices
            anchor_atom_ids[batch_positions] = anchor_indices
            coords[batch_positions] = source.points[target_indices]
            for pos in batch_positions.tolist():
                source_names[pos] = source.name
                source_groups[pos] = source.group_name
                source_paths[pos] = source.path

        if (
            any(value is None for value in source_names)
            or any(value is None for value in source_groups)
            or any(value is None for value in source_paths)
        ):
            raise RuntimeError("Static Line-JEPA batch assembly left source metadata unset.")

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
            "coords": torch.from_numpy(coords),
            "source_name": [str(value) for value in source_names],
            "source_group": [str(value) for value in source_groups],
            "source_path": [str(value) for value in source_paths],
        }
        return result

    def resolve_explicit_centers(
        self,
        *,
        source_names: Sequence[str],
        center_coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Snap analysis centers to atoms in their corresponding static source."""
        names = [str(value) for value in source_names]
        coords = np.asarray(center_coords, dtype=np.float32)
        if coords.shape != (len(names), 3):
            raise ValueError(f"Expected {len(names)} center coordinates, got {coords.shape}.")

        source_slots = np.empty((len(names),), dtype=np.int64)
        center_atom_ids = np.empty((len(names),), dtype=np.int64)
        snap_distances = np.empty((len(names),), dtype=np.float32)
        slots_by_name: dict[str, list[int]] = {}
        positions_by_name: dict[str, list[int]] = {}
        for slot, source in enumerate(self.sources):
            slots_by_name.setdefault(source.group_name, []).append(slot)
        for row, source_name in enumerate(names):
            positions_by_name.setdefault(source_name, []).append(row)

        for source_name, positions_list in positions_by_name.items():
            matching_slots = slots_by_name.get(source_name, [])
            if not matching_slots and len(slots_by_name) == 1:
                matching_slots = list(range(len(self.sources)))
            if not matching_slots:
                raise KeyError(
                    f"Unknown source {source_name!r}; available={sorted(slots_by_name)}."
                )
            positions = np.asarray(positions_list, dtype=np.int64)
            queries = [self.sources[slot].tree.query(coords[positions], k=1) for slot in matching_slots]
            distances = np.stack([query[0] for query in queries])
            atom_ids = np.stack([query[1] for query in queries])
            choice = np.argmin(distances, axis=0)
            columns = np.arange(len(positions))
            source_slots[positions] = np.asarray(matching_slots)[choice]
            center_atom_ids[positions] = atom_ids[choice, columns]
            snap_distances[positions] = distances[choice, columns]

        return source_slots, center_atom_ids, snap_distances

    @staticmethod
    def _explicit_line_geometry(
        source: _LineStaticSource,
        anchors: np.ndarray,
        candidates: np.ndarray,
        directions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        delta = source.points[candidates] - anchors[:, None, :]
        line_t = np.einsum("bkc,bc->bk", delta, directions, optimize=True)
        dist2 = np.einsum("bkc,bkc->bk", delta, delta, optimize=True)
        perp2 = np.maximum(dist2 - line_t**2, 0.0)
        anchor_slots = np.argmin(dist2, axis=1)
        return delta, line_t, dist2, perp2, anchor_slots

    def _select_centered_line_atoms_from_candidates(
        self,
        *,
        source: _LineStaticSource,
        anchor_positions: np.ndarray,
        anchor_indices: np.ndarray,
        candidate_indices: np.ndarray,
        directions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select a balanced line whose middle entry is always the requested atom."""
        if self.line_selection_method == "fixed_slots":
            return self._select_fixed_slot_atoms_from_candidates(
                source=source,
                anchor_positions=anchor_positions,
                anchor_indices=anchor_indices,
                candidate_indices=candidate_indices,
                directions=directions,
                target_index=self.target_line_index,
            )
        delta, line_t, dist2, perp2, anchor_slots = self._explicit_line_geometry(
            source, anchor_positions, candidate_indices, directions
        )

        if self.line_min_separation_radius_factor > 0.0:
            selected, selected_t, selected_perp = self._select_separated_line_vectorized(
                source=source,
                candidate_indices=candidate_indices,
                anchor_indices=anchor_indices,
                delta=delta,
                line_t=line_t,
                dist2=dist2,
                perp2=perp2,
            )
        else:
            side_count = self.line_atoms // 2
            negative_slots = self._closest_centered_side_slots(
                line_t=line_t,
                perp2=perp2,
                side_mask=line_t < 0.0,
                required=side_count,
                side_name="negative",
                source=source,
            )
            positive_slots = self._closest_centered_side_slots(
                line_t=line_t,
                perp2=perp2,
                side_mask=line_t > 0.0,
                required=side_count,
                side_name="positive",
                source=source,
            )
            selected_slots = np.concatenate(
                (negative_slots, anchor_slots[:, None], positive_slots),
                axis=1,
            )
            selected, selected_t, selected_perp = self._gather_line_selection(
                candidate_indices=candidate_indices,
                line_t=line_t,
                perp2=perp2,
                selected_slots=selected_slots,
            )

        if not np.array_equal(selected[:, self.target_line_index], anchor_indices):
            raise RuntimeError(
                f"Centered directional line lost its anchor in {source.path}."
            )
        return selected, selected_t, selected_perp

    def _select_endpoint_line_atoms_from_candidates(
        self,
        *,
        source: _LineStaticSource,
        anchor_positions: np.ndarray,
        anchor_indices: np.ndarray,
        candidate_indices: np.ndarray,
        directions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select a one-sided line with the requested atom at index zero."""
        if self.line_selection_method == "fixed_slots":
            return self._select_fixed_slot_atoms_from_candidates(
                source=source,
                anchor_positions=anchor_positions,
                anchor_indices=anchor_indices,
                candidate_indices=candidate_indices,
                directions=directions,
                target_index=0,
            )
        delta, line_t, dist2, perp2, anchor_slots = self._explicit_line_geometry(
            source, anchor_positions, candidate_indices, directions
        )
        context_count = self.line_atoms - 1
        if self.line_min_separation_radius_factor > 0.0:
            min_distance = float(source.radius) * self.line_min_separation_radius_factor
            context_slots_list = self._select_separated_side_vectorized(
                source=source,
                side_name="positive",
                base_allowed=(line_t > 0.0) & (dist2 >= min_distance * min_distance),
                existing_selected_deltas=[],
                required=context_count,
                delta=delta,
                dist2=dist2,
                line_t=line_t,
                perp2=perp2,
                min_distance=min_distance,
                min_dist2=min_distance * min_distance,
            )
            context_slots = np.stack(context_slots_list, axis=1)
        else:
            context_slots = self._closest_centered_side_slots(
                line_t=line_t,
                perp2=perp2,
                side_mask=line_t > 0.0,
                required=context_count,
                side_name="positive endpoint context",
                source=source,
            )
        selected_slots = np.concatenate((anchor_slots[:, None], context_slots), axis=1)
        selected_order = np.argsort(
            np.take_along_axis(line_t, selected_slots, axis=1),
            axis=1,
            kind="mergesort",
        )
        selected, selected_t, selected_perp = self._gather_line_selection(
            candidate_indices=candidate_indices,
            line_t=line_t,
            perp2=perp2,
            selected_slots=np.take_along_axis(selected_slots, selected_order, axis=1),
        )
        if not np.array_equal(selected[:, 0], anchor_indices):
            raise RuntimeError(
                f"Endpoint directional line lost its anchor in {source.path}."
            )
        return selected, selected_t, selected_perp

    @staticmethod
    def _closest_centered_side_slots(
        *,
        line_t: np.ndarray,
        perp2: np.ndarray,
        side_mask: np.ndarray,
        required: int,
        side_name: str,
        source: _LineStaticSource,
    ) -> np.ndarray:
        scores = np.where(side_mask, perp2, np.inf)
        available = np.isfinite(scores).sum(axis=1)
        if np.any(available < int(required)):
            first = int(np.flatnonzero(available < int(required))[0])
            raise RuntimeError(
                "Centered static line selection has too few candidates on one side. "
                f"source={source.path}, side={side_name}, row={first}, "
                f"available={int(available[first])}, required={int(required)}. "
                "Increase data.line_candidate_atoms."
            )
        slots = np.argpartition(scores, int(required) - 1, axis=1)[:, : int(required)]
        selected_t = np.take_along_axis(line_t, slots, axis=1)
        order = np.argsort(selected_t, axis=1, kind="mergesort")
        return np.take_along_axis(slots, order, axis=1)

    def build_atom_environment_batch(
        self,
        *,
        source_slots: np.ndarray,
        atom_ids: np.ndarray,
    ) -> torch.Tensor:
        """Materialize local structures for unique source/atom pairs."""
        source_slots = np.asarray(source_slots, dtype=np.int64).reshape(-1)
        atom_ids = np.asarray(atom_ids, dtype=np.int64).reshape(-1)
        if source_slots.shape != atom_ids.shape or source_slots.size == 0:
            raise ValueError(
                f"Expected equally sized non-empty source/atom arrays, got "
                f"{source_slots.shape} and {atom_ids.shape}."
            )
        environments = np.empty(
            (source_slots.size, self.num_points, 3), dtype=np.float32
        )
        for source_slot_raw in np.unique(source_slots):
            source_slot = int(source_slot_raw)
            source = self.sources[source_slot]
            rows = np.flatnonzero(source_slots == source_slot)
            centers = np.asarray(source.points[atom_ids[rows]], dtype=np.float32)
            local_indices, _ = self._query_local_structures(source=source, centers=centers)
            local = np.asarray(source.points[local_indices], dtype=np.float32)
            if self.center_neighborhoods:
                local -= centers[:, None, :]
            if self.normalize:
                local /= float(source.radius)
            environments[rows] = local
        return torch.from_numpy(environments)

    def build_explicit_direction_batch(
        self,
        *,
        source_slots: np.ndarray,
        center_atom_ids: np.ndarray,
        directions: np.ndarray,
        target_index: int | None = None,
        candidate_indices: np.ndarray | None = None,
        materialize_points: bool = True,
    ) -> dict[str, Any]:
        """Build deterministic line contexts for explicit centers and directions."""
        source_slots = np.asarray(source_slots, dtype=np.int64).reshape(-1)
        center_atom_ids = np.asarray(center_atom_ids, dtype=np.int64).reshape(-1)
        directions = np.asarray(directions, dtype=np.float32)
        batch_size = int(source_slots.size)
        if not batch_size or center_atom_ids.shape != (batch_size,) or directions.shape != (batch_size, 3):
            raise ValueError(
                f"Expected non-empty (B,), (B,), (B,3) inputs; got "
                f"{source_slots.shape}, {center_atom_ids.shape}, {directions.shape}."
            )
        direction_norms = np.linalg.norm(directions, axis=1)
        if not np.isfinite(directions).all() or np.any(direction_norms <= 0.0):
            raise ValueError("Explicit Line-JEPA directions must be finite and non-zero.")
        directions = directions / direction_norms[:, None]
        resolved_target_index = self.target_line_index if target_index is None else int(target_index)
        if resolved_target_index not in {0, self.target_line_index}:
            raise ValueError(
                f"target_index must be 0 or {self.target_line_index}, got {resolved_target_index}."
            )
        if candidate_indices is None:
            candidate_indices = self.query_explicit_line_candidates(
                source_slots=source_slots, center_atom_ids=center_atom_ids
            )
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        if candidate_indices.shape != (batch_size, self.line_candidate_atoms):
            raise ValueError(f"Invalid candidate pool shape {candidate_indices.shape}.")

        line_atom_ids = np.empty((batch_size, self.line_atoms), dtype=np.int64)
        line_t = np.empty((batch_size, self.line_atoms), dtype=np.float32)
        line_perp = np.empty((batch_size, self.line_atoms), dtype=np.float32)
        coords = np.empty((batch_size, 3), dtype=np.float32)
        selector = (
            self._select_endpoint_line_atoms_from_candidates
            if resolved_target_index == 0
            else self._select_centered_line_atoms_from_candidates
        )

        for source_slot_raw in np.unique(source_slots):
            source_slot = int(source_slot_raw)
            source = self.sources[source_slot]
            rows = np.flatnonzero(source_slots == source_slot).astype(np.int64, copy=False)
            atom_ids = center_atom_ids[rows]
            anchors = np.asarray(source.points[atom_ids], dtype=np.float32)
            selected, selected_t, selected_perp = selector(
                source=source,
                anchor_positions=anchors,
                anchor_indices=atom_ids,
                candidate_indices=candidate_indices[rows],
                directions=directions[rows],
            )
            if self.normalize:
                selected_t = (selected_t / float(source.radius)).astype(np.float32, copy=False)
                selected_perp = (selected_perp / float(source.radius)).astype(np.float32, copy=False)

            line_atom_ids[rows] = selected
            line_t[rows] = selected_t
            line_perp[rows] = selected_perp
            coords[rows] = anchors

        ids_tensor = torch.from_numpy(center_atom_ids.copy())
        result = {
            "line_atom_ids": torch.from_numpy(line_atom_ids),
            "line_t": torch.from_numpy(line_t),
            "line_perp": torch.from_numpy(line_perp),
            "line_direction": torch.from_numpy(directions.astype(np.float32, copy=False)),
            "target_line_index": torch.full((batch_size,), resolved_target_index, dtype=torch.long),
            "target_atom_id": ids_tensor,
            "anchor_atom_id": ids_tensor,
            "instance_id": ids_tensor,
            "coords": torch.from_numpy(coords),
            "source_name": [self.sources[int(slot)].group_name for slot in source_slots],
            "source_group": [self.sources[int(slot)].group_name for slot in source_slots],
            "source_path": [self.sources[int(slot)].path for slot in source_slots],
        }
        if materialize_points:
            environments = self.build_atom_environment_batch(
                source_slots=np.repeat(source_slots, self.line_atoms),
                atom_ids=line_atom_ids.reshape(-1),
            )
            result["points"] = environments.reshape(
                batch_size, self.line_atoms, self.num_points, 3
            )
        return result

    def query_explicit_line_candidates(
        self,
        *,
        source_slots: np.ndarray,
        center_atom_ids: np.ndarray,
    ) -> np.ndarray:
        """Query each atom's candidate pool once for reuse across directions."""
        source_slots = np.asarray(source_slots, dtype=np.int64).reshape(-1)
        center_atom_ids = np.asarray(center_atom_ids, dtype=np.int64).reshape(-1)
        if source_slots.shape != center_atom_ids.shape or source_slots.size == 0:
            raise ValueError(
                f"Expected equally sized source/atom arrays, got "
                f"{source_slots.shape} and {center_atom_ids.shape}."
            )
        candidates = np.empty((source_slots.size, self.line_candidate_atoms), dtype=np.int64)
        for source_slot_raw in np.unique(source_slots):
            source_slot = int(source_slot_raw)
            if source_slot < 0 or source_slot >= len(self.sources):
                raise IndexError(f"Source slot {source_slot} out of range for {len(self.sources)}.")
            rows = np.flatnonzero(source_slots == source_slot)
            source = self.sources[source_slot]
            atom_ids = center_atom_ids[rows]
            _, result = source.tree.query(source.points[atom_ids], k=self.line_candidate_atoms)
            candidates[rows] = np.asarray(result).reshape(len(rows), self.line_candidate_atoms)
        return candidates


__all__ = ["LineStaticPointCloudDataset"]
