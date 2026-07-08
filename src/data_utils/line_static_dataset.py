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


@dataclass
class _LineStaticSource:
    name: str
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
        if self.line_selection_method not in {"closest", "radius_then_closest"}:
            raise ValueError(
                "line_selection_method must be 'closest' or 'radius_then_closest', "
                f"got {line_selection_method!r}."
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
        source_paths: list[str | None] = [None] * batch_size

        grouped_positions: dict[int, list[int]] = {}
        for batch_pos, source_slot in enumerate(source_slots.tolist()):
            grouped_positions.setdefault(int(source_slot), []).append(int(batch_pos))

        worker_rng = None if self.deterministic_lines else self._rng_for_worker()
        for source_slot, batch_positions_list in grouped_positions.items():
            source = self.sources[source_slot]
            batch_positions = np.asarray(batch_positions_list, dtype=np.int64)
            if self.deterministic_lines:
                anchor_indices = []
                directions_rows = []
                for pos in batch_positions.tolist():
                    rng = self._rng_for_base_index(int(base_indices[pos]))
                    anchor_indices.append(int(rng.choice(source.center_indices)))
                    directions_rows.append(self._sample_directions(1, rng=rng)[0])
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
                source_paths[pos] = source.path

        if any(value is None for value in source_names) or any(value is None for value in source_paths):
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
            "source_path": [str(value) for value in source_paths],
        }
        return result


__all__ = ["LineStaticPointCloudDataset"]
