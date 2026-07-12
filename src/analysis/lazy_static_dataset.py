"""Cache-backed static dataset that loads atom neighborhoods only when requested."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from src.data_utils.data_load import PointCloudDataset, _ShardValueSequence, _load_points
from src.data_utils.prepare_data import _resolve_drop_func

from .analysis_dataloaders import _analysis_dataloader_kwargs


@dataclass
class _Shard:
    name: str
    path: str
    centers: np.ndarray
    source: dict[str, Any]
    default_radius: float
    auto_cutoff: dict[str, Any] | None
    points: np.ndarray | None = None
    tree: cKDTree | None = None
    radius: float | None = None


def _plain(value: Any) -> Any:
    return (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (DictConfig, ListConfig))
        else value
    )


class LazyStaticAnalysisDataset(Dataset):
    """Expose cached centers immediately and materialize rare representatives lazily."""

    def __init__(self, cfg: DictConfig, *, expected_coords: np.ndarray) -> None:
        data_cfg = cfg.data
        if str(getattr(data_cfg, "sample_type", "regular")).lower() != "regular":
            raise ValueError("Cache-backed static analysis requires data.sample_type='regular'.")
        coords = np.asarray(expected_coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Cached coordinates must have shape (N, 3), got {coords.shape}.")

        self.num_points = int(data_cfg.num_points)
        self.normalize = bool(getattr(data_cfg, "normalize", True))
        self._drop_points = _resolve_drop_func(
            str(getattr(data_cfg, "sampling_method", "drop_farthest"))
        )
        sources = PointCloudDataset._resolve_sources(
            str(getattr(data_cfg, "data_path", "")),
            _plain(getattr(data_cfg, "data_files", None)),
            _plain(getattr(data_cfg, "data_sources", None)),
        )
        entries = [(source, str(name)) for source in sources for name in source["files"]]
        counts = self._infer_cached_file_counts(coords, file_count=len(entries))
        auto_cutoff = PointCloudDataset._resolve_auto_cutoff_config(
            _plain(getattr(data_cfg, "auto_cutoff", None)),
            default_target_points=self.num_points,
            default_radius=float(data_cfg.radius),
        )

        self.shards: list[_Shard] = []
        offset = 0
        for (source, file_name), count in zip(entries, counts, strict=True):
            radius_override = source.get("radius_override")
            self.shards.append(
                _Shard(
                    name=str(source["name"]),
                    path=str((Path(source["root"]) / file_name).resolve()),
                    centers=coords[offset : offset + count],
                    source=source,
                    default_radius=float(data_cfg.radius),
                    auto_cutoff=auto_cutoff,
                    radius=None if radius_override is None else float(radius_override),
                )
            )
            offset += count

        shard_counts = [len(shard.centers) for shard in self.shards]
        self._cumulative_counts = np.cumsum(shard_counts, dtype=np.int64).tolist()
        self.sample_source_names = _ShardValueSequence(
            [shard.name for shard in self.shards], shard_counts
        )
        radii = [float(shard.radius or shard.default_radius) for shard in self.shards]
        self.sample_radii = _ShardValueSequence(radii, shard_counts)
        self.source_radii = {
            shard.name: radius for shard, radius in zip(self.shards, radii, strict=True)
        }

    @staticmethod
    def _infer_cached_file_counts(expected_coords: np.ndarray, *, file_count: int) -> list[int]:
        """Infer concatenated file runs from the large x reset between regular grids."""
        rows = int(expected_coords.shape[0])
        if file_count <= 0 or rows < file_count:
            raise ValueError(
                "Cannot infer cached file boundaries: "
                f"cache_rows={rows}, file_count={file_count}."
            )
        if file_count == 1:
            return [rows]
        x = np.asarray(expected_coords[:, 0], dtype=np.float64)
        deltas = np.diff(x)
        resets = np.argpartition(deltas, file_count - 2)[: file_count - 1] + 1
        threshold = -max(1.0e-6, 0.25 * float(np.ptp(x)))
        if np.any(deltas[resets - 1] > threshold):
            raise RuntimeError(
                "Cache has too few coordinate resets to recover all source files; "
                "recompute it without a global sample cap. "
                f"file_count={file_count}, selected_deltas={deltas[resets - 1].tolist()}."
            )
        boundaries = np.concatenate(([0], np.sort(resets), [rows]))
        return [int(value) for value in np.diff(boundaries)]

    def __len__(self) -> int:
        return int(self._cumulative_counts[-1])

    def _resolve(self, index: int) -> tuple[_Shard, int]:
        resolved = int(index)
        if resolved < 0:
            resolved += len(self)
        if not 0 <= resolved < len(self):
            raise IndexError(f"Static analysis index {index} out of range for {len(self)} rows.")
        shard_index = bisect.bisect_right(self._cumulative_counts, resolved)
        previous = 0 if shard_index == 0 else self._cumulative_counts[shard_index - 1]
        return self.shards[shard_index], resolved - previous

    def center(self, index: int) -> np.ndarray:
        shard, local_index = self._resolve(index)
        return shard.centers[local_index]

    def _load(self, shard: _Shard) -> None:
        if shard.points is not None:
            return
        if shard.radius is None:
            if shard.auto_cutoff is None:
                shard.radius = shard.default_radius
            else:
                ac = shard.auto_cutoff
                shard.radius, _ = PointCloudDataset._estimate_source_cutoff_radius(
                    source_root=str(shard.source["root"]),
                    source_files=list(shard.source["files"]),
                    target_points=max(int(ac["target_points"]), self.num_points),
                    quantile=float(ac["quantile"]),
                    estimation_samples_per_file=int(ac["estimation_samples_per_file"]),
                    seed=int(ac["seed"]) + int(shard.source["index"]),
                    safety_factor=float(ac["safety_factor"]),
                    boundary_margin=ac["boundary_margin"],
                )
        shard.points = _load_points(shard.path)
        shard.tree = cKDTree(shard.points)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard, local_index = self._resolve(index)
        self._load(shard)
        center = shard.centers[local_index]
        neighbor_ids = shard.tree.query_ball_point(center, shard.radius)
        if not neighbor_ids:
            raise RuntimeError(f"No atoms near cached center {center.tolist()} in {shard.path}.")
        points = self._drop_points(shard.points[np.asarray(neighbor_ids)], self.num_points)
        if not np.any(np.sum((points - center) ** 2, axis=1) < 1.0e-16):
            points[int(np.argmax(np.sum((points - center) ** 2, axis=1)))] = center
        local = np.asarray(points - center, dtype=np.float32)
        if self.normalize:
            local /= float(shard.radius)
        return {"points": torch.from_numpy(local), "coords": torch.from_numpy(center.copy())}


def build_lazy_static_analysis_dataloader(
    cfg: DictConfig,
    *,
    expected_coords: np.ndarray,
    batch_size: int,
    dataloader_num_workers: int,
) -> DataLoader:
    dataset = LazyStaticAnalysisDataset(cfg, expected_coords=expected_coords)
    print(
        f"[analysis][fast-path] {len(dataset)} cached centers across "
        f"{len(dataset.shards)} files; neighborhoods load on demand."
    )
    return DataLoader(
        dataset,
        **_analysis_dataloader_kwargs(
            batch_size=int(batch_size),
            dataloader_num_workers=int(dataloader_num_workers),
        ),
    )


__all__ = ["LazyStaticAnalysisDataset", "build_lazy_static_analysis_dataloader"]
