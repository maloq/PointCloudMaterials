"""Cache-backed static dataset that loads atom neighborhoods only when requested."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import json
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
        entries, counts = self._cached_file_entries(
            data_cfg,
            entries=entries,
            row_count=len(coords),
        )
        auto_cutoff = PointCloudDataset._resolve_auto_cutoff_config(
            _plain(getattr(data_cfg, "auto_cutoff", None)),
        )

        self.shards: list[_Shard] = []
        offset = 0
        for (source, file_name), count in zip(entries, counts, strict=True):
            radius_override = source["radius_override"]
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
    def _cached_file_entries(
        data_cfg: Any,
        *,
        entries: list[tuple[dict[str, Any], str]],
        row_count: int,
    ) -> tuple[list[tuple[dict[str, Any], str]], list[int]]:
        cache_dir = Path(str(data_cfg.sample_cache.cache_dir)).expanduser().resolve()
        metadata_path = cache_dir / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        shards = metadata["shards"]

        selected_entries: list[tuple[dict[str, Any], str]] = []
        selected_counts: list[int] = []
        remaining = int(row_count)
        for shard_index, shard in enumerate(shards):
            if remaining == 0:
                break
            if shard_index >= len(entries):
                raise ValueError(
                    "Static sample-cache metadata has more shards than configured source files. "
                    f"metadata={metadata_path}, shard_index={shard_index}, entries={len(entries)}."
                )
            source, file_name = entries[shard_index]
            expected_source = str(source["name"])
            if str(shard["source"]) != expected_source or str(shard["file"]) != file_name:
                raise ValueError(
                    "Static sample-cache shard order does not match the configured sources. "
                    f"metadata={metadata_path}, shard_index={shard_index}, "
                    f"cached=({shard['source']!r}, {shard['file']!r}), "
                    f"configured=({expected_source!r}, {file_name!r})."
                )
            count = min(int(shard["count"]), remaining)
            selected_entries.append((source, file_name))
            selected_counts.append(count)
            remaining -= count
        if remaining != 0:
            raise ValueError(
                "Static inference cache has more rows than its sample-cache metadata. "
                f"metadata={metadata_path}, requested_rows={row_count}, "
                f"metadata_rows={int(metadata['total_samples'])}."
            )
        return selected_entries, selected_counts

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
