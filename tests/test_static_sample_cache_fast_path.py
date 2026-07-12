import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data_utils.data_load import PointCloudDataset


def _write_cache(cache_dir: Path, *, return_coords: bool) -> dict:
    shards_dir = cache_dir / "shards"
    shards_dir.mkdir(parents=True)
    shard_specs = (("a", 3, 0.0), ("b", 4, 100.0))
    shards = []
    total_samples = 0
    for name, count, offset in shard_specs:
        points = np.arange(count * 2 * 3, dtype=np.float32).reshape(count, 2, 3) + offset
        samples_path = f"shards/{name}.samples.npy"
        np.save(cache_dir / samples_path, points)
        coords_path = None
        if return_coords:
            coords = np.arange(count * 3, dtype=np.float32).reshape(count, 3) + offset
            coords_path = f"shards/{name}.coords.npy"
            np.save(cache_dir / coords_path, coords)
        shards.append(
            {
                "source": name,
                "file": f"{name}.npy",
                "samples_path": samples_path,
                "coords_path": coords_path,
                "count": count,
                "radius": 1.0,
            }
        )
        total_samples += count

    metadata = {
        "schema_version": 1,
        "fingerprint": "test-fingerprint",
        "request": {"num_points": 2, "return_coords": return_coords},
        "shards": shards,
        "total_samples": total_samples,
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def _load_test_dataset(cache_dir: Path, metadata: dict, *, return_coords: bool):
    dataset = PointCloudDataset.__new__(PointCloudDataset)
    dataset.num_points = 2
    dataset.return_coords = return_coords
    dataset.source_radii = {}
    dataset._load_sample_cache_from_metadata(cache_dir=cache_dir, metadata=metadata)
    return dataset


def test_node_local_staging_reuses_valid_copy(tmp_path):
    source_dir = tmp_path / "source"
    staged_dir = tmp_path / "local" / "cache"
    metadata = _write_cache(source_dir, return_coords=False)
    dataset = PointCloudDataset.__new__(PointCloudDataset)

    result_dir, result_metadata = dataset._stage_sample_cache_if_configured(
        source_cache_dir=source_dir,
        metadata=metadata,
        cache_cfg={"local_cache_dir": str(staged_dir)},
    )
    assert result_dir == staged_dir
    assert result_metadata["fingerprint"] == metadata["fingerprint"]

    first_mtime = (staged_dir / "metadata.json").stat().st_mtime_ns
    result_dir, _ = dataset._stage_sample_cache_if_configured(
        source_cache_dir=source_dir,
        metadata=metadata,
        cache_cfg={"local_cache_dir": str(staged_dir)},
    )
    assert result_dir == staged_dir
    assert (staged_dir / "metadata.json").stat().st_mtime_ns == first_mtime


def test_bulk_cache_reads_match_individual_reads_and_dataloader(tmp_path):
    cache_dir = tmp_path / "cache"
    metadata = _write_cache(cache_dir, return_coords=True)
    dataset = _load_test_dataset(cache_dir, metadata, return_coords=True)
    indices = [5, 0, 3, 2, 5, -1]

    expected_points = torch.stack([dataset[index]["points"] for index in indices])
    expected_coords = torch.stack([dataset[index]["coords"] for index in indices])
    bulk_samples = dataset.__getitems__(indices)
    torch.testing.assert_close(
        torch.stack([sample["points"] for sample in bulk_samples]),
        expected_points,
    )
    torch.testing.assert_close(
        torch.stack([sample["coords"] for sample in bulk_samples]),
        expected_coords,
    )

    subset = Subset(dataset, indices)
    for num_workers in (0, 2):
        loader = DataLoader(
            subset,
            batch_size=len(indices),
            num_workers=num_workers,
        )
        loader_batch = next(iter(loader))
        torch.testing.assert_close(loader_batch["points"], expected_points)
        torch.testing.assert_close(loader_batch["coords"], expected_coords)
