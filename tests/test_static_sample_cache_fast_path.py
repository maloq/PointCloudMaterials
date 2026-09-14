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
            multiprocessing_context="spawn" if num_workers else None,
        )
        loader_batch = next(iter(loader))
        torch.testing.assert_close(loader_batch["points"], expected_points)
        torch.testing.assert_close(loader_batch["coords"], expected_coords)


def test_static_datamodule_preserves_cached_split_and_lifecycle(
    tmp_path, monkeypatch
):
    from omegaconf import OmegaConf

    from src.data_utils.data_modules import static

    cache_dir = tmp_path / "cache"
    metadata = _write_cache(cache_dir, return_coords=True)
    dataset = _load_test_dataset(cache_dir, metadata, return_coords=True)
    monkeypatch.setattr(static, "PointCloudDataset", lambda **kwargs: dataset)
    cfg = OmegaConf.create({
        "batch_size": 2,
        "num_workers": 0,
        "max_samples": 0,
        "data": {
            "kind": "static", "split_seed": 42, "train_ratio": 0.6,
            "data_sources": [], "radius": 1.0, "sample_type": "random",
            "num_points": 2,
        },
    })
    dm = static.StaticPointCloudDataModule(cfg, return_coords=True)
    dm.setup("fit")
    assert dm.train_dataset.indices == [1, 6, 3, 5]
    assert dm.val_dataset.indices == [4, 0, 2]
    original_train = dm.train_dataset
    for stage in ("fit", "validate", "test"):
        dm.setup(stage)
        assert dm.train_dataset is original_train
        assert dm.test_dataset is dm.val_dataset
    loader = dm.val_dataloader()
    assert not loader.drop_last
    assert loader.pin_memory
    batch = next(iter(loader))
    for key in ("points", "coords"):
        expected = torch.stack([dataset[i][key] for i in [4, 0]])
        torch.testing.assert_close(batch[key], expected, rtol=0, atol=0)
    assert dm.train_dataloader().drop_last
    assert dm.state_dict() == {}


def test_source_reader_preserves_values_and_rejects_wrong_shapes(tmp_path):
    import pytest

    from src.data.static_sources import load_points

    points = np.array([[1.25, 2.5, 3.75], [4, 5, 6]], dtype=np.float64)
    path = tmp_path / "points.npy"
    np.save(path, points)
    loaded = load_points(str(path))
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, points.astype(np.float32))
    off = tmp_path / "points.off"
    off.write_text("OFF\n2 0 0\n1.25 2.5 3.75\n4 5 6\n")
    np.testing.assert_array_equal(load_points(str(off)), loaded)
    np.save(path, np.zeros((4, 2)))
    with pytest.raises(ValueError, match="Expected.*array.*shape"):
        load_points(str(path))
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_points(str(tmp_path / "points.txt"))


def test_source_cutoff_preserves_pooled_quantile_and_global_rng(tmp_path):
    from src.data.static_sources import estimate_source_cutoff_radius

    points = np.zeros((4, 3), dtype=np.float32)
    points[:, 0] = [0, 1, 3, 7]
    np.save(tmp_path / "a.npy", points)
    np.save(tmp_path / "b.npy", points * 2)
    state = np.random.get_state()
    radius, coverage = estimate_source_cutoff_radius(
        source_root=str(tmp_path), source_files=["a.npy", "b.npy"],
        target_points=2, quantile=0.5, estimation_samples_per_file=4,
        seed=42, safety_factor=1.2, boundary_margin=None,
    )
    assert radius == 2.4
    assert coverage == 0.625
    current = np.random.get_state()
    assert state[0] == current[0]
    np.testing.assert_array_equal(state[1], current[1])
    assert state[2:] == current[2:]
