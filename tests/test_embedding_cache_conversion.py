"""Forecast storage migration preserves timelines, lineage, and resumable provenance."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.conversion.embedding_cache import convert_cache
from src.experiment_runner.registry import sha256, write_json
from src.training_methods.embedding_forecast.data import WindowDataset, verify_cache


def fixture(tmp_path, *, complete=True, value=None):
    root = tmp_path / 'cache'
    shard = root / 'source_000_segment_00'
    shard.mkdir(parents=True)
    values = np.random.default_rng(17).normal(size=(2, 28, 256)).astype(np.float32)
    if value is not None:
        values[0, 0, 0] = value
    np.save(shard / 'embeddings.npy', values)
    np.save(shard / 'atom_ids.npy', np.array([10, 20], dtype=np.int64))
    np.save(shard / 'frames.npy', np.arange(28, dtype=np.int64))
    record = dict(directory=shard.name, centers=2, frames=28, source_index=0,
        preparation_seed=17, split='train', temperature_K=400,
        checksums={p.name: sha256(p) for p in shard.glob('*.npy')})
    protocol = dict(config=dict(cache=str(root), seed=17), producer_sha256='old-producer')
    write_json(shard / 'manifest.json', record)
    write_json(root / 'protocol.json', protocol)
    if complete:
        write_json(root / 'manifest.json', dict(state='complete', protocol=protocol,
            cadence_ps=0.75, embedding_dim=256, shards=[record]))
    config = tmp_path / 'config.json'
    write_json(config, dict(data=dict(protocol['config'], storage_dtype='float16')))
    producer = Path(__file__).parents[1] / 'src/training_methods/embedding_forecast/data.py'
    return root, config, producer, tmp_path / 'audit.json', values, record


@pytest.mark.parametrize('complete', [True, False])
def test_conversion_preserves_metadata_and_resumes_after_manifest_interruption(tmp_path, complete):
    root, config, producer, audit, values, original = fixture(tmp_path, complete=complete)
    result = convert_cache(config, producer, audit)
    shard = root / original['directory']
    stored = np.load(shard / 'embeddings.npy')
    np.testing.assert_array_equal(stored, values.astype(np.float16))
    assert stored.dtype == np.float16
    assert result['original_shards'] == [original]
    for name, digest in original['checksums'].items():
        if name != 'embeddings.npy':
            assert sha256(shard / name) == digest
    if complete:
        manifest = verify_cache(root)
        dataset = WindowDataset(root, manifest, 'train', 6, 9, 0.75, 6)
        batch = dataset[[0, 1]]
        assert batch['history'].numpy().dtype == np.float32
        np.testing.assert_array_equal(batch['history'][0], stored[0, :9].astype(np.float32))
    else:
        assert not (root / 'manifest.json').exists()
    # An interruption after array replacement but before manifest publication is recoverable.
    write_json(shard / 'manifest.json', original)
    write_json(root / 'storage_migration.json', dict(state='failed'))
    assert convert_cache(config, producer, audit)['state'] == 'complete'
    assert json.loads((shard / 'manifest.json').read_text())['checksums']['embeddings.npy'] == sha256(shard / 'embeddings.npy')


@pytest.mark.parametrize('value', [np.inf, 100000.0])
def test_unsafe_embedding_keeps_original_and_blocks_readers(tmp_path, value):
    root, config, producer, audit, _, original = fixture(tmp_path, value=value)
    path = root / original['directory'] / 'embeddings.npy'
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Non-finite or float16-overflowing'):
        convert_cache(config, producer, audit)
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match='migration must finish'):
        verify_cache(root)


def test_scientific_config_change_is_rejected_before_mutation(tmp_path):
    root, config, producer, audit, _, _ = fixture(tmp_path)
    settings = json.loads(config.read_text())
    settings['data']['seed'] = 18
    write_json(config, settings)
    with pytest.raises(ValueError, match='only storage_dtype'):
        convert_cache(config, producer, audit)
    assert not (root / 'storage_migration.json').exists()


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
def test_preparation_publishes_configured_storage_and_reuses_verified_shard(tmp_path, monkeypatch, dtype):
    from types import SimpleNamespace
    import torch
    from src.training_methods.embedding_forecast import data
    from src.utils import model_utils

    source_dir = tmp_path / 'trajectory'
    source_dir.mkdir()
    write_json(source_dir / 'manifest.json', dict(state='complete'))
    campaign = tmp_path / 'campaign.json'
    write_json(campaign, dict(state='complete'))
    selection = tmp_path / 'sources.json'
    write_json(selection, dict(sources=[]))
    checkpoint = tmp_path / 'encoder.pt'
    checkpoint.write_bytes(b'encoder identity')
    (tmp_path / 'encoder.yaml').write_text('encoder: fixture\n')
    source = dict(path=str(source_dir), campaign_manifest=str(campaign), timestep_ps=0.001,
        preparation_seed=17, anchors=[0], name='source', split='train', temperature_K=400)
    trajectory = SimpleNamespace(positions=np.random.default_rng(17).uniform(0, 20, (1, 128, 3)).astype(np.float32),
        box_low=np.zeros((1, 3), dtype=np.float32), box_high=np.full((1, 3), 20, dtype=np.float32),
        timesteps=np.array([0], dtype=np.int64), atom_ids=np.arange(1, 129), atom_count=128, frame_count=1)
    monkeypatch.setattr(data, 'source_records', lambda config: [source])
    monkeypatch.setattr(data.ShootingBinaryTrajectory, 'load', lambda path: trajectory)
    monkeypatch.setattr(model_utils, 'resolve_config_path', lambda path: (str(tmp_path), 'encoder'))
    monkeypatch.setattr(data, 'load_snapshot_encoder',
        lambda path, device: (lambda points: torch.full((len(points), 256), 0.12345678), 9.5))
    config = dict(cache=str(tmp_path / 'prepared'), checkpoint=str(checkpoint), sources_config=str(selection),
        cadence_ps=0.75, history_ps=0, future_ps=0, anchor_margin_ps=0, seed=17,
        centers_per_source=2, encoder_batch_size=2, storage_dtype=dtype)
    manifest = data.prepare_cache(config, 'cpu')
    stored = np.load(Path(config['cache']) / 'source_000_segment_00/embeddings.npy')
    assert stored.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(stored, np.full((2, 1, 256), np.float32(0.12345678), dtype=dtype))
    assert data.prepare_cache(config, 'cpu') == manifest
