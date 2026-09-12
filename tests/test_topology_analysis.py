"""The standard collector sees actual histories; probe fitting excludes held-out labels."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from src.analysis.topology import run_topology_analysis
from src.analysis.topology_dataset import RelaxedTopologyAnalysisDataset, topology_dataloader
from src.analysis.utils import gather_inference_batches
from src.data_utils.topology_targets import fit_targets


class HistoryReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.tda_head = None

    @property
    def device(self):
        return self.weight.device

    def forward(self, points):
        assert points.ndim == 4 and points.shape[1] == 5
        weights = torch.arange(1, 6, dtype=points.dtype)[None, :, None, None]
        latent = (points * weights).mean((1, 2)) * self.weight
        return latent, latent, None


def prepared_data(tmp_path, monkeypatch):
    rng = np.random.default_rng(193)
    shards, originals = [], []
    for context, (split, count) in enumerate((('train', 32), ('val', 8), ('test', 8))):
        views = rng.normal(size=(count, 3, 5, 80, 3)).astype(np.float16)
        name = f'{context}.npy'
        np.save(tmp_path/name, views)
        shards.append(dict(views=name, split=split, samples=count, source=context,
                           frame=40, temperature_K=400+50*context))
        origin = tmp_path/str(context)
        origin.mkdir()
        np.save(origin/'centers.npy', np.arange(count))
        originals.append(dict(directory=str(origin), frame=40,
                              provenance=dict(source=dict(path=str(origin)))))
    targets = rng.normal(size=(48, 144)).astype(np.float32)
    np.save(tmp_path/'targets.npy', targets)
    np.savez(tmp_path/'scaling.npz', **fit_targets(targets[:32], 32, .05))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(state='complete', shards=shards)))
    (tmp_path/'original.json').write_text(json.dumps(dict(shards=originals)))
    positions = rng.normal(size=(41, 32, 3)).astype(np.float32)
    monkeypatch.setattr('src.analysis.topology_dataset.ShootingBinaryTrajectory.load',
                        lambda path:SimpleNamespace(root=Path(path), positions=positions,
                                                    atom_ids=np.arange(100,132)))
    cfg = OmegaConf.create(dict(model_type='vicreg', batch_size=8, experiment_name='test',
        seed_everything=193, encoder=dict(name='PretrainedMACEHistoryGeometry'), tda=dict(target='blocks'),
        data=dict(kind='relaxed_histories', cache_dir=str(tmp_path),
                  relaxed_manifest=str(tmp_path/'original.json'), input_mode='history',
                  normalization_radius_A=9.192189)))
    return cfg


def test_standard_collector_keeps_history_and_anchor_identity(tmp_path, monkeypatch):
    cfg = prepared_data(tmp_path, monkeypatch)
    dataset = RelaxedTopologyAnalysisDataset(cfg, 'test')
    item = dataset[0]
    torch.testing.assert_close(item['points'], item['model_input'][-1])
    assert item['instance_id'] == 100
    assert len(np.unique(dataset.atom_ids)) == len(dataset)
    model = HistoryReadout().eval()
    result = gather_inference_batches(model, topology_dataloader(dataset, 3, 0), 'cpu',
                                      max_batches=None, collect_coords=True,
                                      temporal_sequence_mode='temporal')
    expected = model(torch.stack([dataset[i]['model_input'] for i in range(len(dataset))]))[0]
    np.testing.assert_allclose(result['inv_latents'], expected.detach().numpy(), atol=1e-7)
    np.testing.assert_array_equal(result['instance_ids'], dataset.atom_ids)
    changed = RelaxedTopologyAnalysisDataset(cfg, 'test', 'repeat_anchor')[0]
    torch.testing.assert_close(changed['model_input'], item['points'].expand(5,-1,-1))
    reversed_item = RelaxedTopologyAnalysisDataset(cfg, 'test', 'reverse_past')[0]
    torch.testing.assert_close(reversed_item['model_input'][-1], item['points'])
    torch.testing.assert_close(reversed_item['model_input'][:-1], item['model_input'][:-1].flip(0))


def test_pipeline_stage_probe_predictions_do_not_use_held_out_targets(tmp_path, monkeypatch):
    cfg = prepared_data(tmp_path, monkeypatch)
    settings = OmegaConf.create(dict(runtime=dict(seed_base=193), topology=dict(enabled=True,
        batch_size=8, num_workers=0, ridge_alpha=1., history_interventions=['repeat_anchor','reverse_past'],
        data=OmegaConf.to_container(cfg.data))))
    # This exercises the stage attached to a separate static main analysis.
    cfg.data.kind = 'static'
    checkpoint = tmp_path/'model.ckpt'
    checkpoint.write_bytes(b'test checkpoint fingerprint')
    model = HistoryReadout().eval()
    predictions = []
    for attempt in range(2):
        out = tmp_path/f'analysis_{attempt}'
        out.mkdir()
        metrics = run_topology_analysis(model=model, cfg=cfg, analysis_cfg=settings,
            checkpoint_path=str(checkpoint), out_dir=out, main_cache={}, step=lambda _:None)
        with np.load(out/'topology/test_predictions.npz') as saved:
            predictions.append(saved['ridge_predictions'])
        assert metrics['sample_counts'] == dict(train=32,val=8,test=8)
        assert set(metrics['history_interventions']) == {'repeat_anchor','reverse_past'}
        targets = np.load(tmp_path/'targets.npy')
        targets[32:] += 100
        np.save(tmp_path/'targets.npy', targets)
    np.testing.assert_array_equal(predictions[0], predictions[1])
