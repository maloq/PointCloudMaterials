"""Check source weighting and the scientific pairing of context comparisons."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.research.forecast_context.compare import compare, load_groups, source_means, summarize


def test_context_summary_pairs_sources_after_seed_averaging():
    sources = np.array([0, 0, 1])
    np.testing.assert_array_equal(source_means(np.array([1., 1., 9.]), sources), [1., 9.])
    groups = []
    for history, factor in [(0, 1.), (6, .5)]:
        mse = np.array([[1., 9.], [3., 11.]]) * factor
        measures = {k: mse for k in ('mse', 'raw_mse', 'increment_mse', 'persistence_mse',
                                    'history_mean_mse', 'reverse_past_mse', 'repeat_anchor_mse')}
        measures['bin_mse'] = np.repeat(mse[:, :, None], 3, axis=2)
        measures['mse_by_step'] = np.repeat(mse[:, :, None], 12, axis=2)
        groups.append(dict(architecture='autoregressive_gru', history_ps=history, history_frames=1,
                           parameters=10, windows=3, source_ids=np.array([0, 1]), measures=measures))
    rows, _ = summarize(groups, seed=1, repetitions=100)
    assert rows[0]['mse'] == 6. and rows[0]['seed_std'] == 1.
    assert rows[1]['mse'] == 3. and rows[1]['gain_vs_anchor'] == .5
    assert rows[1]['gain_vs_anchor_ci95_lower'] == .5
    assert rows[1]['gain_vs_anchor_ci95_upper'] == .5


@pytest.fixture
def completed_contexts(tmp_path):
    entries = []
    for history in (0, 6):
        variants = [dict(name=f'{architecture}_h{history}', architecture=architecture) for architecture in
                    ('autoregressive_gru', 'mean_residual_gru')]
        config = dict(output=str(tmp_path / 'technical/fits'), data={}, anchor_history_ps=24,
                      stride_ps=24, horizons_ps=[3, 6, 9], training={}, seeds=[1, 2],
                      history_ps=history, variants=variants)
        path = tmp_path / f'config{history}.json'
        path.write_text(json.dumps(config))
        entries.append(dict(history_ps=history, path=str(path)))
        for variant in variants:
            for seed in config['seeds']:
                directory = tmp_path / 'technical/fits' / f'{variant["name"]}-seed{seed}' / 'technical'
                directory.mkdir(parents=True)
                (directory / 'status.json').write_text(json.dumps(dict(state='complete')))
                (directory / 'config.json').write_text(json.dumps(dict(config=config, variant=variant, seed=seed)))
                torch.save(dict(mean=torch.zeros(2), scale=torch.ones(2), cache_manifest_sha256='fixed',
                                implementation_sha256={'run.py': 'fixed'}, epoch=0), directory / 'best.pt')
                (directory / 'data_summary.json').write_text(json.dumps(dict(history_steps=1+int(history/.75), parameters=10)))
                mse = np.array([1., 1., 9.]) * (1. if history == 0 else .5)
                values = {k: mse for k in ('mse', 'raw_mse', 'increment_mse', 'persistence_mse', 'history_mean_mse')}
                values.update(source=np.array([0, 0, 1]), atom_id=np.array([1, 2, 3]),
                              anchor_frame=np.array([32, 32, 32]), temperature_K=np.array([500., 500., 500.]),
                              bin_mse=np.repeat(mse[:, None], 3, axis=1), mse_by_step=np.repeat(mse[:, None], 12, axis=1))
                np.savez(directory / 'test_errors.npz', **values)
                intervention = dict(per_source={'0': {'mse': 1.}, '1': {'mse': 9.}})
                (directory / 'history_interventions.json').write_text(json.dumps({name: intervention for name in ('reverse_past', 'repeat_anchor')}))
    plan = dict(configs=entries, seeds=[1, 2], output=str(tmp_path))
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    return path, plan


def test_context_export_preserves_pairing_and_documents_metrics(completed_contexts):
    path, plan = completed_contexts
    compare(path)
    root = path.parent
    with (root / 'tables/context-quality.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and float(rows[2]['gain_vs_anchor']) == .5
    assert (root / 'tables/METRICS.md').is_file()
    assert (root / 'plots/context-quality.png').is_file()
    assert json.loads((root / 'technical/comparison_status.json').read_text()) == dict(state='complete', fits=8)


@pytest.mark.parametrize('changed', ['identity', 'scale'])
def test_context_comparison_rejects_unpaired_or_renormalized_fits(completed_contexts, changed):
    _, plan = completed_contexts
    config = json.loads(Path(plan['configs'][1]['path']).read_text())
    directory = Path(config['output']) / f'{config["variants"][0]["name"]}-seed1/technical'
    if changed == 'identity':
        with np.load(directory / 'test_errors.npz') as archive:
            rows = {k: archive[k] for k in archive.files}
        rows['anchor_frame'][0] += 1
        np.savez(directory / 'test_errors.npz', **rows)
        with pytest.raises(AssertionError, match='Unpaired anchor_frame'):
            load_groups(plan)
    else:
        checkpoint = torch.load(directory / 'best.pt', weights_only=False)
        checkpoint['scale'] *= 2
        torch.save(checkpoint, directory / 'best.pt')
        with pytest.raises(ValueError, match='same training normalization'):
            load_groups(plan)
