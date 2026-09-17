"""A seed replicate compares the same windows without requiring another modality."""
import json
import pytest
import torch

from src.training_methods.predictive_memory.compare import compare


def completed_fits(root, modalities, steps=3000):
    for modality in modalities:
        for suffix, score in [('H0', 1.), ('H12', .9), ('H48', .8), ('H48-repeat', .85)]:
            technical = root/f'{modality}-{suffix}'/'technical'
            technical.mkdir(parents=True)
            (technical/'status.json').write_text(json.dumps(dict(state='complete', step=steps)))
            (technical/'metrics.json').write_text(json.dumps(dict(selected_step=2500, batch_size=8, trained_windows=steps*8,
                test=dict(joint_nll=dict(mean=score), future_mse=dict(mean=score)))))
            rows = [dict(source_id=s, center_id=10, anchor=a, joint_nll=score)
                    for s in (1, 2) for a in (399, 400, 401)]
            torch.save(dict(test=dict(rows=rows)), technical/'evaluation.pt')


@pytest.mark.parametrize('modalities', [('xv',), ('x', 'xv')])
@pytest.mark.parametrize('steps', [3000, 12000])
def test_complete_paired_comparison_for_selected_modalities(tmp_path, modalities, steps):
    completed_fits(tmp_path, modalities, steps)
    config = dict(output=str(tmp_path), training=dict(steps=steps, batch_size=8), bootstrap_draws=50, seed=18)
    compare(config, modalities=modalities)
    root = tmp_path/'comparison'
    result = json.loads((root/'technical/metrics.json').read_text())
    assert len(result['models']) == 4*len(modalities)
    assert result['paired_test']['xv_H48_gain_over_snapshot']['mean'] == pytest.approx(.2)
    assert result['paired_test']['xv_H48_gain_over_repeated_anchor']['mean'] == pytest.approx(.05)
    assert (root/'tables/METRICS.md').is_file()
    assert json.loads((root/'technical/status.json').read_text()) == dict(state='complete', fits=4*len(modalities))
    assert f'{steps:,}-update budget' in (root/'README.md').read_text()
    if len(modalities) == 1:
        assert not any(name.startswith('x-') for name in result['models'])


def test_replicate_rejects_unmatched_windows_and_incomplete_budgets(tmp_path):
    completed_fits(tmp_path, ('xv',))
    config = dict(output=str(tmp_path), training=dict(steps=3000, batch_size=8), bootstrap_draws=50, seed=18)
    status = tmp_path/'xv-H48/technical/status.json'
    status.write_text(json.dumps(dict(state='complete', step=2000)))
    with pytest.raises(RuntimeError, match='equal-budget'):
        compare(config, modalities=('xv',))
    status.write_text(json.dumps(dict(state='complete', step=3000)))
    path = tmp_path/'xv-H48/technical/evaluation.pt'
    payload = torch.load(path, weights_only=True)
    payload['test']['rows'][0]['center_id'] = 11
    torch.save(payload, path)
    with pytest.raises(ValueError, match='source/center/anchor'):
        compare(config, modalities=('xv',))


def test_comparison_rejects_larger_batches_at_the_same_update_count(tmp_path):
    completed_fits(tmp_path, ('xv',))
    path = tmp_path/'xv-H12/technical/metrics.json'
    metrics = json.loads(path.read_text())
    metrics.update(batch_size=16,trained_windows=48000)
    path.write_text(json.dumps(metrics))
    config = dict(output=str(tmp_path),training=dict(steps=3000,batch_size=8),bootstrap_draws=50,seed=18)
    with pytest.raises(ValueError,match='effective batches'):
        compare(config,modalities=('xv',))
