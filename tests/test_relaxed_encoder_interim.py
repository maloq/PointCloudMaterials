import json

import pytest

from src.data.structural_pretraining.prepare import file_hash, save_json
from src.research.relaxed_encoder.interim import prepare


def test_reuse_assay_preserves_splits_and_requires_finished_checkpoint(tmp_path):
    training, cohort, output = [tmp_path/x for x in ('training', 'cohort', 'output')]
    warm = tmp_path/'parent.pt'
    warm.write_bytes(b'parent checkpoint')
    config = dict(scale=9., assay_plan='original', population='original population',
                  normalization_manifest='normalization', warm_checkpoint=str(warm),
                  cache='cohort cache', runs=[dict(name='cold-control', arm='relaxed_to_relaxed')])
    sources = [dict(id=1, lineage='independent1', source_manifest_sha256='source',
                    center_atom_ids=[1, 2], split='test')]
    plan = dict(identity='training', config=config, sources=sources)
    save_json(training/'technical/plan.json', plan)
    save_json(cohort/'technical/plan.json', dict(plan, identity='cohort'))
    a = cohort/'technical/assay'
    a.mkdir()
    for name in ('population.npz', 'hot-descriptors.npy', 'cold-descriptors.npy',
                 'hot-plan.json', 'cold-plan.json'):
        (a/name).write_bytes(b'immutable assay')
    save_json(a/'ready.json', dict(identity='cohort', frames=[64, 368]))
    for name in ('parent_hot', 'parent_cold'):
        d = a/name
        d.mkdir()
        (d/'features.npy').write_bytes(b'parent features')
        save_json(d/'complete.json', dict(checkpoint_sha256=file_hash(warm),
                                         feature_sha256=file_hash(d/'features.npy')))
        save_json(d/'record.json', dict(population_sha256=file_hash(a/'population.npz')))
    run = training/'technical/runs/cold-control'
    save_json(run/'status.json', dict(state='running'))
    (run/'best.pt').write_bytes(b'new checkpoint')
    recipe = dict(output=str(output), training_output=str(training),
                  assay_output=str(cohort), runs=['cold-control'])
    with pytest.raises(ValueError, match='not complete'):
        prepare(recipe)
    save_json(run/'status.json', dict(state='complete'))
    result = prepare(recipe)
    assert prepare(recipe) == result
    assert result['config']['frames'] == [64, 368]
    assert (output/'technical/assay/population.npz').resolve() == a/'population.npz'
    assert not (output/'technical/assay/parent_cold').is_symlink()
    (output/'technical/assay/parent_cold/normalizer.npz').write_bytes(b'new normalizer')
    assert not (a/'parent_cold/normalizer.npz').exists()
    plan['sources'][0]['split'] = 'train'
    save_json(training/'technical/plan.json', plan)
    with pytest.raises(ValueError, match='split changed'):
        prepare(recipe)
