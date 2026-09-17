"""Scientific pairing, confirmation and budget constraints of the planned queue."""
import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from src.research.local_predictability.plan import build_queue


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / 'configs/local_predictability/two_gpu_16h.json').read_text())


def test_confirmation_and_paired_sampling(config):
    queue = build_queue(config)
    origins = queue['primary_common_origins_ps']
    assert len(origins) == 151
    assert origins[0] == 48
    assert origins[-1] == 498
    assert origins[-1] + 96 + queue['confirmation_padding_ps'] == 600
    assert queue['population_candidate_windows'] == 362400
    assert queue['native_candidate_windows'] == 38400


def test_longer_primary_confirmation_is_not_ignored(config):
    config['assay']['persistence_frames'] = 13
    queue = build_queue(config)
    assert queue['confirmation_padding_ps'] == 9
    assert queue['primary_common_origins_ps'][-1] == 495


def test_one_seed_and_equal_continuation_opportunity(config):
    queue = build_queue(config)
    assert len(queue['jobs']) == 29
    parents = [j for j in queue['jobs'] if j['kind'] == 'native_parent']
    children = [j for j in queue['jobs'] if j['kind'] == 'native_continuation']
    assert len(parents) == 2 and len(children) == 6
    assert {j['seed'] for j in queue['jobs'] if 'seed' in j} == {20260919}
    for parent in parents:
        group = [j for j in children if j['depends_on'] == [parent['id']]]
        assert {j['name'] for j in group} == {'snapshot', 'history12', 'repeat12'}
        assert len({j['updates'] for j in group}) == 1
        assert {j['worker'] for j in group} == {parent['worker']}
    assert not queue['optional_enabled']


@pytest.mark.parametrize('mutation,match', [
    ('extra_seed', 'one shared training seed'),
    ('missing_control', 'separately trained repeated frames'),
    ('duration_mismatch', 'match duration'),
    ('short_reserve', 'final hour'),
])
def test_reject_invalid_comparisons(config, mutation, match):
    changed = copy.deepcopy(config)
    if mutation == 'extra_seed':
        changed['native']['seeds'].append(20260920)
    elif mutation == 'missing_control':
        changed['native']['variants'].pop()
    elif mutation == 'duration_mismatch':
        changed['native']['variants'][-1]['history_ps'] = 48
    elif mutation == 'short_reserve':
        changed['phases'][-2]['end_hour'] = 15.5
        changed['phases'][-1]['start_hour'] = 15.5
    with pytest.raises(ValueError, match=match):
        build_queue(changed)


def test_portable_manifest_preserves_independent_folds(config):
    manifest = json.loads((ROOT / config['source_manifest']).read_text())
    sources = manifest['sources']
    assert len(sources) == len({s['lineage'] for s in sources}) == 150
    assert Counter(s['split'] for s in sources) == config['sampling']['source_counts']
    for temperature in config['sampling']['temperatures_K']:
        validation = [s for s in sources if s['split'] == 'val' and s['temperature_K'] == temperature]
        assert Counter(s['validation_role'] for s in validation) == {'selection': 3, 'calibration': 3}
    for source in sources:
        assert source['frame_count'] == 801
        assert source['relative_trajectory_path']
        assert not Path(source['relative_trajectory_path']).is_absolute()
        assert len(source['manifest_sha256']) == 64
