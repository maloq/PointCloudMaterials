"""Independent-preparation learning curves from actual local atom clouds."""
from collections import Counter
from pathlib import Path
import time

import numpy as np

from src.experiment_runner.registry import sha256, write_json
from src.research.mace_local_state.motion_data import atomic_npz, read_sequence
from .data import local_clouds
from .inventory import read


def nested_selection(records, counts, seeds, core_seed):
    """A common smallest core and nested temperature-balanced extensions."""
    sources = [r['source'] for r in records]
    if len({s['lineage'] for s in sources}) != len(sources):
        raise ValueError('Learning-curve units must have distinct preparation lineages')
    train = [s for s in sources if s['split'] == 'train']
    temperatures = sorted({s['temperature_K'] for s in train})
    if sorted(counts) != counts or any(n % len(temperatures) for n in counts):
        raise ValueError('Counts must increase and balance temperatures exactly')
    groups = {t: np.array(sorted(s['id'] for s in train if s['temperature_K'] == t))
              for t in temperatures}
    if any(len(ids) != counts[-1]//len(temperatures) for ids in groups.values()):
        raise ValueError(f'Largest subset must equal the balanced training population: {groups}')
    rng = np.random.default_rng(core_seed)
    core = {t: rng.choice(ids, counts[0]//len(temperatures), replace=False)
            for t, ids in groups.items()}
    fits = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = {t: np.r_[core[t], rng.permutation(np.setdiff1d(ids, core[t]))]
                 for t, ids in groups.items()}
        for count in counts:
            selected = sorted(int(i) for t in temperatures for i in order[t][:count//len(temperatures)])
            fits.append(dict(seed=seed, count=count, source_ids=selected))
    return dict(core_source_ids=sorted(int(i) for ids in core.values() for i in ids),
                fits=fits, validation_source_ids=sorted(s['id'] for s in sources if s['split']=='val'),
                development_test_source_ids=sorted(s['id'] for s in sources if s['split']=='test'))


def prepare(config):
    root = Path(config['output'])/'technical'; root.mkdir(parents=True, exist_ok=True)
    cache = Path(config['cache']); cache.mkdir(parents=True, exist_ok=True)
    previous = Path(config['sequence_cache'])
    records = [r for r in read(previous/'plan.json')['records']
               if r['source']['lineage'].startswith('independent_melt')]
    splits = Counter(r['source']['split'] for r in records)
    if splits != {'train':90, 'val':30, 'test':30}:
        raise ValueError(f'Changed independent-preparation population: {splits}')
    selection = nested_selection(records, config['training_sources'], config['seeds'], config['core_seed'])
    specification = dict(protocol=config['protocol'], selection=selection, records=records,
                         sequence_plan_sha256=sha256(previous/'plan.json'),
                         candidate_radius_A=config['candidate_radius_A'],
                         extraction_sha256=sha256(Path(__file__)))
    if (cache/'plan.json').exists() and read(cache/'plan.json') != specification:
        raise ValueError('Changed cloud-cache selection or extraction; use a new cache')
    write_json(cache/'plan.json', specification)
    write_json(root/'data-plan.json', specification)
    units = []; start = time.monotonic()
    for number, record in enumerate(records):
        sid = record['source']['id']; path = cache/f'source-{sid:04d}.npz'
        old = previous/path.name; stamp = read(old.with_suffix('.json'))
        if stamp['record'] != record or sha256(old) != stamp['sha256']:
            raise ValueError(f'Changed label/sequence provenance: {old}')
        if path.with_suffix('.json').exists():
            saved = read(path.with_suffix('.json'))
            if saved['label_sha256'] != stamp['sha256'] or sha256(path) != saved['sha256']:
                raise ValueError(f'Changed completed cloud unit: {path}')
            units.append(saved); continue
        p,v,lengths,times,identity = read_sequence(record, config)
        if identity != stamp['source_identity']:
            raise ValueError(f'Positions/velocities differ from those that produced labels: {sid}')
        with np.load(old) as a:
            target = a['raw_target'].copy()
            np.testing.assert_array_equal(a['center_atom_id'],record['center_atom_ids'])
            np.testing.assert_array_equal(a['time_ps'],np.tile(times,(4,1)))
        if target.shape != (4,9,169) or not np.allclose(np.diff(times),.75,rtol=0,atol=1e-9):
            raise ValueError(f'Unexpected held-fixed sampling: {sid}, {target.shape}, {times}')
        centers = np.asarray(record['center_atom_ids'])-1
        clouds = [cloud for f in range(9) for cloud in local_clouds(
            p[f],v[f],lengths[f],centers,config['candidate_radius_A'])]
        atomic_npz(path, positions=np.concatenate([x for x,v in clouds]),
                   velocities=np.concatenate([v for x,v in clouds]),
                   pointers=np.cumsum([0]+[len(x) for x,v in clouds]),
                   raw_target=target, time_ps=times)
        saved = dict(source_id=sid, sha256=sha256(path), label_sha256=stamp['sha256'],
                     source_identity=identity)
        write_json(path.with_suffix('.json'),saved); units.append(saved)
        status = dict(state='preparing', completed=number+1,total=len(records),
                      elapsed_seconds=time.monotonic()-start)
        write_json(root/'prepare-status.json',status)
        print('DATA AMOUNT PREPARE',status,flush=True)
    write_json(cache/'complete.json',dict(state='complete',plan_sha256=sha256(cache/'plan.json'),units=units))
    write_json(root/'prepare-status.json',dict(state='complete',sources=len(records),clouds=36*len(records)))


def load(config):
    cache = Path(config['cache']); plan = read(cache/'plan.json'); done = read(cache/'complete.json')
    if done['state'] != 'complete' or done['plan_sha256'] != sha256(cache/'plan.json'):
        raise ValueError('Incomplete/changed atom-cloud cache')
    if plan['selection'] != nested_selection(plan['records'],config['training_sources'],config['seeds'],config['core_seed']):
        raise ValueError('Config no longer matches retained subset selection')
    sources = {}
    for unit in done['units']:
        path = cache/f"source-{unit['source_id']:04d}.npz"
        if sha256(path) != unit['sha256']: raise ValueError(f'Changed atom clouds: {path}')
        with np.load(path) as a:
            p,v,ptr = a['positions'],a['velocities'],a['pointers']
            # Producer is time-major; trainer uses center-major trajectories.
            clouds = [(p[i:j],v[i:j]) for i,j in zip(ptr[:-1],ptr[1:],strict=True)]
            sources[unit['source_id']] = dict(clouds=[[clouds[t*4+c] for t in range(9)] for c in range(4)],
                raw_target=a['raw_target'].copy(),time_ps=a['time_ps'].copy())
    return plan,sources


def batch(sources, source_ids, starts=None):
    clouds=[]; targets=[]; times=[]
    for i,sid in enumerate(source_ids):
        source=sources[int(sid)]; frames=range(9) if starts is None else range(int(starts[i]),int(starts[i])+3)
        clouds.extend(source['clouds'][c][t] for c in range(4) for t in frames)
        targets.append(source['raw_target'][:,list(frames)])
        times.append(np.tile(source['time_ps'][list(frames)],(4,1)))
    return clouds,np.concatenate(targets),np.concatenate(times)
