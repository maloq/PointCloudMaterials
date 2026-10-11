"""Bind Ta full-cell trajectories and exact reusable historical PTM evidence."""
import importlib.metadata
from pathlib import Path

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import dataset_path, resolve_path
from src.postprocessing.protocols import ancestry, external_audit, external_data, extract, fates, shooting_fates
from src.postprocessing.protocols.shooting_data import read


def implementation():
    files = [Path(__file__), Path(__file__).with_name('ta.py'), Path(__file__).with_name('ta_fates.py'),
        Path(ancestry.__file__), Path(external_audit.__file__), Path(external_data.__file__),
        Path(extract.__file__), Path(fates.__file__), Path(shooting_fates.__file__)]
    return dict(files={p.name: sha(p) for p in files},
        libraries={n: importlib.metadata.version(n) for n in ('numpy','scipy','ovito')})


def bind_source(c, path, *, sid, campaign, branch, parent, seed, legacy=None):
    m = read(path / 'manifest.json')
    if m['format'] != 'pointcloudmaterials.temporal_lammps_trajectory' or m['state'] != 'complete':
        raise ValueError(f'Unexpected Ta binary: {path}')
    a = {k: np.load(path / m['arrays'][k]['file'], mmap_mode='r')
         for k in ('positions','timesteps','atom_ids','atom_types','box_low','box_high')}
    for key, value in a.items():
        if list(value.shape) != m['arrays'][key]['shape'] or value.dtype.name != m['arrays'][key]['dtype']:
            raise ValueError(f'Changed Ta array schema: {path}/{key}')
    if a['positions'].shape != (241, parent['atom_count'], 3) or np.any(a['atom_types'] != 1):
        raise ValueError(f'Ta atom count/type mismatch: {path}')
    if np.any(np.diff(a['atom_ids']) <= 0):
        raise ValueError(f'Ta atom IDs must be sorted/unique: {path}')
    actual = a['timesteps'] * .002
    np.testing.assert_allclose(actual, np.arange(241)*.1, rtol=0, atol=1e-8)
    requested = np.arange(49)*c['cadence_ps']
    rows = np.searchsorted(actual, requested)
    np.testing.assert_allclose(actual[rows], requested, rtol=0, atol=1e-8)
    part = dict(path=str(path), manifest_sha256=sha(path/'manifest.json'), timestep_fs=2.,
        storage_dtype=a['positions'].dtype.name,
        array_sha256={k:m['arrays'][k]['sha256'] for k in a},
        quantization=m.get('provenance',{}).get('quantization', 'not_recorded_in_binary_manifest'))
    return dict(id=sid, campaign=campaign, branch_id=branch, material='Ta', potential='ta-zhong2014-eam',
        parent_id=parent['name'], parent_sha256=parent['source_sha256'],
        ancestry_group=c['ancestry_group'], role='audit_only_no_new_split', velocity_seed=seed,
        parts=[part], frames=[[0,int(i)] for i in rows], times_ps=requested.tolist(),
        atom_count=parent['atom_count'], frame_count=len(rows), cadence_ps=c['cadence_ps'],
        center_atom_ids=[int(a['atom_ids'][0])], baseline_center_indices=[0],
        dynamics='isotropic Nose-Hoover NPT', branch_observed_duration_ps=float(requested[-1]),
        phase_boundary=None, legacy=legacy)


def prepare(c):
    old_path = resolve_path(c['legacy_plan'])
    old = read(old_path)
    legacy = [s for s in old['sources'] if s['material']=='Ta']
    if len(legacy) != 6 or old['config']['ptm_rmsd_cutoff'] != c['ptm_rmsd_cutoff']:
        raise ValueError('Changed historical Ta PTM scope/definition')
    launch_path = dataset_path(c['launch_dataset'])/'manifest.json'
    launch = read(launch_path)
    parents = {p['name']:p for p in launch['config']['parents']}
    initial_path = dataset_path(c['legacy_initial_dataset'])/'manifest.json'
    initial = read(initial_path)
    sources = []
    lineages = [s['lineage'] for s in legacy]
    if any(v != lineages[0] for v in lineages):
        raise ValueError('Historical Ta sources used different spatial/ancestry definitions')
    lineage = {k:v for k,v in lineages[0].items() if k!='thresholds'}
    lineage['thresholds'] = [next(t for t in lineages[0]['thresholds'] if t['name']=='primary')]
    if lineage['thresholds'][0] != dict(name='primary',size=64,persistence_ps=1.5,persistence_frames=4):
        raise ValueError('Historical Ta establishment definition differs')
    for item in legacy:
        path = Path(item['parts'][0]['path'])
        name = path.parent.name
        outcome_path = path.parent/'outcome.json'
        outcome = read(outcome_path)
        if outcome['state'] != 'complete':
            raise ValueError(f'Incomplete historical Ta source: {outcome_path}')
        if name == 'model_1m':
            parent_hash, seed = initial['source']['sha256'], initial['protocol']['velocity_seed']
            potential = initial['potential']['sha256']
            campaign = c['legacy_initial_dataset']
        else:
            parent_hash, seed = outcome['origin']['source_sha256'], outcome['origin']['velocity_seed']
            potential = outcome['potential'][0]['sha256']
            campaign = c['legacy_large_dataset']
        if parent_hash != parents[name]['source_sha256'] or potential != c['potential_sha256']:
            raise ValueError(f'Historical Ta parent/potential mismatch: {path}')
        folder = resolve_path(old['config']['output'])/'technical/sources'/item['id']
        reused = dict(folder=str(folder), plan_identity=old['identity'], source_id=item['id'],
            graph_receipt=read(folder/'graph-complete.json'), ptm_receipt=read(folder/'ptm-complete.json'),
            chunk_frames=old['config']['chunk_frames'], outcome_sha256=sha(outcome_path))
        s = bind_source(c, path, sid='ta-old-'+item['id'], campaign=campaign,
                        branch=name, parent=parents[name], seed=seed, legacy=reused)
        if (s['frames'] != item['frames'] or s['times_ps'] != item['times_ps'] or
                s['parts'][0]['manifest_sha256'] != item['parts'][0]['manifest_sha256']):
            raise ValueError(f'Historical Ta PTM cannot be reused on different observations: {path}')
        # Keep the historical graph's center-readout array identities aligned.
        # Event discovery itself scans every component in the full cell.
        s['center_atom_ids']=item['center_atom_ids']
        s['baseline_center_indices']=item['baseline_center_indices']
        sources.append(s)
    outcomes = []
    for run in launch['runs']:
        root = dataset_path(run['run_id'])
        for branch in run['branches']:
            folder = root/'branches'/branch['name']
            o = read(folder/'outcome.json')
            if (o['state'] != 'complete' or o['origin'] != branch or o['timestep_ps'] != .002
                    or o['potential'][0]['sha256'] != c['potential_sha256']):
                raise ValueError(f'Changed/incomplete Ta shooting outcome: {folder}')
            s = bind_source(c,folder/'trajectory_binary_float16',sid=run['run_id']+'-'+branch['name'],
                campaign=run['run_id'],branch=branch['name'],parent=run['parent'],seed=branch['velocity_seed'])
            sources.append(s)
            outcomes.append(dict(source=s['id'], outcome_sha256=sha(folder/'outcome.json')))
    if len(sources)!=30 or len({(s['parent_sha256'],s['velocity_seed']) for s in sources})!=30:
        raise ValueError('Require exactly 30 unique Ta futures with distinct parent/seed identities')
    loads = [0]*c['lanes']
    for s in sorted(sources,key=lambda s:(-s['atom_count'],s['id'])):
        s['lineage'] = lineage
        s['lane'] = min(range(c['lanes']),key=loads.__getitem__)
        loads[s['lane']] += s['atom_count'] * (1 if s['legacy'] else s['frame_count'])
    value = dict(config=c,sources=sources,implementation=implementation(),
        legacy_plan_sha256=sha(old_path), launch_manifest_sha256=sha(launch_path),
        initial_manifest_sha256=sha(initial_path), outcomes=outcomes, scales=old['scales'],
        lane_cost_units=loads,excluded=[dict(dataset='Ta',reason='Static archived structures lack a trajectory; six parents already enter through their MD descendants')])
    value['identity']=digest(value)
    dest=resolve_path(c['output'])/'technical/plan.json'
    if dest.exists() and read(dest)!=value:
        raise ValueError('Ta audit plan changed; use a new revision')
    write_json(dest,value)
    write_json(dest.with_name('inventory.json'),dict(futures=30,parents=6,ancestry_groups=1,
        historical_reused=6,new_full_cell_searches=24,observation_frames=1470,new_ptm_frames=1176,
        cadence_ps=c['cadence_ps'],duration_ps=24,atom_counts=sorted({s['atom_count'] for s in sources})))
    return value


def load(c):
    p=read(resolve_path(c['output'])/'technical/plan.json')
    if (digest({k:v for k,v in p.items() if k!='identity'})!=p['identity'] or
            p['config']!=c or p['implementation']!=implementation()):
        raise ValueError('Changed Ta configuration/producer; use frozen source')
    return p
