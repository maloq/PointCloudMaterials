"""Native 0.10 ps Al birth search and six-frame crystal-free descriptor cohort."""
from collections import Counter

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.data.structural_pretraining.prepare import source_arrays
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import dataset_path, resolve_path
from src.postprocessing.protocols import external_data
from src.postprocessing.protocols import shooting_fates
from . import shooting_appearance_data as histories
from .data import read

FAMILY = 'birth_dense010'


def prepare(c):
    """Freeze only the explicitly listed completed descendants, never old birth labels."""
    if c['cadence_ps'] != .1 or c['offsets_ps'] != [-.6, -.5, -.4, -.3, -.2, -.1]:
        raise ValueError('This release requires six exact 0.10 ps observations ending 0.10 ps before appearance')
    implementation = check_metric_docs(family=FAMILY)[FAMILY]['files']
    sources, tasks = [], []
    loads = [0] * c['ptm_lanes']
    for index, sid in enumerate(c['source_ids']):
        name = c['source_prefix'] + f'{sid:04d}'
        root = dataset_path(name)
        meta, outcome = read(root / 'input_metadata.json'), read(root / 'outcome.json')
        path = root / 'trajectory_binary_float16'
        manifest = read(path / 'manifest.json')
        if (outcome['state'] != 'complete' or meta['protocol'] != 'al_dense_replay_v1'
                or meta['source_id'] != sid or meta['sampling_ps'] != .1
                or meta['timestep_ps'] != .002 or manifest['state'] != 'complete'):
            raise ValueError(f'Not a completed native 0.10 ps descendant: {root}')
        part = dict(path=str(path), manifest_sha256=sha(path/'manifest.json'), timestep_fs=2.,
                    array_sha256={k: manifest['arrays'][k]['sha256'] for k in
                        ('positions', 'atom_ids', 'atom_types', 'timesteps', 'box_low', 'box_high')})
        a = source_arrays(dict(part, kind='dynamic'))
        times = a['timesteps'].astype(float) * meta['timestep_ps']
        if (times[0] != 0 or not np.allclose(np.diff(times), .1, rtol=0, atol=1e-8)
                or len(times) != outcome['frame_count'] or np.any(np.diff(a['atom_ids']) <= 0)):
            raise ValueError(f'Invalid actual dense timeline/identities: {path}')
        source = dict(id='md01-' + digest(name)[:16], source_id=sid, dataset=name,
            campaign='al-main-010ps-20260927', branch_id=name, parent_id=str(sid),
            parent_sha256=meta['parent_input_sha256']['prepared_liquid.lammps.data'],
            ancestry_group=meta['root_lineage'], role=meta['split'], material='Al',
            potential='al-lee2003-meam', temperature_K=meta['temperature_K'],
            parts=[part], frames=[[0, i] for i in range(len(times))], times_ps=times.tolist(),
            atom_count=len(a['atom_ids']), frame_count=len(times), cadence_ps=.1,
            center_atom_ids=[int(a['atom_ids'][0])], baseline_center_indices=[0],
            lineage=c['lineage'], dynamics='NPT dense replay; fresh 2 fs integration',
            branch_observed_duration_ps=float(times[-1]), lane=index % c['lanes'],
            provenance=dict(metadata_sha256=sha(root/'input_metadata.json'),
                            outcome_sha256=sha(root/'outcome.json')))
        sources.append(source)
        for start in range(0, len(times), c['chunk_frames']):
            stop = min(start+c['chunk_frames'], len(times))
            lane = min(range(c['ptm_lanes']), key=loads.__getitem__)
            size = (stop-start) * source['atom_count']; loads[lane] += size
            tasks.append(dict(source=source['id'], start=start, stop=stop, lane=lane, atom_frames=size))
    recipe = read(resolve_path(c['readout_recipe']))
    plan = dict(config=c, sources=sources, tasks=tasks, implementation=implementation,
        recipe={k: recipe[k] for k in ('linear_C', 'catboost')})
    plan['identity'] = digest(plan)
    technical = resolve_path(c['output']) / 'technical'
    target = technical / 'plan.json'
    if target.exists() and read(target) != plan:
        raise ValueError('Dense data/producer changed; preserve this revision and use a new output')
    write_json(target, plan)
    write_json(technical/'inventory.json', dict(sources=len(sources),
        ancestors=len({s['ancestry_group'] for s in sources}), frames=sum(s['frame_count'] for s in sources),
        temperature_counts=dict(Counter(s['temperature_K'] for s in sources)),
        roles=dict(Counter(s['role'] for s in sources)), native_interval_ps=.1,
        history_span_ps=.5, appearance_lead_ps=.1, reused_old_birth_labels=False))
    return plan


def load(c):
    p = read(resolve_path(c['output']) / 'technical/plan.json')
    if (p['config'] != c or digest({k:v for k,v in p.items() if k != 'identity'}) != p['identity']
            or p['implementation'] != check_metric_docs(family=FAMILY)[FAMILY]['files']):
        raise ValueError('Changed dense input/implementation; use the frozen execution bundle')
    return p


def candidates(p, item):
    folder = external_data.source_folder(p, item['id'])
    receipt = read(folder/'complete.json')
    if receipt['identity'] != digest(dict(plan=p['identity'], source=item['id'])):
        raise ValueError(f'Changed native birth audit: {item["id"]}')
    for name in ('events.json', 'established-fates.json', 'transient-episodes.json'):
        if sha(folder/name) != receipt['files'][name]:
            raise ValueError(f'Changed native labels: {folder/name}')
    events = {e['event_id']:e for e in read(folder/'events.json')}
    rows = []
    for e in read(folder/'established-fates.json'):
        if e['kind'] == 'isolated_establishment':
            origin = events[e['event']]
            rows.append(dict(source=item['id'], event=f'established:{e["event"]}',
                stratum='established', first_frame=e['birth_frame'], first_node=origin['birth_node'],
                centroid_A=origin['centroid_A'], followup_frame=e['confirmation_frame'], fate=e['fate_label']))
    for e in read(folder/'transient-episodes.json'):
        if e['brief_failed_candidate']:
            rows.append(dict(source=item['id'], event=e['episode'],
                stratum='failed_strong' if e['primary_failed_candidate'] else 'failed_other',
                first_frame=e['first_frame'], first_node=e['first_node'],
                centroid_A=e['centroid_A'], followup_frame=e['dissolution_frame'], fate='dissolved'))
    return rows


def source(p, item):
    # Shared producer verifies input arrays, seals native PTM, follows crystal
    # components and distinguishes parent embryos, fronts and isolated births.
    result = shooting_fates.process(p, item)
    from src.research.liquid_predictability.descriptors import patch_descriptors
    xyz, box = external_data.geometry(item, 0)
    from scipy.spatial import cKDTree
    _, ids = cKDTree(xyz, boxsize=box).query(xyz[0], k=80)
    delta = xyz[ids]-xyz[0]; delta -= box*np.rint(delta/box)
    _, columns = patch_descriptors(delta.astype(np.float32))
    if len(columns) != 442:
        raise ValueError('Unexpected physical descriptor schema')
    labels = candidates(p, item)
    hp = dict(identity=digest(dict(plan=p['identity'], candidates=labels, columns=columns)),
              config=p['config'], audit=dict(identity=p['identity'], config=p['config']),
              candidates=labels, descriptor_columns=columns)
    folder = resolve_path(p['config']['cache'])/'sources'/item['id']
    write_json(folder/'input-plan.json', hp)
    prepared = histories.source(p['config'], hp, item, pool='dense010')
    return dict(source=item['id'], audit_counts=result['counts'], **prepared)


def seal(c, p):
    all_rows, features, coverage, counts, evidence = [], [], [], [], []
    cache = resolve_path(c['cache'])
    for item in p['sources']:
        folder = cache/'sources'/item['id']
        done, hp = read(folder/'complete.json'), read(folder/'input-plan.json')
        if (hp['audit']['identity'] != p['identity']
                or done['identity'] != digest(dict(data_plan=hp['identity'], source=item['id']))
                or any(sha(folder/n) != h for n,h in done['files'].items())):
            raise ValueError(f'Incomplete/changed dense histories: {folder}')
        rows = read(folder/'rows.json')
        x = np.load(folder/'features.npy')
        if x.shape != (len(rows), 6, 442) or not np.isfinite(x).all():
            raise ValueError(f'Unexpected dense descriptors: {folder}, {x.shape}')
        all_rows.extend(rows); features.append(x)
        coverage.extend(read(folder/'coverage.json'))
        audit = read(external_data.source_folder(p,item['id'])/'complete.json')
        counts.append(dict(source=item['id'], ancestor=item['ancestry_group'],
            native_interval_ps=.1, **audit['counts'], eligible_events=done['eligible_events'], rows=len(rows)))
        evidence.append(dict(source=item['id'], files=done['files'], identity=done['identity']))
    dest = resolve_path(c['output'])/'analyses/coverage-v1'
    write_metric_rows(counts, dest, family=FAMILY, name='sources')
    if coverage: write_metric_rows(coverage, dest, family=FAMILY, name='eligibility')
    if not all_rows:
        raise ValueError(f'No native 0.1 ps events passed crystal-free/control eligibility; see {dest}')
    rows = {k:np.asarray([r[k] for r in all_rows]) for k in all_rows[0]}
    event_counts = Counter(zip(rows['source'],rows['event']))
    rows['weight'] = np.asarray([1/event_counts[key] for key in zip(rows['source'],rows['event'])])
    if len(set(rows['id'])) != len(rows['id']):
        raise ValueError('Duplicate dense sample identities')
    from .temporal_analysis import matched
    matched(rows, np.arange(len(rows['id'])))
    if len(set(rows['ancestor'])) < 7:
        raise ValueError(f'Only {len(set(rows["ancestor"]))} eligible ancestors; insufficient for declared nested CV. See {dest}')
    np.savez_compressed(cache/'rows.npz', **rows)
    np.save(cache/'features.npy', np.concatenate(features))
    manifest = dict(plan=p['identity'], sources=evidence, files={name:sha(cache/name) for name in ('rows.npz','features.npy')},
        rows=len(rows['id']), events=len(event_counts), ancestors=len(set(rows['ancestor'])),
        input_context=dict(encoder=None, predictor=dict(bank='442 geometry/bond-order/CNA/TDA descriptors',
            geometry='nearest80 center-relative atoms; descriptor support r<8A', computational_halo=None,
            crystal_clearance_A=8., offsets_ps=c['offsets_ps'], velocities=False, relaxation=False,
            conditions=[], species=False, future_geometry=False), training_only_teachers=None))
    manifest['identity'] = digest(manifest)
    write_json(cache/'manifest.json', manifest)
    return rows, manifest


def dataset(c):
    cache = resolve_path(c['cache']); manifest = read(cache/'manifest.json')
    if (manifest['identity'] != digest({k:v for k,v in manifest.items() if k != 'identity'})
            or any(sha(cache/n) != h for n,h in manifest['files'].items())):
        raise ValueError('Dense dataset changed')
    with np.load(cache/'rows.npz') as saved: rows = {k:saved[k] for k in saved.files}
    return rows, np.load(cache/'features.npy'), manifest
