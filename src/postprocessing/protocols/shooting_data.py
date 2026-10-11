"""Bind available Al shooting futures once, preserving nested continuations."""
from collections import Counter
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.data.structural_pretraining.prepare import source_arrays
from src.project_runtime.paths import dataset_path, resolve_path
from src.postprocessing.protocols import ancestry, external_data, external_audit, extract, fates


def read(path):
    return json.loads(Path(path).read_text())


def bound_arrays(item):
    """Read the repository shooting schema at an already resolved campaign path."""
    path = Path(item['path'])
    m = read(path / 'manifest.json')
    if sha(path / 'manifest.json') != item['manifest_sha256']:
        raise ValueError(f'Changed shooting manifest: {path}')
    if m['state'] != 'complete' or m['format'] != 'pointcloudmaterials.shooting_trajectory':
        raise ValueError(f'Unexpected shooting producer/state: {path}')
    arrays = {}
    for name in ('positions', 'atom_ids', 'atom_types', 'timesteps', 'box_low', 'box_high'):
        spec = m['arrays'][name]
        a = np.load(path / spec['file'], mmap_mode='r', allow_pickle=False)
        if list(a.shape) != spec['shape'] or a.dtype.name != spec['dtype']:
            raise ValueError(f'Changed shooting array schema: {path}/{name}')
        arrays[name] = a
    if np.any(arrays['atom_types'] != 1):
        raise ValueError(f'Unexpected mixed species in elemental Al shooting: {path}')
    return arrays


def implementation():
    paths = [Path(__file__), Path(__file__).with_name('shooting_fates.py'),
             Path(__file__).with_name('shooting.py'), Path(ancestry.__file__),
             Path(external_data.__file__), Path(external_audit.__file__),
             Path(extract.__file__), Path(fates.__file__), Path(source_arrays.__code__.co_filename)]
    return dict(files={p.name: sha(p) for p in paths},
                libraries={n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'ovito')})


def part(root, branch, outcome):
    # Recorded absolute IDS paths predate the WORK migration. The registry owns
    # the campaign root; the trajectory basename is the original producer's.
    path = root / branch['branch_dir'] / Path(outcome['trajectory_artifact']['path']).name
    m = read(path / 'manifest.json')
    item = dict(path=str(path), manifest_sha256=sha(path / 'manifest.json'), timestep_fs=3.)
    a = bound_arrays(item)
    if a['positions'].shape != (outcome['frame_count'], 70304, 3):
        raise ValueError(f'Outcome/array shape mismatch: {path}')
    if a['timesteps'][0] != 0 or np.any(np.diff(a['timesteps']) <= 0):
        raise ValueError(f'Invalid shooting timeline: {path}')
    if np.any(np.diff(a['atom_ids']) <= 0):
        raise ValueError(f'Invalid atom identities: {path}')
    item['storage_dtype'] = m['arrays']['positions']['dtype']
    item['array_sha256'] = {k: m['arrays'][k]['sha256'] for k in a}
    return item, a


def prepare(c):
    sources, unavailable, campaigns, nested = [], [], [], {}
    for exclusion in c['excluded_campaigns']:
        receipt = resolve_path(exclusion['retirement_record'])
        if sha(receipt) != exclusion['retirement_sha256']:
            raise ValueError(f'Changed campaign retirement record: {receipt}')
        unavailable.append(dict(campaign=exclusion['campaign'], branch_id=None,
            reason='payload_deleted_in_separate_cleanup'))
    seen = {}
    for name in c['campaigns']:
        root = dataset_path(name)
        manifest = read(root / 'manifest.json')
        campaigns.append(dict(dataset=name, path=str(root), manifest_sha256=sha(root / 'manifest.json')))
        parents = {p['parent_id']: p for p in manifest['parents']}
        for b in manifest['branches']:
            path = root / b['branch_dir'] / 'outcome.json'
            common = dict(campaign=name, branch_id=b['branch_id'])
            if not path.exists():
                unavailable.append(dict(common, reason='no_completed_outcome')); continue
            o = read(path)
            if o['state'] != 'complete':
                unavailable.append(dict(common, reason='outcome_' + o['state'])); continue
            for key in ('parent_id', 'branch_id', 'source_run_id', 'source_split'):
                if o[key] != b[key]:
                    raise ValueError(f'Branch/outcome mismatch: {path}/{key}')
            p = parents[b['parent_id']]
            if p['source_run_id'] != b['source_run_id'] or p['source_split'] != b['source_split']:
                raise ValueError(f'Parent/branch ancestry mismatch: {path}')
            item, a = part(root, b, o)
            provenance = dict(common, outcome_sha256=sha(path), parent=p, branch=b)
            if name == c['nested_original']:
                nested[b['branch_id']] = (item, o, provenance)
                continue
            frames = [[0, i] for i in range(len(a['timesteps']))]
            parts, evidence = [item], None
            protocol = manifest['protocol']
            if protocol['timestep_fs'] != 3. or protocol['sample_interval_ps'] != c['cadence_ps']:
                raise ValueError(f'Unexpected shooting protocol: {name}')
            times = a['timesteps'].astype(float) * .003
            np.testing.assert_allclose(times, np.arange(len(times)) * c['cadence_ps'], rtol=0, atol=1e-8)
            if name == c['nested_fixed']:
                old_part, old_outcome, old_provenance = nested.pop(b['branch_id'])
                if b['source_outcome_sha256'] != old_provenance['outcome_sha256']:
                    raise ValueError(f'Nested continuation points to a different outcome: {path}')
                if p['data_sha256'] != old_provenance['parent']['data_sha256']:
                    raise ValueError(f'Nested parent identity changed: {path}')
                original = bound_arrays(old_part)
                np.testing.assert_array_equal(original['atom_ids'], a['atom_ids'])
                # Use the original, higher-precision segment wherever available.
                # Fixed24 is the same future's prefix + continuation, not a shot.
                old_steps = {int(v): i for i, v in enumerate(original['timesteps'])}
                new_steps = {int(v): i for i, v in enumerate(a['timesteps'])}
                shared = sorted(set(old_steps) & set(new_steps))
                maximum = 0.
                for step in shared:
                    oi, ni = old_steps[step], new_steps[step]
                    np.testing.assert_array_equal(original['box_low'][oi], a['box_low'][ni])
                    np.testing.assert_array_equal(original['box_high'][oi], a['box_high'][ni])
                    # The compositor stores wrapped float16 coordinates. Account
                    # for a value rounding to the box edge before wrapping.
                    box = a['box_high'][ni] - a['box_low'][ni]
                    rounded = original['positions'][oi].astype(np.float16)
                    delta = rounded.astype(float) - a['positions'][ni].astype(float)
                    delta -= box * np.rint(delta / box)
                    maximum = max(maximum, float(np.abs(delta).max()))
                if maximum > 1e-6:
                    raise ValueError(f'Nested shared-prefix mismatch: {path}, max={maximum}')
                last = max(int(original['timesteps'][-1]), int(a['timesteps'][-1]))
                steps = list(range(0, last + 1, 100))
                if last % 100 or any(s not in old_steps and s not in new_steps for s in steps):
                    raise ValueError(f'Cannot select exact 0.3 ps nested history: {path}')
                parts = [old_part, item]
                frames = [[0, old_steps[s]] if s in old_steps else [1, new_steps[s]] for s in steps]
                times = np.asarray(steps) * .003
                evidence = dict(original=old_provenance, shared_frames=len(shared),
                    rounded_prefix_max_error_A=maximum, original_end_ps=float(original['timesteps'][-1] * .003),
                    continuation_used=any(pi == 1 for pi, _ in frames),
                    continuation_thermostat_seed=o.get('continuation_thermostat_seed'),
                    rule='original prefix; unique continuation suffix; exact 0.3 ps observations')
            # Identical stored paths or array identities cannot create a second future.
            key = digest(dict(parts=[q['array_sha256'] for q in parts], frames=frames))
            if key in seen:
                unavailable.append(dict(common, reason='duplicate_future', retained_source=seen[key])); continue
            sid = 'shot-' + digest(dict(campaign=name, branch=b['branch_id']))[:16]
            seen[key] = sid
            lineage = p['source_run_id']
            if lineage.startswith('source_group_'):
                lineage = lineage.split('/', 1)[1]
            settings = dict(c['lineage'], thresholds=[dict(name='primary', size=c['establishment_atoms'],
                persistence_ps=c['establishment_persistence_ps'],
                persistence_frames=1 + round(c['establishment_persistence_ps'] / c['cadence_ps']))])
            sources.append(dict(id=sid, **common, material='Al', potential='al-lee2003-meam',
                parent_id=p['parent_id'], parent_sha256=p['data_sha256'], ancestry_group=lineage,
                role=p['source_split'], parent_frame_time_ps=p['source_frame_time_ps'],
                temperature_K=p['temperature_K'], parts=parts, frames=frames, times_ps=times.tolist(),
                atom_count=70304, frame_count=len(frames), cadence_ps=c['cadence_ps'],
                center_atom_ids=[int(a['atom_ids'][0])], baseline_center_indices=[0],
                lineage=settings, phase_boundary=evidence, provenance=provenance,
                dynamics='nested_continuation' if evidence else protocol['ensemble'],
                protocol=protocol, branch_observed_duration_ps=float(times[-1])))
    if nested:
        raise ValueError(f'Original nested futures without their registered fixed24 counterpart: {list(nested)}')
    tasks, loads = [], [0] * c['lanes']
    for s in sorted(sources, key=lambda s: (-s['frame_count'], s['id'])):
        lane = min(range(c['lanes']), key=loads.__getitem__)
        s['lane'] = lane
        loads[lane] += s['frame_count'] * s['atom_count']
        for start in range(0, s['frame_count'], c['chunk_frames']):
            tasks.append(dict(source=s['id'], start=start, stop=min(start+c['chunk_frames'], s['frame_count'])))
    value = dict(config=c, sources=sorted(sources, key=lambda s: s['id']), tasks=tasks,
                 unavailable=unavailable, campaigns=campaigns, implementation=implementation(), lane_atom_frames=loads)
    value['identity'] = digest(value)
    path = resolve_path(c['output']) / 'technical/plan.json'
    if path.exists() and read(path) != value:
        raise ValueError('Shooting audit population or producer changed; use a new output revision')
    write_json(path, value)
    counts = dict(branches=len(sources), parents=len({s['parent_sha256'] for s in sources}),
        ancestral_sources=len({s['ancestry_group'] for s in sources}), frames=sum(s['frame_count'] for s in sources),
        campaigns=dict(Counter(s['campaign'] for s in sources)), excluded=dict(Counter(s['reason'] for s in unavailable)))
    write_json(path.with_name('inventory.json'), counts)
    return value
