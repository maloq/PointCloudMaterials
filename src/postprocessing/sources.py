"""Inventory all Al observations and bind exact, separate cadence views."""
from collections import Counter
from datetime import datetime, timezone
from src.postprocessing.io import read

import importlib.metadata
import json
from pathlib import Path

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.project_runtime.paths import resolve_path, path_snapshot
from src.postprocessing.io import arrays


def part(path, timestep_fs):
    path = Path(path).resolve()
    m = read(path / 'manifest.json')
    return dict(path=str(path), manifest_sha256=sha(path/'manifest.json'),
        timestep_fs=float(timestep_fs), array_sha256={k:m['arrays'][k]['sha256'] for k in
        ('positions', 'atom_ids', 'atom_types', 'timesteps', 'box_low', 'box_high')})


def native(parts):
    """Stitch only explicitly supplied continuous phases, checking their boundary."""
    frames, times, offset = [], [], 0.
    previous = None
    for pi, p in enumerate(parts):
        a = arrays(p['path'], p['manifest_sha256'])
        t = (a['timesteps']-a['timesteps'][0]).astype(float)*p['timestep_fs']/1000
        if len(t) < 2 or np.any(np.diff(t) <= 0):
            raise ValueError(f'Invalid temporal source: {p["path"]}')
        first = 0
        if previous is not None:
            np.testing.assert_array_equal(previous['atom_ids'], a['atom_ids'])
            b = previous['box_high'][-1].astype(float)-previous['box_low'][-1].astype(float)
            np.testing.assert_allclose(b, a['box_high'][0]-a['box_low'][0], atol=1e-4, rtol=0)
            delta = previous['positions'][-1].astype(float)-a['positions'][0].astype(float)
            delta -= b*np.rint(delta/b)
            if np.abs(delta).max() > .001:
                raise ValueError(f'Unverified continuous phase boundary: {p["path"]}')
            first = 1
        frames.extend([[pi, f] for f in range(first,len(t))])
        times.extend((offset+t[first:]).tolist())
        offset += float(t[-1]); previous = a
    return frames, times


def metadata_at(path, dataset):
    """Read nearby producer records, with explicit precedence over inventory facts."""
    result, evidence = {}, []
    root = Path(dataset['location']['resolved']).resolve()
    parents = list(Path(path).parents)
    for folder in reversed(parents[:parents.index(root)+1] if root in parents else parents[:2]):
        for filename in ('config.json','metadata.json','outcome.json','input_metadata.json'):
            f = folder/filename
            if f.is_file():
                m = read(f)
                if isinstance(m,dict):
                    result.update(m); evidence.append(dict(path=str(f),sha256=sha(f)))
    return result,evidence


@path_snapshot()
def prepare(c):
    if (resolve_path(c['output'])/'technical/launch.json').exists():
        raise ValueError('Search already launched; use a new release for population changes')
    registry_path = resolve_path(c['registry']); registry = read(registry_path)
    sources, coverage, consumed, seen_content = [], list(registry.get('selection_audit', [])), {}, {}
    excluded = read(resolve_path(c['exclude_plan'])) if c.get('exclude_plan') else None
    if excluded is not None:
        for s in excluded['sources']:
            for p in s['parts']: consumed[str(Path(p['path']).resolve())] = s['id']
            seen_content[s['input_identity']] = s['id']

    def add(s):
        verified=[]
        for p in s['parts']:
            current=part(p['path'],p['timestep_fs'])
            if 'manifest_sha256' in p and p['manifest_sha256']!=current['manifest_sha256']:
                raise ValueError(f'Previously composed input changed: {p["path"]}')
            verified.append(current)
        s['parts']=verified
        t = np.asarray(s['times_ps'],float)
        if len(t) < 2: raise ValueError(f'Source too short: {s["id"]}')
        dt = np.diff(t)
        if not np.allclose(dt,dt[0],rtol=0,atol=1e-8):
            values,counts=np.unique(np.round(dt,8),return_counts=True)
            base=float(values[np.argmax(counts)])
            cuts=[0,*((np.flatnonzero(np.abs(dt-base)>1e-8)+1).tolist()),len(t)]
            coverage.append(dict(dataset=s['dataset'],path=s['parts'][0]['path'],source=s['id'],
                state='segmented_irregular_timeline',intervals_ps=values.tolist(),cuts=cuts))
            for index,(begin,end) in enumerate(zip(cuts[:-1],cuts[1:])):
                if end-begin<2:continue
                add(dict(s,id=s['id']+f'-segment{index}',frames=s['frames'][begin:end],
                    times_ps=s['times_ps'][begin:end],segmented_parent=s['id']))
            return
        s['native_cadence_ps'] = float(dt[0]); s['views']={}
        if 'native_cadence_ps' in c and abs(dt[0]-c['native_cadence_ps'])>1e-8:
            coverage.append(dict(dataset=s['dataset'],path=s['parts'][0]['path'],
                state='outside_requested_native_cadence',native_cadence_ps=float(dt[0])))
            return
        for name,step in c['cadences'].items():
            stride=int(round(step/dt[0]))
            if stride>=1 and abs(stride*dt[0]-step)<1e-8:
                selection=list(range(0,len(t),stride))
                if len(selection)>=2:s['views'][name]=selection
        if not s['views']:
            coverage.append(dict(dataset=s['dataset'], path=s['parts'][0]['path'],
                state='incompatible_cadence',native_cadence_ps=float(dt[0]), source=s['id']))
            return
        identity = digest(dict(arrays=[p['array_sha256'] for p in s['parts']],frames=s['frames']))
        if identity in seen_content:
            coverage.append(dict(dataset=s['dataset'],path=s['parts'][0]['path'],
                state='duplicate_arrays',retained_source=seen_content[identity]));return
        seen_content[identity]=s['id']
        s['input_identity']=identity
        s['sampled_frames']=sorted(set(f for v in s['views'].values() for f in v))
        s['frame_count']=len(t)
        a=arrays(s['parts'][0]['path'],s['parts'][0]['manifest_sha256'])
        if np.any(np.diff(a['atom_ids'])<=0):raise ValueError(f'Unordered atom IDs: {s["id"]}')
        s['atom_count']=len(a['atom_ids'])
        for p in s['parts']: consumed[p['path']]=s['id']
        sources.append(s)
        coverage.append(dict(dataset=s['dataset'],path=s['parts'][0]['path'],source=s['id'],
            state='included',native_cadence_ps=s['native_cadence_ps'],protocols=list(s['views']),
            ancestor=s['ancestry_group'],domain=s['origin'],dataset_role=s.get('dataset_role'),
            producer_protocol=s.get('producer_protocol')))

    if excluded is None:
        from src.postprocessing.protocols.extract import raw_source
        old=read(resolve_path(c['earlier_plan']));old_sha=sha(resolve_path(c['earlier_plan']))
        for s in old['sources']:
            raw=raw_source(s);p=part(raw.root,s['timestep_fs']);frames,times=native([p])
            add(dict(id=f'old-{s["id"]}',dataset=s['dataset'],parts=[p],frames=frames,times_ps=times,
                ancestry_group=s['lineage'],role=s['role'],potential='al-lee2003-meam',
                temperature_K=s['audit_metadata']['temperature_K'],origin='historical_fixed_cohort',
                provenance=dict(source=s,prepared_sha256=old_sha)))
        # This producer already verifies and composes nested continuations without
        # counting their shared prefix twice. Its exact native .30 ps arrays persist.
        shooting=read(resolve_path(c['shooting_plan']))
        for s in shooting['sources']:
            add(dict(id=s['id'],dataset=s['campaign'],parts=s['parts'],frames=s['frames'],
                times_ps=s['times_ps'],ancestry_group=s['ancestry_group'],role=s['role'],
                potential=s['potential'],temperature_K=s['temperature_K'],origin='shooting',
                parent_sha256=s['parent_sha256'],provenance=s['provenance']))
        external=read(resolve_path(c['external_plan']))
        for s in external['sources']:
            if s['material']!='Al':continue
            ps=[part(resolve_path(p['path']),p['timestep_fs']) for p in s['parts']]
            fs,ts=native(ps)
            add(dict(id='external-'+s['id'],dataset='al-million-continuous' if len(ps)>1 else 'al-eam-six-24ps',
                parts=ps,frames=fs,times_ps=ts,ancestry_group=s['ancestry_group'],role='external',
                potential=s['potential'],temperature_K=None,origin='external_continuous',
                provenance=dict(previous_plan_sha256=sha(resolve_path(c['external_plan'])),phase_boundary=s['phase_boundary'])))

    # Enumerate registered observations, not only the historical training cohort.
    for d in registry['datasets']:
        if 'Al' not in d['materials']:
            if not d['materials'] and d['role']=='raw_dynamics':
                coverage.append(dict(dataset=d['id'],state='unknown_material_review_required'))
            continue
        if d['materials']!=['Al']:
            coverage.append(dict(dataset=d['id'],state='mixed_material_outside_elemental_Al_protocol'));continue
        if 'allowed_roles' in c and d['role'] not in c['allowed_roles']:
            coverage.append(dict(dataset=d['id'],state='outside_requested_production_role',role=d['role']));continue
        if 'relax' in d['id'].lower() or 'relax' in d['role'].lower():
            coverage.append(dict(dataset=d['id'],state='paired_relaxed_view_not_independent_dynamics'));continue
        if d['role'] not in ('raw_dynamics','simulation','storage_conversion') and d['kind']!='simulation':
            coverage.append(dict(dataset=d['id'],state='derived_or_static_view',role=d['role']));continue
        if not d['location']['available']:
            coverage.append(dict(dataset=d['id'],state='unavailable'));continue
        records=read(resolve_path('docs/datasets')/d['records_file'])['records']
        temporal=[x for x in records if x['kind']=='trajectory']
        if not temporal:
            coverage.append(dict(dataset=d['id'],state='no_temporal_binary',role=d['role']));continue
        for x in temporal:
            path=Path(x['path']).parent.resolve()
            if str(path) in consumed:
                coverage.append(dict(dataset=d['id'],path=str(path),state='already_included',retained_source=consumed[str(path)]));continue
            tr=x['trajectory']
            if tr['frame_count']<2:
                coverage.append(dict(dataset=d['id'],path=str(path),state='single_configuration'));continue
            if not tr['usable_binary']:
                coverage.append(dict(dataset=d['id'],path=str(path),state='binary_incomplete'));continue
            # float32 and float16 siblings are views of the same observations.
            if path.name=='trajectory_binary_float32' and (path.with_name('trajectory_binary_float16')/'manifest.json').is_file():
                coverage.append(dict(dataset=d['id'],path=str(path),state='paired_precision_view',retained_path=str(path.with_name('trajectory_binary_float16'))));continue
            prepared_liquid_shot = (d['metadata'].get('sampling_protocol') ==
                                   'prepared_liquid_velocity_reshoot_v1')
            if 'shooting' in d['id'] and not prepared_liquid_shot:
                coverage.append(dict(dataset=d['id'],path=str(path),state='shooting_prefix_or_diagnostic',reason='see frozen shooting compositor/exclusions'));continue
            if any(v in str(path).lower() for v in ('/preflight/','/smoke/','stopping-validation','/melt_validation.',
                                                   'dense001-conversion-check','birth-source-validation')):
                coverage.append(dict(dataset=d['id'],path=str(path),state='preparation_or_debug_observations'));continue
            m=read(path/'manifest.json');meta,evidence=metadata_at(path,d)
            # Offline conversions retain the original text path; that producer's
            # adjacent metadata remains the authority for integration time.
            original=m.get('source',{}).get('path')
            if original:
                origin_meta,origin_evidence=metadata_at(Path(original),d)
                meta.update(origin_meta);evidence.extend(origin_evidence)
            if meta.get('root_lineage')=='infrastructure_fixture':
                coverage.append(dict(dataset=d['id'],path=str(path),state='preparation_or_debug_observations'));continue
            provenance=m.get('provenance',{})
            known=provenance.get('metadata',provenance)
            if 'float32_manifest' in provenance:
                known=provenance['float32_manifest'].get('provenance',{})
            step=known.get('timestep_ps',meta.get('timestep_ps'))
            if step is None and 'timestep_fs' in known:step=known['timestep_fs']/1000
            if step is None and 'timestep_fs' in meta:step=meta['timestep_fs']/1000
            if step is None:
                # Uniform inventory facts are usable only when a single producer
                # timestep is recorded, never inferred from a directory name.
                ps=d['facts'].get('timestep_ps',[]);fs=d['facts'].get('timestep_fs',[])
                if len(ps)==1:step=ps[0]
                elif len(fs)==1:step=fs[0]/1000
            if step is None:
                coverage.append(dict(dataset=d['id'],path=str(path),state='unknown_integration_timestep'));continue
            if d['role'] not in ('raw_dynamics','simulation','storage_conversion') and d['kind']!='simulation':
                coverage.append(dict(dataset=d['id'],path=str(path),state='derived_or_static_view',role=d['role']));continue
            p=part(path,step*1000);fs,ts=native([p])
            ancestor=known.get('root_lineage',meta.get('root_lineage',d['metadata'].get('ancestry')))
            if known.get('initial_sha256'):
                ancestor='initial_configuration_sha256:'+known['initial_sha256']
            if ancestor is None and meta.get('preparation_seed') is not None:ancestor=f'independent_melt_{meta["preparation_seed"]}'
            if ancestor is None:ancestor='unknown_within_dataset:'+d['id']
            # A stopped prefix is retained as a historical view, not a fresh melt.
            domain='historical_partial' if any(k in d['id'] for k in ('failed','stopped','archive')) else 'original'
            if prepared_liquid_shot and domain == 'original':
                domain = 'shooting'
            outcome_path=path.parent/'outcome.json'
            completion_path=path.parent/'complete.json'
            outcome=read(outcome_path) if outcome_path.is_file() else read(completion_path) if completion_path.is_file() else {}
            if outcome:
                evidence.append(dict(path=str(outcome_path if outcome_path.is_file() else completion_path),sha256=sha(outcome_path if outcome_path.is_file() else completion_path)))
            elif d['id'].startswith('meam-gpu-blackwell-'):
                domain='historical_partial'
            if c.get('required_simulation_state') and outcome.get('state')!=c['required_simulation_state']:
                coverage.append(dict(dataset=d['id'],path=str(path),state='production_run_not_complete',
                    producer_state=outcome.get('state','unrecorded')));continue
            add(dict(id='al-'+digest(str(path))[:16],dataset=d['id'],parts=[p],frames=fs,times_ps=ts,
                dataset_role=d['role'],producer_protocol=meta.get('protocol',m.get('source',{}).get('protocol')),
                ancestry_group=ancestor,role=meta.get('split','unassigned'),
                potential=d['potential_ids'],temperature_K=meta.get('temperature_K'),
                origin=domain,provenance=dict(metadata=evidence,manifest_provenance=provenance,
                scientific_state=outcome.get('state',meta.get('state','unrecorded')),stop_reason=outcome.get('termination',meta.get('stop_reason')))))
    tasks=[];loads=[0]*c['extraction_lanes']
    for s in sorted(sources,key=lambda s:-s['atom_count']*len(s['sampled_frames'])):
        for start in range(0,len(s['sampled_frames']),c['chunk_frames']):
            end=min(start+c['chunk_frames'],len(s['sampled_frames']))
            lane=int(np.argmin(loads));cost=(end-start)*s['atom_count'];loads[lane]+=cost
            tasks.append(dict(source=s['id'],start=start,stop=end,lane=lane,atom_frames=cost))
    jobs=[];loads2=[0]*c['analysis_lanes']
    for s in sorted(sources,key=lambda s:-s['atom_count']*len(s['sampled_frames'])):
        for cadence,frames in s['views'].items():
            for detector in c['detectors']:
                lane=int(np.argmin(loads2));loads2[lane]+=len(frames)*s['atom_count']
                jobs.append(dict(source=s['id'],cadence=cadence,detector=detector,lane=lane))
    libraries={k:importlib.metadata.version(k) for k in ('numpy','scipy','ovito','freud-analysis')}
    p=dict(config=c,sources=sources,tasks=tasks,jobs=jobs,coverage=coverage,
        registry_sha256=sha(registry_path),registry_generated_at=registry['generated_at'],
        remote_holdings=registry['remote_holdings'],libraries=libraries)
    if excluded is not None:
        p['excluded_plan_sha256']=sha(resolve_path(c['exclude_plan']))
        p['excluded_identity']=excluded['identity']
    p['identity']=digest(p)
    root=resolve_path(c['output']);write_json(root/'technical/plan.json',p)
    inventory=dict(sources=len(sources),cadences={k:sum(k in s['views'] for s in sources) for k in c['cadences']},
        unique_observed_frames=sum(len(s['sampled_frames']) for s in sources),chunks=len(tasks),
        atom_frames=sum(loads),coverage_states=dict(Counter(x['state'] for x in coverage)),
        by_dataset=dict(Counter(s['dataset'] for s in sources)),generated_at=datetime.now(timezone.utc).isoformat())
    write_json(root/'technical/inventory.json',inventory)
    arrays.cache_clear()
    return p,inventory
