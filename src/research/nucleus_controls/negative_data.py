"""Additional liquid ancestors and repeated failed-embryo continuations."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path
from src.postprocessing.protocols.extract import raw_source
from src.research.nucleus_recovery import Reference, read, roots
from .data import source_graph
from .analysis import reference

FAMILY='nucleus_negative_extension'


def make_reference(arguments):
    c,case=arguments
    result=reference(c,case)
    write_json(roots(c)[0]/'technical/reference'/f'{case["key"]}.json',result)
    return case['key']


def prepare(c):
    check_metric_docs(family=FAMILY)
    out,_,archive=roots(c);path=out/'technical/plan.json'
    previous_path=resolve_path(c['previous_plan']);previous=read(previous_path)
    if previous['identity']!=c['previous_identity']:raise ValueError('Changed previous controls plan')
    # Reused starting states retain precisely the previous physics and endpoints.
    for key in ['duration_ps','evaluation_horizons_ps','response_temperature_K','timestep_fs','friction_fs',
                'mass_amu','source_npt','site_radius_A','ptm','lineage','establishment_atoms','persistence_ps',
                'dissolution_confirmation_frames','meam_files','birth_identity']:
        if c[key]!=previous['config'][key]:raise ValueError('Changed inherited protocol: '+key)
    if path.exists():
        plan=read(path)
        if plan['config']!=c:raise ValueError('Existing negative extension differs')
    else:
        cache=resolve_path(c['birth_cache'])
        if read(cache/'manifest.json')['identity']!=c['birth_identity']:raise ValueError('Changed source release')
        sources=read(cache/'plan.json')['sources']
        excluded={s['ancestry'] for s in previous['cases']}
        candidates=[];screen=[];span=round(c['duration_ps']/.75)
        for item in sources:
            sid=item['id']
            if item['lineage'] in excluded:
                screen.append(dict(source=sid,reason='ancestor in previous matched campaign'));continue
            g,_,maximum,_=source_graph(c,sid)
            eligible=np.flatnonzero((maximum[:-span]<=c['random_maximum_cluster_atoms']) &
                (g['crystalline_atoms'][:-span]/item['atom_count']<=c['maximum_initial_crystal_fraction']))
            screen.append(dict(source=sid,ancestry=item['lineage'],eligible_frames=len(eligible)))
            if len(eligible):
                candidates.append((digest(dict(seed=c['seed'],source=sid)),item,eligible))
        candidates.sort(key=lambda x:x[0])
        if len(candidates)<c['random_cases']:raise ValueError('Insufficient eligible independent liquid ancestors')
        selected=candidates[:c['random_cases']];cases=[]
        for _,item,eligible in selected:
            sid=item['id'];rng=np.random.default_rng(c['seed']+sid)
            f=int(rng.choice(eligible));anchor=int(rng.integers(item['atom_count']))
            raw=raw_source(item)
            if raw.manifest['velocity_units']!='angstrom_per_ps':raise ValueError('Wrong saved velocity units')
            folder=resolve_path(c['original_audit'])/'technical/sources'/str(sid)
            key=f'new-liquid-source{sid}'
            case=dict(key=key,block=key,cohort='random_liquid',source=str(sid),ancestry=item['lineage'],
                role=item['role'],temperature_K=item['audit_metadata']['temperature_K'],cadence_ps=.75,chunk_frames=32,
                raw=str(raw.root),raw_manifest_sha256=item['manifest_sha256'],audit_folder=str(folder),
                graph_sha256=sha(folder/'graph.npz'),lineage=c['lineage'],atom_count=item['atom_count'],
                start_frame=f,stop_frame=f+span,
                threshold=dict(size=c['establishment_atoms'],persistence_frames=1+round(c['persistence_ps']/.75)),
                selection=dict(source=sid,frame=f,eligible_frames=eligible.tolist(),anchor_row=anchor,
                    rule='uniform eligible source by seeded hash; uniform present-liquid frame; no future outcome filter'),
                position_dtype=str(raw.positions.dtype),velocity_dtype=str(raw.velocities.dtype),
                core_rows=[],anchor_row=anchor,anchor_atom_id=int(raw.atom_ids[anchor]),initial_cluster_atoms=0)
            ref=Reference(case);q,box,_=ref.frame(f)
            case['initial_crystal_fraction']=float(np.isin(ref.labels(f),[1,2,3]).mean())
            state=dict(name=key,event=key,offset='fixed-12ps',start_frame=f,stop_frame=f+span,start_ps=f*.75,
                duration_ps=c['duration_ps'],cadence_ps=.75,panel='random_liquid',initial_path=str(archive/'initial'/f'{key}.npz'))
            initial=Path(state['initial_path']);initial.parent.mkdir(parents=True,exist_ok=True)
            np.savez(initial,q=q.astype(np.float32),v_A_ps=np.array(raw.velocities[f],np.float32),
                box=box.astype(np.float32),atom_ids=raw.atom_ids)
            state['initial_sha256']=sha(initial);case['state']=state;cases.append(case)
            print('prepared',key,'frame',f,flush=True)
        for original in previous['cases']:
            if original['cohort']!='dissolved':continue
            case=deepcopy(original);state=case['state'];source=Path(state['initial_path'])
            if sha(source)!=state['initial_sha256']:raise ValueError('Changed reused embryo starting state')
            target=archive/'initial'/source.name;shutil.copy2(source,target)
            if sha(target)!=state['initial_sha256']:raise ValueError('Starting state copy differs')
            state['initial_path']=str(target)
            case['reused_from']=dict(plan_identity=previous['identity'],case=original['key'],initial_path=str(source))
            cases.append(case)
        if len({s['ancestry'] for s in cases})!=len(cases):raise ValueError('New cohort contains repeated ancestors')
        tasks=[]
        old_seeds={t['seed'] for t in previous['tasks'] if t['seed'] is not None}
        for index,case in enumerate(cases):
            repeats=c['random_noise_replicates'] if case['cohort']=='random_liquid' else c['dissolved_extra_replicates']
            first=0 if case['cohort']=='random_liquid' else previous['config']['noise_replicates']
            for r in range(repeats):
                replicate=first+r
                seed=int(digest(dict(protocol=c['protocol'],seed=c['seed'],case=case['key'],replicate=replicate))[:15],16)
                if seed in old_seeds:raise ValueError('Reused stochastic seed')
                old_seeds.add(seed)
                tasks.append(dict(name=case['key']+f'-extra-response450-r{replicate}',case=case['key'],state=case['key'],
                    integrator='baoab',variant='as_saved',temperature_K=c['response_temperature_K'],arms=['response450K'],
                    replicate=replicate,seed=seed,priority=(r,index,1)))
            if case['cohort']=='random_liquid':
                tasks.append(dict(name=case['key']+'-source-npt',case=case['key'],state=case['key'],integrator='npt',
                    variant='as_saved',temperature_K=case['temperature_K'],arms=['sourceNPT'],replicate=0,seed=None,priority=(0,index,0)))
        tasks.sort(key=lambda t:t['priority'])
        plan=dict(config=c,cases=cases,tasks=tasks,screen=screen,
            previous_plan_sha256=sha(previous_path),source_plan_sha256=sha(cache/'plan.json'),
            pool_release_sha256=sha(resolve_path(c['pool_release'])),
            potential_sha256={k:sha(resolve_path(v)) for k,v in c['meam_files'].items()},
            scope=dict(original_sources_screened=len(sources),eligible_new_ancestors=len(candidates),
                new_liquid_ancestors=len(selected),reused_dissolved_ancestors=len(cases)-len(selected),
                independent_ancestors=len(cases),new_independent_ancestry_created=0,
                new_liquid_temperature_counts=dict(Counter(str(s['temperature_K']) for s in cases if s['cohort']=='random_liquid')),
                deferred=previous['scope']['deferred']))
        plan['identity']=digest(plan);write_json(path,plan);write_json(archive/'plan.json',plan)
    missing=[]
    for case in plan['cases']:
        target=out/'technical/reference'/f'{case["key"]}.json'
        if target.exists():continue
        if case['cohort']=='dissolved':
            source=previous_path.parent/'reference'/target.name
            value=read(source);value['reused_reference_sha256']=sha(source)
            write_json(target,value)
        else:missing.append((c,case))
    if missing:
        with ProcessPoolExecutor(max_workers=c['reference_workers']) as pool:
            for key in pool.map(make_reference,missing):print('reference measured',key,flush=True)
    return plan
