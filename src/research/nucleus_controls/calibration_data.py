"""Present-state-selected embryo cohort for a paired birth-rate comparison."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.data.fixed_cohort.protocol import digest,sha,write_json
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path
from src.postprocessing.protocols.extract import raw_source
from src.research.nucleus_recovery import Reference,read,roots
from .data import source_graph
from .negative_data import make_reference

FAMILY='nucleus_rate_calibration'


def prepare(c):
    check_metric_docs(family=FAMILY);out,_,archive=roots(c);path=out/'technical/plan.json'
    if path.exists():
        plan=read(path)
        if plan['config']!=c:raise ValueError('Existing calibration recipe differs')
    else:
        cache=resolve_path(c['birth_cache'])
        if read(cache/'manifest.json')['identity']!=c['birth_identity']:raise ValueError('Changed source release')
        sources=read(cache/'plan.json')['sources'];span=round(c['duration_ps']/.75);candidates=[];screen=[]
        for s in sources:
            g,_,maximum,first=source_graph(c,s['id']);f=np.arange(len(maximum)-span)
            eligible=f[(maximum[f]>=c['minimum_present_cluster_atoms']) & (maximum[f]<=c['maximum_present_cluster_atoms'])
                & (g['crystalline_atoms'][f]/s['atom_count']<=c['maximum_initial_crystal_fraction']) & (f<first)]
            screen.append(dict(source=s['id'],eligible_frames=eligible.tolist()))
            if len(eligible):candidates.append((digest(dict(seed=c['seed'],source=s['id'])),s,eligible))
        candidates.sort(key=lambda r:r[0])
        if len(candidates)<c['cases']:raise ValueError('Insufficient eligible ancestors')
        cases=[];tasks=[]
        previous=[]
        for old_path in c['previous_plans']:
            old=read(resolve_path(old_path))
            previous.extend((x['source'],x['start_frame'],old['identity'],x['key']) for x in old['cases'])
        for index,(_,item,eligible) in enumerate(candidates[:c['cases']]):
            sid=item['id'];rng=np.random.default_rng(c['seed']+sid);f=int(rng.choice(eligible));key=f'present-embryo-source{sid}'
            raw=raw_source(item)
            if raw.manifest['velocity_units']!='angstrom_per_ps':raise ValueError('Wrong saved velocity units')
            folder=resolve_path(c['original_audit'])/'technical/sources'/str(sid)
            g,_,_,_=source_graph(c,sid);nodes=np.flatnonzero(g['frame']==f)
            node=int(min(nodes,key=lambda n:(-int(g['size'][n]),int(n))));size=int(g['size'][node])
            case=dict(key=key,block=key,cohort='present_embryo',source=str(sid),ancestry=item['lineage'],role=item['role'],
                temperature_K=item['audit_metadata']['temperature_K'],cadence_ps=.75,chunk_frames=32,
                raw=str(raw.root),raw_manifest_sha256=item['manifest_sha256'],audit_folder=str(folder),graph_sha256=sha(folder/'graph.npz'),
                lineage=c['lineage'],atom_count=item['atom_count'],start_frame=f,stop_frame=f+span,
                threshold=dict(size=c['establishment_atoms'],persistence_frames=1+round(c['persistence_ps']/.75)),
                selection=dict(source=sid,frame=f,node=node,size=size,eligible_frames=eligible.tolist(),future_labels_used=False),
                position_dtype=str(raw.positions.dtype),velocity_dtype=str(raw.velocities.dtype))
            ref=Reference(case);q,box,dense=ref.frame(f);core=np.flatnonzero(dense==node)
            if len(core)!=size:raise ValueError('Changed present cluster membership')
            delta=q[core]-q[core[0]];delta-=box*np.rint(delta/box);center=np.mod(q[core[0]]+delta.mean(0),box)
            delta=q-center;delta-=box*np.rint(delta/box);anchor=int(np.argmin(np.sum(delta**2,axis=1)))
            case.update(core_rows=core.tolist(),anchor_row=anchor,anchor_atom_id=int(raw.atom_ids[anchor]),
                initial_cluster_atoms=size,initial_crystal_fraction=float(np.isin(ref.labels(f),[1,2,3]).mean()),
                previous_exact_states=[dict(plan_identity=p,case=k) for s,j,p,k in previous if s==str(sid) and j==f])
            state=dict(name=key,event=key,offset='fixed-12ps',start_frame=f,stop_frame=f+span,start_ps=f*.75,
                duration_ps=c['duration_ps'],cadence_ps=.75,panel=case['cohort'],initial_path=str(archive/'initial'/f'{key}.npz'))
            initial=Path(state['initial_path']);initial.parent.mkdir(parents=True,exist_ok=True)
            np.savez(initial,q=q.astype(np.float32),v_A_ps=np.array(raw.velocities[f],np.float32),box=box.astype(np.float32),atom_ids=raw.atom_ids)
            state['initial_sha256']=sha(initial);case['state']=state;cases.append(case)
            tasks.append(dict(name=key+'-source-npt',case=key,state=key,integrator='npt',variant='as_saved',temperature_K=case['temperature_K'],
                arms=['sourceNPT'],replicate=0,seed=None,priority=(0,index,0)))
            for r in range(c['noise_replicates']):
                seed=int(digest(dict(protocol=c['protocol'],seed=c['seed'],case=key,replicate=r))[:15],16)
                tasks.append(dict(name=key+f'-source-baoab-r{r}',case=key,state=key,integrator='baoab',variant='as_saved',temperature_K=case['temperature_K'],
                    arms=['sourceT'],replicate=r,seed=seed,priority=(r,index,1)))
            print('prepared',key,'frame',f,'present cluster',size,flush=True)
        if len({x['ancestry'] for x in cases})!=len(cases):raise ValueError('Duplicate calibration ancestor')
        tasks.sort(key=lambda t:t['priority'])
        plan=dict(config=c,cases=cases,tasks=tasks,screen=screen,source_plan_sha256=sha(cache/'plan.json'),
            pool_release_sha256=sha(resolve_path(c['pool_release'])),potential_sha256={k:sha(resolve_path(v)) for k,v in c['meam_files'].items()},
            scope=dict(screened_sources=len(sources),eligible_sources=len(candidates),independent_ancestors=len(cases),
                exact_starts_reused=sum(bool(x['previous_exact_states']) for x in cases),new_independent_ancestry_created=0,
                selection='present 8–63 atom largest cluster, <=1% crystal, no prior establishment; no future outcome selection',
                deferred=['other Al cadences','dense and shooting descendants','million-atom data','relaxed views']))
        plan['identity']=digest(plan);write_json(path,plan);write_json(archive/'plan.json',plan)
    # Seal the entire present-state cohort before measuring any saved future.
    missing=[(c,s) for s in plan['cases'] if not (out/'technical/reference'/f'{s["key"]}.json').exists()]
    if missing:
        with ProcessPoolExecutor(max_workers=c['reference_workers']) as pool:
            for key in pool.map(make_reference,missing):print('reference measured',key,flush=True)
    return plan
