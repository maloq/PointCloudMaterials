"""Freeze size/temperature matched historical cohorts before new continuations."""
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path
from src.postprocessing.protocols.extract import raw_source
from src.postprocessing.io import frame_geometry
from src.research.nucleus_recovery import Reference, read, roots

FAMILY = 'nucleus_controls'


def source_graph(c, source):
    folder = resolve_path(c['original_audit'])/'technical/sources'/str(source)
    if sha(folder/'graph.npz') != read(folder/'graph-complete.json')['sha256']:
        raise ValueError('Changed source graph: '+str(source))
    with np.load(folder/'graph.npz') as data:
        g = {k:data[k] for k in data.files}
    events = [e for e in read(folder/'events.json') if e['threshold']=='primary']
    if sha(folder/'events.json') != read(folder/'audit.json')['files']['events.json']:
        raise ValueError('Changed original establishments')
    maximum = np.zeros(len(g['start']),np.int32)
    np.maximum.at(maximum,g['frame'][1:],g['size'][1:])
    first_confirmation = min((e['confirmation_frame'] for e in events),default=len(maximum))
    return g,events,maximum,first_confirmation


def scan(c, sources):
    dissolved, positives, exclusions = [],[],[]
    span = round(c['duration_ps']/.75)
    for item in sources:
        sid = item['id'];g,events,maximum,first = source_graph(c,sid)
        directory = resolve_path(c['fate_audit'])/'technical/sources'/str(sid)
        receipt = read(directory/'complete.json')
        path = directory/'transient-episodes.json'
        if sha(path) != receipt['files'][path.name]:
            raise ValueError('Changed fate catalogue: '+str(sid))
        records = [r for r in read(path) if r['primary_failed_candidate']]
        edges=g['edges']
        if records:
            _,component=connected_components(coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),
                shape=(len(g['size']),len(g['size']))).tocsr(),directed=False)
            for r in records:
                nodes=np.flatnonzero(component==component[r['first_node']])
                peak=int(nodes[g['size'][nodes]==r['peak_atoms']][0])
                f=int(g['frame'][peak]);reasons=[]
                if f>=first:reasons.append('established crystal already in cell')
                if maximum[f]>c['maximum_initial_cluster_atoms']:reasons.append('another large cluster at restart')
                if g['crystalline_atoms'][f]/item['atom_count']>c['maximum_initial_crystal_fraction']:reasons.append('initial crystal fraction above cap')
                if r['dissolution_frame']>f+span:reasons.append('dissolution confirmation beyond common horizon')
                if f+span>=item['frame_count']:reasons.append('reference tail unavailable')
                if reasons:
                    exclusions.append(dict(episode=r['episode'],source=sid,reasons=reasons));continue
                dissolved.append(dict(source=sid,frame=f,node=peak,size=int(g['size'][peak]),label=r,
                    temperature_K=item['audit_metadata']['temperature_K'],fate_file=str(path),fate_sha256=sha(path)))
        parents=defaultdict(list)
        for parent,child,count,strong,lag in edges:
            if strong:parents[int(child)].append((int(count),-int(parent)))
        for e in events:
            if e['kind']!='isolated_establishment':continue
            node=e['birth_node']
            while parents[node]:
                node=-max(parents[node])[1];f=int(g['frame'][node]);size=int(g['size'][node])
                if e['confirmation_frame']-f>span:break
                if (f<first and f<e['birth_frame'] and f+span<item['frame_count'] and
                    size>=c['minimum_seed_atoms'] and maximum[f]<=c['maximum_initial_cluster_atoms'] and
                    g['crystalline_atoms'][f]/item['atom_count']<=c['maximum_initial_crystal_fraction']):
                    positives.append(dict(source=sid,frame=f,node=node,size=size,label=e,
                        temperature_K=item['audit_metadata']['temperature_K']))
    # One dissolved episode per ancestral source, selected by its earliest peak.
    by_source=defaultdict(list)
    for r in dissolved:by_source[r['source']].append(r)
    chosen=[min(rs,key=lambda r:(r['frame'],r['node'])) for _,rs in sorted(by_source.items())]
    for rs in by_source.values():
        for r in rs:
            if r not in chosen:exclusions.append(dict(episode=r['label']['episode'],source=r['source'],reasons=['one earliest eligible episode per source']))
    return chosen,positives,exclusions


def match(c, dissolved, candidates):
    # Positive sources are unique across blocks. They may share their own block's
    # ancestor, but never another block's ancestor.
    occupied={r['source'] for r in dissolved}
    source_ids=sorted({r['source'] for r in candidates})
    cost=np.full((len(dissolved),len(source_ids)),1e9)
    best={}
    for i,d in enumerate(dissolved):
        for j,sid in enumerate(source_ids):
            if sid in occupied and sid!=d['source']:continue
            eligible=[p for p in candidates if p['source']==sid and p['temperature_K']==d['temperature_K']
                      and abs(p['size']-d['size'])<=c['positive_size_tolerance_atoms']]
            def key(p):
                tie=int(digest(dict(seed=c['seed'],d=d['source'],p=p))[:8],16)/2**32
                return abs(p['size']-d['size'])*100+(0 if sid==d['source'] else 5)+tie
            if eligible:
                selected=min(eligible,key=key);cost[i,j]=key(selected);best[i,j]=selected
    rows,cols=linear_sum_assignment(cost)
    if len(rows)!=len(dissolved) or np.any(cost[rows,cols]>=1e9):
        raise ValueError('Cannot match all dissolved sources at exact temperature and declared size tolerance')
    return {int(i):best[i,j] for i,j in zip(rows,cols)}


def prepare(c):
    check_metric_docs(family=FAMILY)
    out,scratch,archive=roots(c);path=out/'technical/plan.json'
    if path.exists():
        plan=read(path)
        if plan['config']!=c:raise ValueError('Existing controls plan differs')
        return plan
    cache=resolve_path(c['birth_cache'])
    if read(cache/'manifest.json')['identity']!=c['birth_identity']:raise ValueError('Changed source release')
    original=read(cache/'plan.json');sources={s['id']:s for s in original['sources']}
    dissolved,positives,exclusions=scan(c,list(sources.values()))
    if len(dissolved)!=c['expected_blocks']:raise ValueError(f'Expected {c["expected_blocks"]} blocks; found {len(dissolved)}')
    matched=match(c,dissolved,positives)
    cases=[];tasks=[];blocks=[]
    for index,d in enumerate(dissolved):
        block=f'block-{index:02d}-source{d["source"]}'
        p=matched[index]
        item=sources[d['source']];g,_,maximum,_=source_graph(c,d['source'])
        span=round(c['duration_ps']/.75)
        # This filter reads only present-frame observables, never its future.
        eligible=np.flatnonzero((maximum[:len(maximum)-span]<=c['random_maximum_cluster_atoms']) &
            (g['crystalline_atoms'][:len(maximum)-span]/item['atom_count']<=c['maximum_initial_crystal_fraction']))
        if not len(eligible):raise ValueError('No eligible random full-cell liquid for source '+str(d['source']))
        rng=np.random.default_rng(c['seed']+d['source'])
        random_frame=int(rng.choice(eligible));anchor=int(rng.integers(item['atom_count']))
        liquid=dict(source=d['source'],frame=random_frame,node=None,size=0,label=None,
            temperature_K=d['temperature_K'],anchor_row=anchor,eligible_frames=eligible.tolist())
        block_cases=[]
        for cohort,selected in [('dissolved',d),('establishing',p),('random_liquid',liquid)]:
            source=selected['source'];item=sources[source];raw=raw_source(item)
            if raw.manifest['velocity_units']!='angstrom_per_ps':raise ValueError('Wrong saved velocity units')
            folder=resolve_path(c['original_audit'])/'technical/sources'/str(source)
            case=dict(key=block+'-'+cohort,block=block,cohort=cohort,source=str(source),ancestry=item['lineage'],
                role=item['role'],temperature_K=selected['temperature_K'],cadence_ps=.75,chunk_frames=32,
                raw=str(raw.root),raw_manifest_sha256=item['manifest_sha256'],audit_folder=str(folder),
                graph_sha256=sha(folder/'graph.npz'),lineage=c['lineage'],atom_count=item['atom_count'],
                start_frame=selected['frame'],stop_frame=selected['frame']+span,
                threshold=dict(size=c['establishment_atoms'],persistence_frames=1+round(c['persistence_ps']/.75)),
                selection=selected,position_dtype=str(raw.positions.dtype),velocity_dtype=str(raw.velocities.dtype))
            ref=Reference(case);f=case['start_frame'];q,box,dense=ref.frame(f)
            if cohort=='random_liquid':
                core=np.empty(0,np.int32);anchor=selected['anchor_row']
            else:
                core=np.flatnonzero(dense==selected['node'])
                if len(core)!=selected['size']:raise ValueError('Selected embryo membership changed')
                delta=q[core]-q[core[0]];delta-=box*np.rint(delta/box)
                center=np.mod(q[core[0]]+delta.mean(0),box)
                distance=q-center;distance-=box*np.rint(distance/box)
                anchor=int(np.argmin(np.sum(distance**2,axis=1)))
            case.update(core_rows=core.tolist(),anchor_row=anchor,anchor_atom_id=int(raw.atom_ids[anchor]),
                initial_cluster_atoms=selected['size'],initial_crystal_fraction=float(np.isin(ref.labels(f),[1,2,3]).mean()))
            state=dict(name=case['key'],event=case['key'],offset='fixed-12ps',start_frame=f,stop_frame=f+span,
                start_ps=f*.75,duration_ps=c['duration_ps'],cadence_ps=.75,panel=cohort,
                initial_path=str(archive/'initial'/f'{case["key"]}.npz'))
            initial=Path(state['initial_path']);initial.parent.mkdir(parents=True,exist_ok=True)
            np.savez(initial,q=q.astype(np.float32),v_A_ps=np.array(raw.velocities[f],np.float32),box=box.astype(np.float32),atom_ids=raw.atom_ids)
            state['initial_sha256']=sha(initial);case['state']=state
            cases.append(case);block_cases.append(case['key'])
            for replicate in range(c['noise_replicates']):
                seed=c['seed']+int(digest(dict(protocol=c['protocol'],case=case['key'],replicate=replicate))[:8],16)%1000000000
                tasks.append(dict(name=case['key']+f'-response450-r{replicate}',case=case['key'],state=case['key'],integrator='baoab',
                    variant='as_saved',temperature_K=c['response_temperature_K'],arms=['response450K'],replicate=replicate,seed=seed,
                    priority=(replicate,index,['establishing','dissolved','random_liquid'].index(cohort),1)))
            tasks.append(dict(name=case['key']+'-source-npt',case=case['key'],state=case['key'],integrator='npt',variant='as_saved',
                temperature_K=case['temperature_K'],arms=['sourceNPT'],replicate=0,seed=None,
                priority=(0,index,['establishing','dissolved','random_liquid'].index(cohort),0)))
            print('prepared',case['key'],'source',source,'frame',f,'seed atoms',len(core),flush=True)
        blocks.append(dict(block=block,cases=block_cases,temperature_K=d['temperature_K'],dissolved_size=d['size'],
            establishing_size=p['size'],source_ids=sorted({d['source'],p['source']}),
            ancestors=sorted({sources[d['source']]['lineage'],sources[p['source']]['lineage']})))
    seen=set()
    for block in blocks:
        if seen.intersection(block['ancestors']):raise ValueError('Matched blocks share an ancestral melt')
        seen.update(block['ancestors'])
    tasks.sort(key=lambda t:t['priority'])
    plan=dict(config=c,cases=cases,tasks=tasks,blocks=blocks,selection_exclusions=exclusions,
        potential_sha256={k:sha(resolve_path(v)) for k,v in c['meam_files'].items()},
        source_plan_sha256=sha(cache/'plan.json'),pool_release_sha256=sha(resolve_path(c['pool_release'])),
        scope=dict(original_sources_screened=len(sources),matched_blocks=len(blocks),initial_states=len(cases),
            independent_ancestors=len({s['ancestry'] for s in cases}),
            deferred=['interface-associated dissolved episodes','established nuclei that later melted','shooting controls',
                'dense 0.1/0.01 ps descendants','million-atom trajectories','relaxed views']))
    plan['identity']=digest(plan)
    write_json(path,plan);write_json(archive/'plan.json',plan)
    print('controls prepared',len(cases),'states',len(tasks),'branches',flush=True)
    return plan
