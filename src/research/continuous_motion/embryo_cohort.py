"""Present-selected subcritical embryos and prospective establishment labels."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import REPO, resolve_path
from src.experiment_runner.execution import ExecutionBundle, allocation_deadline, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.research.response_dense_birth.data import Access, read

FAMILY='embryo_establishment_cohort'
OWNER='current response/nucleus-recovery chat; authorized separate embryo-to-nucleus cohort'
MODULE='src.research.continuous_motion.embryo_cohort'


def output(c):return resolve_path(c['output'])
def cache(c):return resolve_path(c['cache'])


def prepare(c):
    upstream=resolve_path(c['source_plan']);old=read(upstream)
    if old['identity']!=c['source_identity']:raise ValueError('Changed pinned source inventory')
    sources=[{k:v for k,v in item.items() if k!='reuse'} for item in old['sources']]
    releases={r['root']:r for r in old['releases']}
    for release in releases.values():
        path=Path(release['root'])/'technical/plan.json'
        if sha(path)!=release['plan_sha256']:raise ValueError('Changed lineage release')
        rules=read(path)['config']
        if (rules['embryo_atoms'],rules['establishment_atoms'],rules['persistence_ps'])!=(8,64,c['persistence_ps']):
            raise ValueError('Incompatible lineage thresholds')
    split=read(resolve_path(c['fixed_release'])/'splits.json')
    if split['identity']!=c['fixed_identity']:raise ValueError('Changed Al64 split')
    roles={s['lineage']:s['role'] for s in split['sources']}
    for item in sources:
        s=item['source']
        if roles[s['ancestry_group']]!=s['role']:raise ValueError('Ancestry role changed')
        if sha(item['parent_path'])!=releases[item['release']]['parent_sha256']:raise ValueError('Changed parent inventory')
        np.testing.assert_allclose(np.diff(np.asarray(s['times_ps'])[s['views']['010']]),.1,atol=1e-8,rtol=0)
        done=read(Path(item['label_folder'])/'complete.json')
        if done['identity']!=item['label_identity']:raise ValueError('Changed source labels')
    p=dict(config=c,sources=sources,releases=old['releases'],excluded=old['excluded'],
        upstream_sha256=sha(upstream),implementations=check_metric_docs(family=FAMILY)[FAMILY]['files'])
    p['identity']=digest(p);path=output(c)/'technical/plan.json'
    if path.exists() and read(path)!=p:raise ValueError('Changed cohort plan; use new revision')
    write_json(path,p);cache(c).mkdir(parents=True,exist_ok=True)
    write_json(output(c)/'technical/prediction-context.json',dict(
        encoder=dict(positions='unrelaxed nearest80 centered on a present embryo member',velocities='native same-atom same-frame A/ps',history=1,conditions=[],relaxation=False),
        predictor=dict(inputs='not fitted in collection; prospective frozen readouts must use only declared Q/V representations or size/structure controls',conditions=[],horizons_ps=c['horizons_ps'],horizon_as_covariate=False),
        observation='One current-state center per 8–63 atom root-family largest component at fixed 1 ps landmarks; no future-outcome center sampling',
        target='Same connected descendant begins a 64+ atom 1.5 ps persistent interval within 1/3/6 ps',
        censoring='Common 7.5 ps source followup; unresolved/absorbed/coalesced family loss before endpoint ascertainment is excluded, not labelled a failure',
        spatial_veto='No 64+ current component or known already-established family largest component within 8 A of center; fragmented established-family frames conservatively excluded',
        split='Original Al64 roles retained; future evaluation pools validation/calibration/test, fitting uses train ancestors only',
        independent_units='Melt ancestors; repeated frames, family members and archive views are not independent events'))
    return p


def collect_source(c,p,item):
    folder=cache(c)/'sources'/item['key'];folder.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(plan=p['identity'],source=item['key']))
    if (folder/'complete.json').exists():
        d=read(folder/'complete.json')
        if d['identity']!=identity or any(sha(folder/f)!=h for f,h in d['files'].items()):raise ValueError('Changed source packet')
        return d
    labelroot=Path(item['label_folder']);receipt=read(labelroot/'complete.json')
    for name in ['events.json','sizes.jsonl.gz']:
        if sha(labelroot/name)!=receipt['files'][name]:raise ValueError('Changed lineage artifact '+name)
    s=item['source'];events={e['track']:e for e in read(labelroot/'events.json')}
    spacing=c['observation_spacing_ps'];horizons=np.asarray(c['horizons_ps']);follow=horizons.max()+c['persistence_ps']
    by_frame=defaultdict(list);counts=Counter();times=np.asarray(s['times_ps'])
    with gzip.open(labelroot/'sizes.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);t=row['time_ps']
            if abs(t/spacing-round(t/spacing))<1e-8:by_frame[int(row['frame'])].append(row)
    access=Access(read(item['parent_path']),s);ids=access.arrays['atom_ids'];family_ids={}
    rows=[];clouds=[];velocities=[]
    deadline=read(output(c)/'technical/launch.json')['deadline_unix']
    for frame,traces in sorted(by_frame.items()):
        if time.time()>deadline-120:raise TimeoutError('Source collection deadline; completed packets retained')
        t=times[frame]
        if times[-1]<t+follow-1e-8:counts['landmarks_without_common_followup']+=1;continue
        eligible=[];established_cids=[];fragmented_established=False
        for trace in traces:
            e=events[trace['family']];confirmation=e['establishment_confirmation_ps']
            established=(confirmation is not None and confirmation<=t+1e-8) or e['inherited_established_crystal'] or (e['origin']=='parent_present' and len(e['first_members'])>=64)
            if established:
                established_cids.append(trace['largest_component'])
                fragmented_established |= trace['components']>1
            if e['record_kind']!='root' or e['origin']!='emergence_from_liquid' or not 8<=trace['size']<64:continue
            counts['present_subcritical_candidates']+=1
            start=e['establishment_start_ps']
            if established or (start is not None and start<=t+1e-8):counts['already_established']+=1;continue
            terminal=e['terminal_time_ps']
            if e['terminal_status'] in ('absorbed','coalesced','identity_unresolved') and terminal<=t+follow+1e-8 and (confirmation is None or confirmation>terminal+1e-8):
                counts['competing_or_unresolved_fate']+=1;continue
            eligible.append((trace,e))
        if not eligible:continue
        if fragmented_established:
            counts['unknown_established_fragment_locations']+=len(eligible);continue
        q,box,lab,_=access.frame(frame);sizes=np.bincount(lab);sizes[0]=0
        excluded_atoms=np.flatnonzero((sizes[lab]>=64)|np.isin(lab,established_cids))
        tree=cKDTree(q[excluded_atoms],boxsize=box) if len(excluded_atoms) else None
        selected=[]
        for trace,e in eligible:
            members=np.flatnonzero(lab==trace['largest_component'])
            if len(members)!=trace['size']:raise ValueError('Trace/component size mismatch')
            delta=q[members]-q[members[0]];delta-=box*np.rint(delta/box)
            centroid=delta.mean(0);dist=np.linalg.norm(delta-centroid,axis=1)
            center=int(members[np.argmin(dist)])
            clearance=float(tree.query(q[center],workers=1)[0]) if tree is not None else None
            if clearance is not None and clearance<=c['radius_A']:counts['near_established_or64_component']+=1;continue
            selected.append((center,trace,e,clearance))
        if not selected:continue
        atoms=np.array([r[0] for r in selected])
        _,neighbors=cKDTree(q,boxsize=box).query(q[atoms],k=80,workers=1)
        if not np.array_equal(neighbors[:,0],atoms):raise ValueError('Nearest-neighbor identity mismatch')
        position=q[neighbors]-q[atoms,None];position-=box*np.rint(position/box)
        position=position.astype(np.float32)
        # Full-state identity guards deduplication of historical partial views.
        pi,raw=s['frames'][frame]
        velocity=np.asarray(access.velocities[pi][raw,neighbors],np.float32)
        full_state=digest(dict(q=hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),
            v=hashlib.sha256(np.ascontiguousarray(np.asarray(access.velocities[pi][raw],np.float32)).tobytes()).hexdigest(),box=box.tolist()))
        for j,(atom,trace,e,clearance) in enumerate(selected):
            fid=trace['family']
            if fid not in family_ids:
                fq,fb,_,_=access.frame(e['first_frame']);part,rf=s['frames'][e['first_frame']]
                fv=np.asarray(access.velocities[part][rf],np.float32)
                family_ids[fid]='embryo-'+digest(dict(ancestor=s['ancestry_group'],time_ps=round(e['first_time_ps'],8),members=e['first_members'],
                    q=hashlib.sha256(np.ascontiguousarray(fq).tobytes()).hexdigest(),v=hashlib.sha256(fv.tobytes()).hexdigest(),box=fb.tolist()))[:32]
            physical=family_ids[fid];start=e['establishment_start_ps']
            y=[int(start is not None and t<start<=t+h+1e-8) for h in horizons]
            if any(a>b for a,b in zip(y,y[1:])):raise ValueError('Non-nested future labels')
            sample=digest(dict(family=physical,time_ps=round(float(t),8),atom=int(ids[atom]),state=full_state))
            rows.append(dict(id=sample,event=physical,event_view=item['key']+':'+str(fid),source=s['id'],dataset=s['dataset'],ancestor=s['ancestry_group'],
                role=s['role'],frame=frame,time_ps=float(t),atom=int(ids[atom]),size=int(trace['size']),label=y,index=len(clouds),
                nearest_excluded_component_A=clearance,establishment_start_ps=start,establishment_confirmation_ps=e['establishment_confirmation_ps'],
                terminal_status=e['terminal_status'],terminal_time_ps=e['terminal_time_ps'],native_cadence_ps=s['native_cadence_ps'],label_cadence_ps=.1))
            clouds.append(position[j]);velocities.append(velocity[j]);counts['accepted_observations']+=1
        write_json(folder/'progress.json',dict(frame=frame,observations=len(rows),counts=dict(counts)))
    for name,data in [('positions',clouds),('velocities',velocities)]:
        array=np.asarray(data,np.float32).reshape(-1,80,3)
        if not np.isfinite(array).all():raise FloatingPointError('Nonfinite '+name)
        np.save(folder/(name+'.npy'),array)
    write_json(folder/'rows.json',rows)
    d=dict(identity=identity,source=s['id'],ancestor=s['ancestry_group'],role=s['role'],counts=dict(counts),observations=len(rows),
        physical_families=len(set(r['event'] for r in rows)),files={n:sha(folder/n) for n in ['rows.json','positions.npy','velocities.npy']})
    write_json(folder/'complete.json',d);return d


def seal(c,p):
    rows=[];positions=[];velocities=[];seen={};source_counts=[];duplicates=[]
    for item in p['sources']:
        folder=cache(c)/'sources'/item['key'];d=read(folder/'complete.json')
        if d['identity']!=digest(dict(plan=p['identity'],source=item['key'])) or any(sha(folder/f)!=h for f,h in d['files'].items()):raise ValueError('Changed source packet')
        source_counts.append(dict(source=d['source'],role=d['role'],ancestor=d['ancestor'],observations=d['observations'],physical_families=d['physical_families'],**d['counts']))
        q=np.load(folder/'positions.npy');v=np.load(folder/'velocities.npy')
        for row in read(folder/'rows.json'):
            if row['id'] in seen:
                old=seen[row['id']]
                if row['label']!=old['label'] or row['role']!=old['role']:raise ValueError('Identical full state has contradictory labels/roles')
                duplicates.append(dict(id=row['id'],retained_source=old['source'],dropped_source=row['source']));continue
            seen[row['id']]=row;index=row['index'];rows.append(dict(row,index=len(rows)));positions.append(q[index]);velocities.append(v[index])
    if not rows:raise ValueError('No eligible embryos')
    count=Counter(r['event'] for r in rows)
    data={k:np.asarray([r[k] for r in rows]) for k in ['id','event','event_view','source','ancestor','role','frame','time_ps','atom','size','label','index']}
    data['weight']=np.asarray([1/count[r['event']] for r in rows],np.float64)
    data['indices']=np.repeat(data['index'][:,None],len(c['horizons_ps']),axis=1)
    for ancestor in set(data['ancestor']):
        if len(set(data['role'][data['ancestor']==ancestor]))!=1:raise ValueError('Ancestor crosses roles')
    np.save(cache(c)/'positions.npy',np.asarray(positions,np.float32));np.save(cache(c)/'velocities.npy',np.asarray(velocities,np.float32))
    np.savez_compressed(cache(c)/'rows.npz',**data);write_json(cache(c)/'rows.json',rows)
    summary=[]
    for role in ['train','selection','calibration','test','pooled_test']:
        m=data['role']!='train' if role=='pooled_test' else data['role']==role
        for j,h in enumerate(c['horizons_ps']):
            positive=m&(data['label'][:,j]==1)
            summary.append(dict(role=role,horizon_ps=h,observations=int(m.sum()),embryo_families=len(set(data['event'][m])),ancestors=len(set(data['ancestor'][m])),
                positive_observations=int(positive.sum()),positive_families=len(set(data['event'][positive])),positive_ancestors=len(set(data['ancestor'][positive])),
                family_weighted_prevalence=float(np.average(data['label'][m,j],weights=data['weight'][m])) if m.any() else None))
    out=output(c)/'analyses/cohort-v1'
    for name,values,columns in [('sources',source_counts,None),('summary',summary,None),('duplicates',duplicates,['id','retained_source','dropped_source'])]:
        if columns is None:columns=list(dict.fromkeys(k for row in values for k in row))
        write_metric_rows(values,out,family=FAMILY,name=name,columns=columns)
    done=dict(identity=p['identity'],state='complete',observations=len(rows),physical_families=len(count),melt_ancestors=len(set(data['ancestor'])),
        source_views=len(p['sources']),summary=summary,velocity_units='angstrom_per_ps',files={n:sha(cache(c)/n) for n in ['positions.npy','velocities.npy','rows.npz','rows.json']})
    write_json(cache(c)/'complete.json',done);write_json(output(c)/'technical/complete.json',done)


def run(c):
    p=read(output(c)/'technical/plan.json')
    with recorded_stage(output(c)/'technical/coordinator.json',owner=OWNER,stage='collection') as progress:
        with ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(collect_source,c,p,item) for item in p['sources']]
            for i,future in enumerate(as_completed(futures),1):
                d=future.result();progress.update(sources_complete=i,total_sources=len(futures),last_source=d['source']);print('collected',d['source'],d['observations'],flush=True)
        progress.update(stage='sealing');seal(c,p)


def launch(c):
    prepare(c);tech=output(c)/'technical'
    if (tech/'launch.json').exists():raise FileExistsError('Cohort already launched')
    deadline=min(time.time()+c['wall_seconds'],allocation_deadline(reserve_seconds=300))
    if deadline-time.time()<600:raise TimeoutError('Less than ten minutes remain in allocation')
    bundle=ExecutionBundle.freeze(REPO,tech/'code',c,directories=('src','docs/metrics'))
    receipt=dict(owner=OWNER,started_unix=time.time(),deadline_unix=deadline,allocation=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,code=str(bundle.root))
    write_json(tech/'launch.json',receipt)
    command=[sys.executable,'-u','-m',MODULE,'run','--config',str(bundle.config_path)]
    env=dict(os.environ,PCM_PROJECT_ROOT=str(REPO),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1')
    with (tech/'coordinator.log').open('a') as log:
        child=subprocess.Popen(command,cwd=bundle.root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(pid=child.pid,process_group=child.pid,command=command);write_json(tech/'launch.json',receipt);print(receipt,flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=['prepare','launch','run']);parser.add_argument('--config',required=True)
    a=parser.parse_args();c=read(a.config);check_metric_docs(family=FAMILY);globals()[a.stage](c)


if __name__=='__main__':main()
