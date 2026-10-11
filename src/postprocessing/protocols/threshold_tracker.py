"""Historical v2 threshold episodes; distinct from current family lineage rules."""
import fcntl
from collections import Counter, defaultdict
import time
import numpy as np
from scipy.spatial import cKDTree
from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.postprocessing.io import arrays, geometry, read
from src.postprocessing.extraction import Frames, cluster_members, position, resolved_root

def first_streak(observations,minimum,duration,dt):
    first=previous=None
    for x in observations:
        if x['size']<minimum:first=previous=None;continue
        if previous is None or x['time_ps']-previous>dt+1e-8:first=x['time_ps']
        previous=x['time_ps']
        if previous-first>=duration-1e-8:return first,previous
    return None,None


def episodes(track,c,dt,last_time):
    """Separate excursions by a sustained return below the embryo-size threshold."""
    observations=track['observations'];high=[i for i,x in enumerate(observations) if x['size']>=c['embryo_atoms']]
    if not high:return []
    groups=[];group=[];previous=None
    for index in high:
        t=observations[index]['time_ps']
        # Below-threshold period starts one observation after the last high state.
        if previous is not None and t-previous-dt>=c['liquid_return_ps']-1e-8:
            groups.append(group);group=[]
        group.append(index);previous=t
    groups.append(group)
    result=[]
    for gi,g in enumerate(groups):
        first=observations[g[0]];last=observations[g[-1]]
        obs=observations[g[0]:g[-1]+1]
        # Missing observations break persistence. Identity links can bridge them;
        # they cannot manufacture continuously observed establishment.
        first_contact=min((x['time_ps'] for x in track['interference']
                           if x['time_ps']>=first['time_ps']),default=np.inf)
        clean=[x for x in obs if x['time_ps']<first_contact]
        a,confirm=first_streak(clean,c['establishment_atoms'],c['establishment_persistence_ps'],dt)
        _,ec=first_streak(obs,c['embryo_atoms'],c['embryo_persistence_ps'],dt)
        _,gc=first_streak(clean,c['growth_atoms'],c['growth_persistence_ps'],dt)
        return_time=last['time_ps']+dt+c['liquid_return_ps']
        interference=[x for x in track['interference'] if first['time_ps']<=x['time_ps']<=return_time+1e-8]
        # Neither a merge nor a split is a demonstrated dissolution.
        dissolved=return_time<=last_time+1e-8 and not interference
        if dissolved:status='dissolved'
        elif interference or gi<len(groups)-1:status='unresolved_interaction'
        elif last_time-last['time_ps']<dt+1e-8:status='present_at_end'
        else:status='right_censored'
        peak=max(obs,key=lambda x:x['size'])
        kind=track['origin'] if gi==0 else 'reappearance_same_track'
        row=dict(track=track['id'],episode=gi,origin=kind,first_time_ps=first['time_ps'],
            last_high_time_ps=last['time_ps'],peak_atoms=peak['size'],peak_time_ps=peak['time_ps'],
            embryo_confirmation_ps=ec,establishment_start_ps=a,establishment_confirmation_ps=confirm,
            growth_confirmation_ps=gc,terminal_status=status,
            dissolution_confirmation_ps=return_time if dissolved else None,
            fate=('grew_then_dissolved' if gc is not None and dissolved else 'dissolved' if dissolved
                  else 'documented_growth' if gc is not None else status),
            nearest_established_crystal_A=track['nearest_established_crystal_A'],
            weak_ancestry=track['weak_ancestry'],parent_tracks=track['parent_tracks'],
            interaction_count=len(interference),left_censored=kind=='parent_present',
            first_frame=first['frame'],first_component=first['component'],
            peak_frame=peak['frame'],peak_component=peak['component'],centroid_A=first['centroid_A'],
            periodic_extent_ambiguous=any(x['periodic_extent_ambiguous'] for x in obs),
            establishment_sensitivity={str(n):first_streak(clean,n,c['establishment_persistence_ps'],dt)[1]
                for n in c['establishment_sensitivity_atoms']})
        result.append(row)
    return result


def track(p,task):
    c=p['config'];s=next(s for s in p['sources'] if s['id']==task['source']);dt=c['cadences'][task['cadence']]
    root=resolved_root(c['output'])/'technical/sources'/s['id']/task['cadence']/task['detector']
    root.mkdir(parents=True,exist_ok=True);identity=digest(dict(plan=p['identity'],task=task))
    receipt=root/'complete.json'
    with (root/'track.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists():
            done=read(receipt)
            if done['identity']!=identity or any(sha(root/n)!=h for n,h in done['files'].items()):raise ValueError('Changed tracked result')
            return done
        started=time.monotonic();access=Frames(p,s,task['detector'])
        tracks={};active={};next_id=1;statistics=[]
        a=arrays(s['parts'][0]['path'],s['parts'][0]['manifest_sha256']);atom_ids=a['atom_ids'].copy()
        frames=s['views'][task['cadence']]
        for fi,frame in enumerate(frames):
            t=s['times_ps'][frame];labels=access.labels(frame);groups=cluster_members(labels)
            active={k:v for k,v in active.items() if t-v['time_ps']<=c['maximum_link_interval_ps']+1e-8}
            positions,box=geometry(s,frame)
            parents=defaultdict(list);children=defaultdict(list);weak=defaultdict(list)
            for tid,old in active.items():
                ids,counts=np.unique(labels[old['indices']],return_counts=True)
                for cid,count in zip(ids,counts):
                    if cid==0:continue
                    ratio=float(count)/min(len(old['indices']),len(groups[int(cid)]))
                    record=(tid,int(cid),int(count),ratio,t-old['time_ps'])
                    if count>=c['minimum_shared_atoms'] and ratio>=c['minimum_overlap_fraction']:
                        parents[int(cid)].append(record);children[tid].append(record)
                    else:weak[int(cid)].append(tid)
            # Prefer the most recent observation, then maximal shared membership.
            edges=sorted((r for rows in parents.values() for r in rows),key=lambda r:(r[4],-r[2],-r[3],r[0],r[1]))
            assigned={};taken=set()
            for tid,cid,count,ratio,lag in edges:
                if tid not in taken and cid not in assigned:assigned[cid]=tid;taken.add(tid)
            updates={};established_atoms=[]
            # Only establishments confirmed before this observation can mark a
            # pre-existing interface; future confirmation never travels backward.
            for cid,tid in assigned.items():
                if tracks[tid]['confirmed_ps'] is not None and tracks[tid]['confirmed_ps']<t-1e-8:
                    established_atoms.extend(groups[cid].tolist())
            existing=cKDTree(positions[established_atoms],boxsize=box) if established_atoms else None
            for cid,indices in groups.items():
                old_parents=sorted(set(r[0] for r in parents[cid]))
                if cid not in assigned:
                    tid=next_id;next_id+=1
                    distance=float(existing.query(positions[indices],workers=1)[0].min()) if existing is not None else None
                    kind=('parent_present' if fi==0 else 'split_descendant' if old_parents
                          else 'uncertain_ancestry' if weak[cid] else 'interface_associated'
                          if distance is not None and distance<=c['interface_distance_A'] else 'isolated')
                    tracks[tid]=dict(id=tid,origin=kind,parent_tracks=old_parents,weak_ancestry=bool(weak[cid]),
                        nearest_established_crystal_A=distance,observations=[],interference=[],confirmed_ps=None,
                        last_large_ps=None,large_start_ps=None)
                else:tid=assigned[cid]
                if len(old_parents)>1:
                    for parent in old_parents:
                        tracks[parent]['interference'].append(dict(time_ps=t,kind='merge',tracks=old_parents))
                        if parent!=tid and parent not in taken:active.pop(parent,None)
                for parent in old_parents:
                    if len(children[parent])>1:
                        tracks[parent]['interference'].append(dict(time_ps=t,kind='split'))
                center,ambiguous=position(positions,box,indices)
                ob=dict(time_ps=t,frame=frame,size=len(indices),centroid_A=center.tolist(),
                    periodic_extent_ambiguous=ambiguous,component=cid)
                tr=tracks[tid];tr['observations'].append(ob)
                # Linear-time running establishment counter, separate from fate.
                if len(indices)>=c['establishment_atoms']:
                    prev=tr['last_large_ps'];start=tr['large_start_ps']
                    if prev is None or t-prev>dt+1e-8:start=t
                    tr['large_start_ps']=start;tr['last_large_ps']=t
                    if tr['confirmed_ps'] is None and t-start>=c['establishment_persistence_ps']-1e-8:tr['confirmed_ps']=t
                else:tr['last_large_ps']=None
                updates[tid]=dict(indices=indices,time_ps=t)
            active.update(updates)
            statistics.append(dict(frame=frame,time_ps=t,tracked_clusters=len(groups),
                tracked_atoms=int(np.count_nonzero(labels)),largest_cluster=max((len(v) for v in groups.values()),default=0)))
            if fi%32==0 or fi==len(frames)-1:
                write_json(root/'progress.json',dict(frames=fi+1,total_frames=len(frames),tracks=len(tracks),seconds=time.monotonic()-started))
        records=[]
        for tr in tracks.values():
            for e in episodes(tr,c,dt,s['times_ps'][frames[-1]]):
                records.append(dict(source=s['id'],dataset=s['dataset'],ancestor=s['ancestry_group'],
                    role=s['role'],potential=s['potential'],source_origin=s['origin'],
                    cadence_ps=dt,native_cadence_ps=s['native_cadence_ps'],
                    detector=task['detector'],**e))
        # Read each chunk once in temporal order instead of decompressing the
        # whole cell again for every embryo's onset and peak.
        memberships=defaultdict(list)
        for e in records:
            for name in ('first','peak'):
                memberships[e[name+'_frame']].append((e,name))
        for frame,requests in sorted(memberships.items()):
            groups=cluster_members(access.labels(frame))
            for e,name in requests:
                e[name+'_members']=atom_ids[groups[e[name+'_component']]].tolist()
        # Keep memberships only at event onset/peak, sufficient for paired event
        # matching; complete frame-level memberships are in the extraction bank.
        write_json(root/'events.json',records);write_json(root/'trajectory.json',statistics)
        counts=Counter(events=len(records),isolated=sum(r['origin']=='isolated' for r in records),
            established=sum(r['establishment_confirmation_ps'] is not None for r in records),
            dissolved=sum(r['terminal_status']=='dissolved' for r in records))
        done=dict(identity=identity,**task,counts=dict(counts),seconds=time.monotonic()-started,
            files={n:sha(root/n) for n in ('events.json','trajectory.json')})
        write_json(receipt,done);arrays.cache_clear()
        return done
