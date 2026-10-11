"""Embryo births with continuing descendant families and separate interaction fates.

Consumes immutable v2 component labels; never recomputes atom classifications.
"""
from collections import Counter, defaultdict
import fcntl
import gzip
import json
import time

import numpy as np
from scipy.spatial import cKDTree

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.postprocessing.io import read
from src.postprocessing.extraction import Frames, cluster_members, position, resolved_root
from src.postprocessing.io import arrays, geometry
from src.postprocessing.membership import overlap_counts


def run(plan, parent, task):
    c=plan['config'];s=next(s for s in parent['sources'] if s['id']==task['source'])
    dt=parent['config']['cadences'][task['cadence']]
    root=resolved_root(c['output'])/'technical/sources'/s['id']/task['cadence']/task['detector']
    root.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(plan=plan['identity'],task=task))
    with (root/'track.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=root/'complete.json'
        if receipt.exists():
            old=read(receipt)
            if old['identity']!=identity or any(sha(root/n)!=v for n,v in old['files'].items()):
                raise ValueError(f'Changed lineage result: {root}')
            return old
        started=time.monotonic();access=Frames(parent,s,task['detector'])
        atom_ids=arrays(s['parts'][0]['path'],s['parts'][0]['manifest_sha256'])['atom_ids']
        frames=s['views'][task['cadence']];t0=s['times_ps'][frames[0]]
        families={};nodes={};events=[];event_by_family={};interactions=[];statistics=[]
        next_family=1;next_node=1
        # Only live components and candidate records survive in memory; transient
        # sub-eight-atom roots are discarded when their reset is confirmed.
        trace_path=root/'sizes.jsonl.gz'
        trace=gzip.open(trace_path,'wt')

        def new_family(t, kind='root', founders=(), uncertain=False):
            nonlocal next_family
            fid=next_family;next_family+=1
            families[fid]=dict(id=fid,kind=kind,founders=list(founders),record=None,
                first_seen=t,last_seen=t,last_n=0,starts={},previous=None,
                established=False,uncertain_crystal=uncertain,ambiguous_loss=False,
                ended=False,interactions=0,fragments=0,inherited_established=False)
            return fid

        def finish(fid,status,t,receiver=None):
            f=families[fid]
            if f['ended']:return
            f['ended']=True
            e=f['record']
            if e is not None:
                e.update(terminal_status=status,terminal_time_ps=t,receiver_family=receiver,
                    interaction_count=f['interactions'],fragmentation_count=f['fragments'])
                est=e['establishment_confirmation_ps'] is not None
                e['fate']=('grew_then_dissolved' if status=='dissolved' and e['growth_confirmation_ps'] is not None
                    else 'dissolved' if status=='dissolved'
                    else 'absorbed_after_establishment' if status=='absorbed' and est
                    else 'absorbed_before_establishment' if status=='absorbed'
                    else 'coalesced_after_establishment' if status=='coalesced' and est
                    else 'coalesced_before_establishment' if status=='coalesced'
                    else 'documented_growth' if status=='present_at_end' and e['growth_confirmation_ps'] is not None
                    else status)

        try:
            for fi,frame in enumerate(frames):
                t=s['times_ps'][frame];labels=access.labels(frame);groups=cluster_members(labels)
                # Reset only after an observed, sustained absence of every
                # connected >=4-atom descendant. Small label dips do not rebirth.
                for fid,f in list(families.items()):
                    if not f['ended'] and t-f['last_seen']>dt+c['reset_ps']+1e-8:
                        finish(fid,'identity_unresolved' if f['ambiguous_loss'] else 'dissolved',
                            f['last_seen']+dt+c['reset_ps'])
                nodes={nid:n for nid,n in nodes.items() if not families[n['family']]['ended']
                       and t-n['time']<=dt+c['reset_ps']+1e-8}
                adjacent=defaultdict(list);gaps=defaultdict(list);weak_crystal=set();weak_counts=Counter()
                overlap_size=max(groups,default=0)+1
                overlap_workspace=np.zeros(overlap_size,np.int64)
                overlap_touched=np.empty(overlap_size,np.int32)
                for nid,n in nodes.items():
                    ids,counts=overlap_counts(labels,n['indices'],overlap_workspace,overlap_touched)
                    for cid,shared in zip(ids,counts):
                        if not cid:continue
                        cid=int(cid);shared=int(shared)
                        rp=shared/len(n['indices']);rc=shared/len(groups[cid])
                        if shared>=c['minimum_shared_atoms'] and max(rp,rc)>=c['continuation_fraction']:
                            edge=(nid,n['family'],shared,rp,rc)
                            (adjacent if abs(t-n['time']-dt)<1e-8 else gaps)[cid].append(edge)
                        else:
                            weak_counts[cid]+=1
                            if families[n['family']]['established']:weak_crystal.add(cid)
                parents={cid:adjacent[cid] if adjacent[cid] else gaps[cid] for cid in groups}
                edges=sorted(((0 if adjacent[cid] else 1,-e[2],-min(e[3],e[4]),e[0],cid)
                    for cid,es in parents.items() for e in es))
                selected={};used=set()
                for _,_,_,nid,cid in edges:
                    if nid not in used and cid not in selected:selected[cid]=nid;used.add(nid)
                # A non-primary split child inherits its predecessor family;
                # splitting cannot invent a new birth.
                component_family={};merge_specs={};prior_sizes=defaultdict(int)
                for n in nodes.values():
                    if abs(t-n['time']-dt)<1e-8:prior_sizes[n['family']]+=len(n['indices'])
                for cid,indices in groups.items():
                    es=parents[cid]
                    if not es:
                        component_family[cid]=new_family(t,uncertain=cid in weak_crystal);continue
                    chosen=selected.get(cid)
                    if chosen is None:chosen=max(es,key=lambda e:(e[2],min(e[3],e[4]),-e[0]))[0]
                    fid=nodes[chosen]['family'];component_family[cid]=fid
                    if not adjacent[cid]:continue
                    weights=Counter()
                    for _,family,shared,_,_ in adjacent[cid]:weights[family]+=shared
                    if len(weights)>1:
                        dominant=max(weights,key=lambda f:(weights[f],prior_sizes[f],-f))
                        component_family[cid]=dominant
                        # Contribution relative to the dominant incoming body:
                        # trace contact is not the same as comparable coalescence.
                        major=[f for f in weights if f==dominant or
                            (weights[f]>=c['embryo_atoms'] and
                             weights[f]>=c['minor_parent_fraction']*weights[dominant]) or
                            families[f]['established']]
                        merge_specs[cid]=(dominant,major,weights)
                # Only completely consumed families terminate. A family with a
                # separate surviving fragment retains that fragment and its fate.
                survivors=defaultdict(set)
                for cid,fid in component_family.items():survivors[fid].add(cid)
                consumed=set();composite_for={}
                for cid,(dominant,major,weights) in merge_specs.items():
                    distinct=[f for f in major if f not in consumed]
                    established_incoming=[f for f in weights if families[f]['established']]
                    if len(distinct)>1:
                        # Composite families can grow, but they are not a third
                        # embryo birth and never retroactively establish donors.
                        key=tuple(sorted(distinct))
                        if key not in composite_for:
                            founders=sorted({a for f in distinct for a in (families[f]['founders'] or [f])})
                            composite_for[key]=new_family(t,'composite',founders)
                        receiver=composite_for[key];component_family[cid]=receiver
                        families[receiver]['established']=bool(established_incoming)
                        families[receiver]['inherited_established']=bool(established_incoming)
                    else:receiver=dominant
                    for donor in weights:
                        if donor==receiver:continue
                        donor_has_other=bool(survivors[donor]-{cid})
                        kind=('fragment_transfer' if donor_has_other else
                              'coalesced' if donor in distinct and len(distinct)>1 else 'absorbed')
                        # Joining a pre-existing crystal is absorption for an
                        # unestablished donor, irrespective of the size ratio.
                        if established_incoming and not families[donor]['established'] and not donor_has_other:kind='absorbed'
                        if donor not in consumed:
                            families[donor]['interactions']+=1
                            interactions.append(dict(time_ps=t,frame=frame,component=cid,donor=donor,
                                receiver=receiver,kind=kind,shared_atoms=int(weights[donor]),
                                incoming_fraction=weights[donor]/max(1,prior_sizes[donor]),
                                outgoing_fraction=weights[donor]/len(groups[cid])))
                            if not donor_has_other:
                                finish(donor,kind,t,receiver);consumed.add(donor)
                    families[receiver]['interactions']+=1
                # Establishment belongs to a connected descendant, never the
                # summed size of disconnected fragments.
                current=defaultdict(list)
                for cid,fid in component_family.items():current[fid].append(cid)
                # Membership tracking, persistence and fates use the sealed
                # component labels. Coordinates are needed only when recording
                # an onset centroid and its distance to established crystal.
                needs_geometry=any(families[fid]['record'] is None and
                    max(len(groups[cid]) for cid in cids)>=c['embryo_atoms']
                    for fid,cids in current.items())
                if needs_geometry:
                    x,box=geometry(s,frame)
                    # Capture establishment BEFORE processing this frame's
                    # milestones, exactly as in the original eager calculation.
                    old_crystal_indices=[groups[cid] for cid,fid in component_family.items() if families[fid]['established']]
                    old_crystal=cKDTree(x[np.concatenate(old_crystal_indices)],boxsize=box) if old_crystal_indices else None
                for fid,cids in current.items():
                    f=families[fid]
                    if f['ended']:raise ValueError(f'Consumed family still assigned: {fid}/{frame}')
                    cid=max(cids,key=lambda q:len(groups[q]));indices=groups[cid];n=len(indices)
                    if len(cids)>1:f['fragments']+=1
                    if f['record'] is None and n>=c['embryo_atoms']:
                        distance=float(old_crystal.query(x[indices],workers=1)[0].min()) if old_crystal is not None else None
                        origin=('composite' if f['kind']=='composite' else 'parent_present' if fi==0
                            else 'unresolved_history' if t-t0<c['prehistory_ps']-1e-8
                            else 'unresolved_crystal_connection' if f['uncertain_crystal']
                            else 'near_interface_emergence' if distance is not None and distance<=c['interface_distance_A']
                            else 'emergence_from_liquid')
                        center,ambiguous=position(x,box,indices)
                        e=dict(source=s['id'],dataset=s['dataset'],ancestor=s['ancestry_group'],
                            source_origin=s['origin'],potential=s['potential'],role=s['role'],
                            cadence_ps=dt,native_cadence_ps=s['native_cadence_ps'],detector=task['detector'],
                            track=fid,episode=0,record_kind=f['kind'],
                            is_birth_record=f['kind']=='root' and origin in ('emergence_from_liquid','near_interface_emergence'),origin=origin,
                            founder_families=f['founders'],first_time_ps=t,onset_interval_start_ps=None if fi==0 else t-dt,
                            first_frame=frame,first_component=cid,first_members=atom_ids[indices].tolist(),
                            centroid_A=center.tolist(),periodic_extent_ambiguous=ambiguous,
                            nearest_established_crystal_A=distance,weak_overlap_count=weak_counts[cid],
                            peak_atoms=n,peak_time_ps=t,peak_frame=frame,peak_component=cid,
                            last_high_time_ps=t,last_visible_time_ps=t,
                            establishment_start_ps=None,establishment_confirmation_ps=None,
                            embryo_confirmation_ps=None,growth_confirmation_ps=None,milestones={},
                            terminal_status=None,terminal_time_ps=None,fate=None,receiver_family=None,
                            interaction_count=0,fragmentation_count=0,left_censored=fi==0)
                        e.update(descendant_establishment_confirmation_ps=None,
                            inherited_established_crystal=f['inherited_established'])
                        f['record']=e;events.append(e);event_by_family[fid]=e
                    e=f['record']
                    if e is not None:
                        e['last_visible_time_ps']=t
                        if n>=c['embryo_atoms']:e['last_high_time_ps']=t
                        if n>e['peak_atoms']:e.update(peak_atoms=n,peak_time_ps=t,peak_frame=frame,peak_component=cid)
                        for threshold in c['thresholds']:
                            if n<threshold:f['starts'].pop(threshold,None);continue
                            if f['previous'] is None or abs(t-f['previous']-dt)>1e-8:f['starts'][threshold]=t
                            start=f['starts'].setdefault(threshold,t)
                            for duration in c['durations_ps']:
                                key=f'{threshold}@{duration:g}ps'
                                if key not in e['milestones'] and t-start>=duration-1e-8:
                                    e['milestones'][key]=dict(start_ps=start,confirmation_ps=t)
                                    if duration==c['persistence_ps']:
                                        if threshold==c['embryo_atoms']:e['embryo_confirmation_ps']=t
                                        if threshold==c['establishment_atoms']:
                                            e['establishment_start_ps']=start;e['establishment_confirmation_ps']=t;f['established']=True
                                            if not f['inherited_established']:
                                                for founder in f['founders'] or [fid]:
                                                    original=event_by_family.get(founder)
                                                    if original is not None and original['descendant_establishment_confirmation_ps'] is None:
                                                        original['descendant_establishment_confirmation_ps']=t
                                        if threshold==c['growth_atoms']:e['growth_confirmation_ps']=t
                        trace.write(json.dumps(dict(family=fid,time_ps=t,frame=frame,largest_component=cid,
                            size=n,components=len(cids)),separators=(',',':'))+'\n')
                    # Initial macroscopic crystal is pre-existing evidence now;
                    # confirmation is still recorded separately in milestones.
                    if fi==0 and n>=c['establishment_atoms']:f['established']=True
                    f.update(last_seen=t,last_n=n,previous=t)
                updated={};touched=set();selected_nodes=set(selected.values())
                for cid,fid in component_family.items():
                    nid=selected.get(cid)
                    if nid is None or nid in touched:nid=next_node;next_node+=1
                    touched.add(nid);updated[nid]=dict(family=fid,indices=groups[cid],time=t)
                    # Remove absorbed predecessor nodes; historical overlap must
                    # not manufacture repeated merges in following frames.
                    for old_nid,old_fid,_,_,_ in adjacent[cid]:
                        if old_nid not in selected_nodes:nodes.pop(old_nid,None)
                nodes.update(updated)
                ended={fid for fid,f in families.items() if f['ended']}
                nodes={nid:n for nid,n in nodes.items() if n['family'] not in ended}
                for fid in ended:del families[fid]
                # Expire old nodes of a family even when another fragment lives.
                # Weak/gap contacts to a vanished body are recorded as uncertainty
                # only for its terminal fate, not as a global birth veto.
                for cid,es in gaps.items():
                    if adjacent[cid]:
                        for _,fid,_,_,_ in es:
                            if fid in families and fid not in current:families[fid]['ambiguous_loss']=True
                statistics.append(dict(frame=frame,time_ps=t,tracked_atoms=int(np.count_nonzero(labels)),
                    largest_cluster=max(map(len,groups.values()),default=0),active_families=len(families),events=len(events)))
                if fi%32==0 or fi==len(frames)-1:
                    write_json(root/'progress.json',dict(frames=fi+1,total_frames=len(frames),events=len(events),
                        seconds=time.monotonic()-started))
            last=s['times_ps'][frames[-1]]
            for fid,f in list(families.items()):
                reset=f['last_seen']+dt+c['reset_ps']
                status=('identity_unresolved' if f['ambiguous_loss'] else 'dissolved') if reset<=last+1e-8 else (
                    'present_at_end' if abs(last-f['last_seen'])<1e-8 else 'right_censored')
                finish(fid,status,reset if status in ('dissolved','identity_unresolved') else last)
        finally:trace.close()
        write_json(root/'events.json',events);write_json(root/'interactions.json',interactions)
        write_json(root/'trajectory.json',statistics)
        counts=dict(events=len(events),birth_records=sum(e['is_birth_record'] for e in events),
            liquid_emergences=sum(e['origin']=='emergence_from_liquid' for e in events),
            established_liquid_emergences=sum(e['origin']=='emergence_from_liquid' and e['establishment_confirmation_ps'] is not None for e in events),
            dissolved=sum(e['terminal_status']=='dissolved' for e in events))
        names=('events.json','interactions.json','trajectory.json','sizes.jsonl.gz')
        done=dict(identity=identity,**task,counts=counts,seconds=time.monotonic()-started,
            files={n:sha(root/n) for n in names})
        write_json(receipt,done);arrays.cache_clear();return done
