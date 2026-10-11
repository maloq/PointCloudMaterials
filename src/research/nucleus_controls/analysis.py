"""Finite-horizon nucleus establishment and tracked-seed fate measurements."""
from collections import defaultdict
import fcntl
from pathlib import Path
import time

import numpy as np

from src.data.fixed_cohort.protocol import write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.postprocessing.protocols import ancestry
from src.postprocessing.protocols import fates
from src.postprocessing.io import frame_geometry
from src.research.nucleus_recovery import Reference, read, roots, graph_prefix, add_frame
from .data import FAMILY


def measure(c, case, task, q, boxes, labels):
    ref=Reference(case);state=case['state'];start=state['start_frame']
    graph,previous=graph_prefix(ref,start)
    dense=[];largest=[];fractions=[];divergence=[]
    for j in range(len(q)):
        xyz=np.mod(np.asarray(q[j],float),boxes[j])
        nodes=add_frame(graph,previous,xyz,boxes[j],labels[j],start+j,c['lineage'])
        dense.append(nodes)
        present=np.unique(nodes[nodes>0]);largest.append(max((graph['size'][n] for n in present),default=0))
        fractions.append(float(np.isin(labels[j],[1,2,3]).mean()))
        old,old_box=frame_geometry(ref.raw,start+j)
        delta=xyz/boxes[j]-old/old_box;delta-=np.rint(delta);delta*=old_box
        divergence.append(dict(time_ps=j*case['cadence_ps'],all_atoms_rms_A=float(np.sqrt(np.mean(np.sum(delta**2,axis=1))))))
    graph={k:np.asarray(v,np.int64 if k=='edges' else np.int32) for k,v in graph.items()}
    graph['edges']=graph['edges'].reshape(-1,5);graph['center_node']=np.empty((0,len(graph['start'])),np.int32)
    events,lineage_roots,_=ancestry.establish(graph,case['threshold'])
    class Geometry:
        rows=np.empty(0,dtype=int)
        def frame(self, frame):
            j=frame-start
            return ref.frame(frame) if j<0 else (np.mod(np.asarray(q[j],float),boxes[j]),boxes[j],dense[j])
        def labels(self, frame):
            return ref.labels(frame) if frame<start else labels[frame-start]
        def members(self, nodes):
            by_frame=defaultdict(list)
            for n in nodes:by_frame[int(graph['frame'][n])].append(n)
            return np.unique(np.concatenate([np.flatnonzero(np.isin(self.frame(f)[2],ns)) for f,ns in by_frame.items()]))
    geometry=Geometry()
    ancestry.locate_births(c,dict(id=case['key'],center_atom_ids=[],legacy_center_atom_ids=[]),graph,events,lineage_roots,geometry)
    parents,children=fates.graph_links(graph)
    seed_node=None;descendants=set();seed_fate=None
    if case['core_rows']:
        core=np.asarray(case['core_rows']);nodes,counts=np.unique(dense[0][core],return_counts=True)
        eligible=[(int(count),int(n)) for n,count in zip(nodes,counts) if n>0 and count>=c['lineage']['minimum_shared_atoms']
                  and count>=c['lineage']['minimum_overlap_fraction']*min(len(core),int(graph['size'][n]))]
        if eligible:
            seed_node=max(eligible)[1];pending=[seed_node]
            while pending:
                n=pending.pop()
                if n in descendants:continue
                descendants.add(n);pending.extend(children[n])
            seed_fate=fates.verify_disappearance(c,graph,np.array(sorted(descendants)),children,geometry,len(graph['start']))
        else:
            seed_fate=dict(terminal_status='initial_seed_unresolved',dissolution_frame=None)
    candidates=[]
    for e in events:
        if e['confirmation_frame']<start:continue
        xyz,box,_=geometry.frame(e['birth_frame'])
        delta=np.asarray(e['centroid_A'])-xyz[case['anchor_row']]
        delta-=box*np.rint(delta/box)
        distance=float(np.linalg.norm(delta))
        candidates.append(dict(**e,new_isolated=e['birth_frame']>start and e['kind']=='isolated_establishment',
            near_anchor=distance<=c['site_radius_A'],anchor_distance_A=distance,
            from_initial_seed=e['birth_node'] in descendants,
            birth_time_ps=(e['birth_frame']-start)*case['cadence_ps'],
            confirmation_time_ps=(e['confirmation_frame']-start)*case['cadence_ps']))
    result=dict(task=task['name'],case=case['key'],block=case['block'],cohort=case['cohort'],source=case['source'],
        ancestry=case['ancestry'],role=case['role'],source_temperature_K=case['temperature_K'],
        initial_cluster_atoms=case['initial_cluster_atoms'],initial_crystal_fraction=fractions[0],
        initial_largest_cluster_atoms=int(largest[0]),arm=task['arms'][0],integrator=task['integrator'],
        replicate=task['replicate'],seed=task['seed'],
        seed_terminal_status=seed_fate['terminal_status'] if seed_fate else None,
        seed_dissolved=int(seed_fate['terminal_status']=='dissolved') if seed_fate and seed_node is not None else None,
        seed_dissolution_ps=(seed_fate['dissolution_frame']-start)*case['cadence_ps'] if seed_fate and seed_fate['dissolution_frame'] is not None else None)
    for horizon in c['evaluation_horizons_ps']:
        suffix=f'{horizon:g}ps';j=round(horizon/case['cadence_ps'])
        qualifying=[e for e in candidates if e['new_isolated'] and e['confirmation_time_ps']<=horizon+1e-8]
        result.update({f'any_new_nucleus_{suffix}':int(bool(qualifying)),
            f'local_new_nucleus_{suffix}':int(any(e['near_anchor'] for e in qualifying)),
            f'seed_established_{suffix}':int(any(e['from_initial_seed'] for e in qualifying)) if case['core_rows'] else None,
            f'crystal_fraction_{suffix}':fractions[j],
            f'largest_cluster_atoms_{suffix}':int(largest[j]),
            f'reached_10pct_crystal_{suffix}':int(max(fractions[:j+1])>=.1)})
    result['maximum_crystal_fraction']=max(fractions)
    result['maximum_cluster_atoms']=int(max(largest))
    result['first_new_nucleus_ps']=min((e['birth_time_ps'] for e in candidates if e['new_isolated']),default=None)
    result['tracked_seed_peak_atoms']=int(graph['size'][list(descendants)].max()) if descendants else None
    return result,graph,candidates,divergence


def reference(c, case):
    ref=Reference(case);state=case['state'];frames=range(state['start_frame'],state['stop_frame']+1)
    geometry=[frame_geometry(ref.raw,f) for f in frames]
    q=np.stack([p for p,b in geometry]);boxes=np.stack([b for p,b in geometry])
    labels=np.stack([ref.labels(f) for f in frames])
    task=dict(name=case['key']+'-saved-reference',arms=['saved_reference'],integrator='saved',replicate=0,seed=None)
    result,_,_,_=measure(c,case,task,q,boxes,labels)
    if case['cohort']=='establishing' and result['seed_established_12ps']!=1:
        raise ValueError('Selected positive seed does not reproduce its saved establishment: '+case['key'])
    if case['cohort']=='dissolved' and result['seed_dissolved']!=1:
        raise ValueError('Selected negative seed does not reproduce its saved dissolution: '+case['key'])
    return result


def analyze(c,plan,task,case,root):
    q=np.load(root/'positions.building.npy',mmap_mode='r');boxes=np.load(root/'boxes.npy');labels=np.load(root/'labels.npy',mmap_mode='r')
    result,graph,events,divergence=measure(c,case,task,q,boxes,labels)
    result['plan_identity']=plan['identity']
    np.savez_compressed(root/'graph.npz',**graph)
    write_json(root/'events.json',events);write_json(root/'divergence.json',divergence)
    write_json(root/'result.json',result)
    return result


def summarize(c):
    out,_,archive=roots(c);plan=read(out/'technical/plan.json')
    with (out/'technical/summary.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        records=[read(archive/'branches'/t['name']/'result.json') for t in plan['tasks']
                 if (archive/'branches'/t['name']/'complete.json').exists()]
        groups=defaultdict(list)
        for r in records:groups[r['cohort'],r['arm']].append(r)
        for case in plan['cases']:
            r=read(out/'technical/reference'/f'{case["key"]}.json');groups[r['cohort'],r['arm']].append(r)
        summaries=[];means={};block_ids=sorted(b['block'] for b in plan['blocks'])
        metrics=[f'{m}_{h:g}ps' for h in c['evaluation_horizons_ps'] for m in
                 ['any_new_nucleus','local_new_nucleus','seed_established','crystal_fraction','largest_cluster_atoms','reached_10pct_crystal']]+['seed_dissolved']
        for (cohort,arm),rs in sorted(groups.items()):
            by_case=defaultdict(list)
            for r in rs:by_case[r['case']].append(r)
            required=c['noise_replicates'] if arm=='response450K' else 1
            ready=[v for v in by_case.values() if len(v)==required]
            row=dict(cohort=cohort,arm=arm,completed_continuations=len(rs),completed_cases=len(ready),
                planned_cases=len(plan['blocks']),independent_ancestors=len({v[0]['ancestry'] for v in ready}))
            for metric in metrics:
                values=[np.mean([r[metric] for r in v]) for v in ready if v[0][metric] is not None]
                row[metric]=float(np.mean(values)) if values else None
                row[metric+'_ci_low']=row[metric+'_ci_high']=None
                if len(values)>1:
                    rng=np.random.default_rng(c['seed']);a=np.asarray(values)
                    draws=a[rng.integers(len(a),size=(c['bootstrap_draws'],len(a)))].mean(1)
                    row[metric+'_ci_low'],row[metric+'_ci_high']=map(float,np.quantile(draws,[.025,.975]))
                for v in ready:
                    means[v[0]['block'],cohort,arm,metric]=float(np.mean([r[metric] for r in v])) if v[0][metric] is not None else None
            summaries.append(row)
        differences=[]
        for arm in ['sourceNPT','response450K','saved_reference']:
            for left,right in [('establishing','dissolved'),('dissolved','random_liquid'),('establishing','random_liquid')]:
                for metric in metrics:
                    a=[means[b,left,arm,metric]-means[b,right,arm,metric] for b in block_ids
                       if means.get((b,left,arm,metric)) is not None and means.get((b,right,arm,metric)) is not None]
                    if not a:continue
                    row=dict(arm=arm,comparison=left+'-minus-'+right,metric=metric,matched_blocks=len(a),difference=float(np.mean(a)),ci_low=None,ci_high=None)
                    if len(a)>1:
                        rng=np.random.default_rng(c['seed']);a=np.asarray(a)
                        values=a[rng.integers(len(a),size=(c['bootstrap_draws'],len(a)))].mean(1)
                        row['ci_low'],row['ci_high']=map(float,np.quantile(values,[.025,.975]))
                    differences.append(row)
        if records:write_metric_rows(records,out,family=FAMILY,name='branches')
        write_metric_rows(summaries,out,family=FAMILY,name='cohorts')
        if differences:write_metric_rows(differences,out,family=FAMILY,name='matched-differences')
        reference_rows=[read(out/'technical/reference'/f'{case["key"]}.json') for case in plan['cases']]
        write_metric_rows(reference_rows,out,family=FAMILY,name='saved-reference')
        progress=dict(completed=len(records),planned=len(plan['tasks']),cases=len(plan['cases']),scope=plan['scope'],updated_unix=time.time())
        write_json(out/'technical/progress.json',progress)
        render(out,summaries)
        (out/'README.md').write_text('# Dissolved-embryo and random-liquid continuations\n\n'
            f'Completed {len(records)}/{len(plan["tasks"])} new continuations, 12 ps each. '
            f'{len(plan["blocks"])} temperature/size matched blocks; {plan["scope"]["independent_ancestors"]} original ancestors.\n\n'
            'The small dissolved embryos were not previously established nuclei. Random liquid frames are sampled from present structure alone. '
            'Primary outcomes are new isolated nuclei anywhere and within 8 Å of the tracked anchor. '
            'Only cases with all planned repetitions enter cohort rates; partial coverage is provisional.\n\n'
            '[Cohort rates](tables/cohorts.csv) · [Paired differences](tables/matched-differences.csv) · '
            '[Saved reference](tables/saved-reference.csv) · [Definitions](tables/METRICS.md)\n\n'
            'The common wall-clock deadline is recorded in technical/budget.json. No model is trained. '
            'Other Al cadences, shooting controls, million-atom data, relaxed data and established nuclei that later melted remain outside this campaign.\n')
    return progress


def render(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cohorts=['establishing','dissolved','random_liquid'];arms=['saved_reference','sourceNPT','response450K']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,title in zip(axes,['any_new_nucleus_12ps','local_new_nucleus_12ps'],['New isolated nucleus anywhere','New isolated nucleus near anchor']):
        for i,arm in enumerate(arms):
            values=[next((r[metric] for r in rows if r['cohort']==cohort and r['arm']==arm),None) for cohort in cohorts]
            x=[j for j,v in enumerate(values) if v is not None]
            ax.bar(np.array(x)+(i-1)*.24,[values[j] for j in x],width=.24,label=arm)
        ax.set_xticks(range(3),['Establishing','Dissolved embryo','Random liquid']);ax.set_ylim(0,1.08)
        ax.set_title(title+' — 12 ps');ax.set_ylabel('Fraction of continuations');ax.legend(fontsize=8)
    fig.tight_layout();(out/'plots').mkdir(exist_ok=True)
    fig.savefig(out/'plots/cohorts.png',dpi=160);fig.savefig(out/'plots/cohorts.svg');plt.close(fig)
