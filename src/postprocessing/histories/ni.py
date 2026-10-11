"""Find four Ni births in saved prefixes and visualize their continuous histories."""
from src.postprocessing.histories.common import precursor_links, local_environment, shape
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import gzip
import html
import json
import multiprocessing
import os
import time

import numpy as np

from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.artifacts import file_hash as sha,json_digest as digest,write_json
from src.experiment_runner.execution import ExecutionBundle,SlurmQueue,recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs,write_metric_rows
from src.project_runtime.paths import REPO,resolve_path
from src.postprocessing.protocols.ni import track_config
from src.postprocessing import lineage as lineage_tracker
from src.postprocessing.histories.common import root, slug, clusters, node
from src.postprocessing.visualization.births import render
from src.postprocessing.io import geometry
from src.postprocessing.extraction import Frames, extract as extract_chunk
from src.postprocessing.io import read
from src.postprocessing.sources import part

FAMILY='ni_birth_histories'
MODULE='src.postprocessing.histories.ni'


def prepare(c):
    sources=[];provenance=[]
    for request in c['sources']:
        recipe=read(resolve_path(request['config']));campaign=read(resolve_path(recipe['launch'])/'plan.json')
        record=next(r for r in campaign['records'] if r['id']==request['source'])
        folder=resolve_path(recipe['archive'])/record['id'];done=read(folder/'complete.json')
        binary=ShootingBinaryTrajectory.load(folder/'trajectory_binary_float16')
        if sha(binary.root/'manifest.json')!=done['trajectory_manifest_sha256']:raise ValueError('Changed completed Ni source')
        if not np.array_equal(binary.timesteps,np.arange(binary.frame_count)*100):raise ValueError('Non-native 0.1 ps Ni timeline')
        monitor=list(csv.DictReader((folder/'structure.csv').open()))
        crossing=next(float(r['time_ps']) for r in monitor if int(r['largest_cluster_atoms'])>=128)
        stop=min(binary.frame_count,round((crossing+c['after_monitor_128_ps'])/.1)+1)
        data_part=part(binary.root,1.)
        s=dict(id=record['id'],dataset=recipe['campaign_id'],parts=[data_part],frames=[[0,i] for i in range(stop)],
            times_ps=(binary.timesteps[:stop]*.001).tolist(),sampled_frames=list(range(stop)),views={'010':list(range(stop))},
            frame_count=stop,atom_count=binary.atom_count,material='Ni',potential=['ni-mendelev2012-eam'],
            ancestry_group=record['id'],role=record['role'],origin='independent_melt_cooling_and_npt_hold',
            native_cadence_ps=.1,temperature_K=record['temperature_K'])
        s['input_identity']=digest(dict(arrays=data_part['array_sha256'],frames=s['frames']));sources.append(s)
        provenance.append(dict(source=record['id'],source_record=record,campaign_plan=str(resolve_path(recipe['launch'])/'plan.json'),
            campaign_identity=campaign['identity'],completion_sha256=sha(folder/'complete.json'),source_total_frames=binary.frame_count,
            monitor_sha256=sha(folder/'structure.csv'),coarse_128_crossing_ps=crossing,searched_through_ps=s['times_ps'][-1],
            potential_sha256=recipe['potential_sha256'],cooling_ps=recipe['cooling_ps']))
    parent=dict(config=dict(cache=c['cache'],cadences={'010':.1},detectors=['ptm'],ptm_rmsd_cutoff=c['ptm_rmsd_cutoff'],
        neighbor_cutoff_A=c['neighbor_cutoff_A'],minimum_tracked_atoms=4,chunk_frames=16),sources=sources)
    parent['identity']=digest(parent)
    tc=track_config(dict(analysis=str(root(c)),interface_distance_A=7.))
    tracking=dict(config=tc,parent_identity=parent['identity']);tracking['identity']=digest(tracking)
    plan=dict(config=c,parent=parent,tracking=tracking,provenance=provenance,
        implementation=check_metric_docs(family=FAMILY)[FAMILY]['files'],
        selection='Earliest PTM root birth with own connected 64-atom/1.5-ps establishment, one per specified completed source; descriptive selection, not a rate cohort.')
    plan['identity']=digest(plan);path=root(c)/'technical/search-plan.json'
    if path.exists() and read(path)!=plan:raise ValueError('Ni example inputs changed; use another revision')
    write_json(path,plan);return plan


def extract(plan,s):
    parent=plan['parent'];chunk=parent['config']['chunk_frames']
    for start in range(0,s['frame_count'],chunk):
        stop=min(s['frame_count'],start+chunk)
        extract_chunk(parent,dict(source=s['id'],start=start,stop=stop))
        write_json(root(plan['config'])/'technical'/f"{s['id']}-progress.json",
                   dict(state='classifying',frames=stop,total=s['frame_count'],updated_unix=time.time()))


def precursor_graph(s,first,start,frame_nodes,founders):
    anchor,nodes,edges=precursor_links(first,start,frame_nodes,founders,2)
    ordered=sorted(nodes.values(),key=lambda n:(n['time_ps'],n['id']));strong=[n for n in ordered if n['path_status'] in ('direct','gap')]
    return dict(root_node=anchor['id'],nodes=ordered,edges=edges,earliest_direct_ps=min(n['time_ps'] for n in ordered if n['path_status']=='direct'),
        earliest_with_gaps_ps=min(n['time_ps'] for n in strong),lookback_start_ps=s['times_ps'][start],
        tail_left_censored=any(n['frame']==start for n in strong),known_established_connections=[],
        interpretation='Retrospective shared-atom links. Gap means an absent strong identity link, not an absent source frame. Weak links do not establish ancestry.')


def movie(plan,s,e):
    c=plan['config'];folder=root(c)/'analyses/births'/e['slug'];folder.mkdir(parents=True,exist_ok=True)
    tracker=root(c)/'technical/sources'/s['id']/'010/ptm';history=[]
    with gzip.open(tracker/'sizes.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row['family']==e['track']:history.append(row)
    trace={r['frame']:r for r in history};first=e['first_frame'];start=max(0,first-round(c['lookback_ps']/.1))
    stop=round((e['establishment_confirmation_ps']+c['establishment_after_ps'])/.1)
    if stop>=s['frame_count']:raise ValueError('Search prefix does not cover requested post-establishment movie')
    binary=ShootingBinaryTrajectory.load(s['parts'][0]['path']);ids=binary.atom_ids;founders=np.searchsorted(ids,e['first_members'])
    if not np.array_equal(ids[founders],e['first_members']):raise ValueError('Ni founder atom IDs changed')
    q,box=geometry(s,first);delta=q[founders]-e['centroid_A'];delta-=box*np.rint(delta/box);anchor=int(founders[np.argmin(np.linalg.norm(delta,axis=1))])
    access=Frames(plan['parent'],s,'ptm');frames=[];frame_nodes={};packed=defaultdict(list);offsets=[0]
    for frame in range(start,stop+1):
        q,box=geometry(s,frame);ptm=access.raw_ptm(frame);labels=access.labels(frame)
        if frame<=first:frame_nodes[frame]=[node(s['id'],frame,s['times_ps'][frame],m,ids) for m in clusters(q,box,ptm,c['neighbor_cutoff_A'])]
        d,local,patch,(q6,qbar6,coherence6)=local_environment(q,box,anchor,c['radius_A'])
        track=trace.get(frame);member=np.zeros(len(local),bool) if track is None else labels[local]==track['largest_component']
        for key,value in dict(positions=d[local].astype(np.float32),atom_ids=ids[local],ptm=ptm[local],catalogued_family=member,founder=np.isin(ids[local],e['first_members'])).items():packed[key].append(value)
        offsets.append(offsets[-1]+len(local));frames.append(dict(frame=frame,time_ps=s['times_ps'][frame],relative_time_ps=s['times_ps'][frame]-e['first_time_ps'],
            largest_family=0 if track is None else track['size'],q6=q6,qbar6=qbar6,coherence6=coherence6,
            first_shell_A=float(np.linalg.norm(patch[1:13],axis=1).mean()),crystal_fraction=float(np.isin(ptm[local],[1,2,3]).mean()),box=box.tolist()))
    graph=precursor_graph(s,first,start,frame_nodes,e['first_members']);graph['birth_event_id']=e['event_id']
    for i,f in enumerate(frames):
        ancestors={atom for n in graph['nodes'] if n['frame']==f['frame'] and n['path_status'] in ('direct','gap') for atom in n['atom_ids']}
        mask=np.isin(packed['atom_ids'][i],list(ancestors));packed['precursor'].append(mask)
        cloud=packed['positions'][i][mask|packed['catalogued_family'][i]]
        f.update(shape(cloud))
    write_json(folder/'birth.json',dict(e,anchor_atom_id=int(ids[anchor]),searched_prefix_end_ps=s['times_ps'][-1]))
    write_json(folder/'graph.json',graph);write_json(folder/'family-history.json',history);write_json(folder/'movie-frames.json',frames)
    np.savez_compressed(folder/'movie.npz',offsets=np.asarray(offsets,np.int64),**{k:np.concatenate(v) for k,v in packed.items()})
    summary=dict(birth_event_id=e['event_id'],source=s['id'],ancestor=s['ancestry_group'],role=s['role'],material='Ni',temperature_K=s['temperature_K'],
        onset_phase='cooling' if e['first_time_ps']<20 else 'isothermal',first_8_ps=e['first_time_ps'],establishment_start_ps=e['establishment_start_ps'],
        establishment_confirmation_ps=e['establishment_confirmation_ps'],searched_prefix_end_ps=s['times_ps'][-1],movie_start_ps=frames[0]['time_ps'],
        movie_end_ps=frames[-1]['time_ps'],movie_frames=len(frames),earliest_direct_ps=graph['earliest_direct_ps'],
        earliest_with_gaps_ps=graph['earliest_with_gaps_ps'],gap_edges=sum(x['kind']=='gap' for x in graph['edges']),tail_left_censored=graph['tail_left_censored'])
    write_json(folder/'summary.json',summary)
    files={name:sha(folder/name) for name in ('birth.json','graph.json','family-history.json','movie-frames.json','movie.npz','summary.json')}
    write_json(folder/'complete.json',dict(identity=digest([plan['identity'],e['event_id']]),files=files))
    render(c,e['slug'])
    page=(folder/'viewer.html').read_text().replace('Birth catalogue','Ni examples').replace('Catalogue embryo onset','Tracked embryo onset').replace('catalogue embryo onset','tracked embryo onset').replace('the catalogue onset','the tracked onset')
    (folder/'viewer.html').write_text(page)
    return e


def worker(args):
    plan,index=args;s=plan['parent']['sources'][index];c=plan['config'];done=root(c)/'technical'/f"{s['id']}-example.json"
    if done.exists():return read(done)
    with recorded_stage(root(c)/'technical'/f"{s['id']}-worker.json",source=s['id']):
        extract(plan,s);lineage_tracker.run(plan['tracking'],plan['parent'],dict(source=s['id'],cadence='010',detector='ptm'))
        path=root(c)/'technical/sources'/s['id']/'010/ptm/events.json';events=read(path)
        candidates=[e for e in events if e['record_kind']=='root' and e['origin']=='emergence_from_liquid' and e['establishment_confirmation_ps'] is not None]
        if not candidates:raise ValueError('No confirmed liquid-origin root birth in searched Ni prefix: '+s['id'])
        e=min(candidates,key=lambda e:(e['establishment_confirmation_ps'],e['first_time_ps'],e['track']))
        event_id=f"{s['id']}:ptm:{e['track']}:0";e=dict(e,event_id=event_id,slug=slug(event_id),source_id=s['id'],source_event_file=str(path),source_event_sha256=sha(path))
        result=movie(plan,s,e);write_json(done,result);print('Ni example complete',s['id'],flush=True);return result


def gallery(c,events,plan):
    r=root(c);rows=[];cards=[]
    for e in events:
        d=r/'analyses/births'/e['slug'];rows.append(read(d/'summary.json'));href='analyses/births/'+e['slug']
        cards.append(f'<article><h2>{html.escape(e["source"])}</h2><p>Embryo {e["first_time_ps"]:.1f} ps → establishment {e["establishment_confirmation_ps"]:.1f} ps · {html.escape(e["role"])}</p><a href="{href}/viewer.html">Interactive atoms and six property plots</a><p><a href="{href}/plots/early-appearance.gif">Early appearance GIF</a> · <a href="{href}/plots/establishment.gif">Establishment GIF</a> · <a href="{href}/plots/precursor-graph.png">Precursor graph</a></p></article>')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Four Ni births</title><style>body{font:16px system-ui;background:#101923;color:#dce8f3;max-width:1050px;margin:40px auto;padding:0 20px;line-height:1.6}a{color:#7dd7ff}article{padding:15px 0;border-bottom:1px solid #496278}</style><h1>Four Ni births</h1><p>One confirmed birth in each of four independent completed 1100 K melts. This is an illustrative subset, not a Ni catalogue or a nucleation-rate estimate. Exact 0.1 ps saved frames; continuous atom movies from precursor history through establishment. Earlier clusters remain attached history nodes.</p><p>Birth confirmation: a connected descendant of at least 64 PTM-crystalline atoms for 1.5 ps. Ni neighbor cutoff 3.2 Å; PTM RMSD cutoff 0.1. A gap in the precursor graph means an uncertain identity link, not missing coordinates. Full family traces end at each explicitly bounded search prefix. Times are measured from the saved liquid start; the cooling ramp ends at 20 ps.</p>'''
    (r/'index.html').write_text(page+''.join(cards));write_metric_rows(rows,r,family=FAMILY,name='four-births')
    write_json(r/'technical/plan.json',dict(config=c,events=events,examples=[e['slug'] for e in events],search_identity=plan['identity']))
    write_json(r/'technical/complete.json',dict(state='complete',births=len(events),sources=[e['source'] for e in events],search_identity=plan['identity']))


def run(c):
    plan=read(root(c)/'technical/search-plan.json');results=[]
    with ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,(plan,i)) for i in range(len(plan['parent']['sources']))]
        for future in as_completed(futures):
            results.append(future.result());write_json(root(c)/'technical/progress.json',dict(completed=[e['source'] for e in results],total=4))
    gallery(c,sorted(results,key=lambda e:e['source']),plan)


def launch(c):
    tech=root(c)/'technical';receipt=tech/'launch.json'
    if receipt.exists():raise FileExistsError(receipt)
    bundle=ExecutionBundle.freeze(REPO,tech/'code',c,directories=('src','configs/postprocessing','docs/metrics'),files=((REPO/'machine.local.yaml','machine.local.yaml'),(REPO/'configs/datasets.json','configs/datasets.json')))
    env={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OVITO_THREAD_COUNT')}
    q=SlurmQueue(tech,bundle,MODULE,env,receipt,dict(owner='current-chat four Ni visualization examples',jobs={}), 'ni-four-vis')
    with q.submission():q.submit('run',['--cpus-per-task='+str(c['workers']),'--mem=16G','--time='+c['time_limit']])
    print(q.receipt,flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('prepare','launch','run'));parser.add_argument('--config',required=True)
    args=parser.parse_args();c=read(args.config);check_metric_docs(family=FAMILY)
    if args.stage=='prepare':prepare(c)
    elif args.stage=='launch':launch(c)
    else:run(c)


if __name__=='__main__':main()
