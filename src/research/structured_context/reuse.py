"""Archived relaxed observations; matched sparse causal inputs, fixed MD targets.

No simulation entry point is called by this workflow.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
import torch

from src.project_runtime.paths import resolve_path,dataset_path
from src.data.structural_pretraining.prepare import file_hash,save_json,digest
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data.trajectories.lammps import TemporalLAMMPSBinaryTrajectory
from src.data.relaxed_targets.worker import lock
from src.data.conversion.relaxation import read_relaxed
from src.research.relaxed_encoder.prepare import paired_clouds
from src.training_methods.shared_pretraining.queue import deadline_for_job,snapshot
from .relaxed import load_encoder,encode,observed_information
from .geometry import stencil,representatives


def histories(frames,anchors,count,max_span):
    """Only real, strictly ordered observations; never pad or interpolate."""
    frames=sorted(set(frames));result={}
    for anchor in anchors:
        if anchor not in frames:continue
        past=[f for f in frames if anchor-max_span<=f<=anchor]
        if len(past)>=count:result[anchor]=past[-count:]
    return result


def read_record(path,identity):
    record=json.loads(path.read_text())
    if record['identity']!=identity:raise ValueError(f'Cell release changed: {path}')
    # Both existing producers expose the actual archive, including reused cells.
    return dict(path=str(path),sha256=file_hash(path),archive=record['archive'],
        source=record['task']['source'],frame=record['task']['frame'])


def freeze(config):
    root=resolve_path(config['output'])/'technical';root.mkdir(parents=True,exist_ok=True)
    path=root/'plan.json'
    if path.exists():
        plan=json.loads(path.read_text())
        if plan['reuse_config']!=config:raise ValueError('Reuse configuration changed')
        return plan
    reference=resolve_path(config['reference_plan']);plan=json.loads(reference.read_text())
    training=json.loads(resolve_path(config['relaxation_plan']).read_text())
    selection=json.loads(resolve_path(config['selection']).read_text())
    checkpoint=resolve_path(training['config']['output'])/'technical/runs'/selection['name']/'best.pt'
    selected=next(r for r in training['config']['runs'] if r['name']==selection['name'])
    if selected['arm']!='relaxed_to_relaxed':raise ValueError('Encoder requires relaxed input training')
    manifest=json.loads(resolve_path(training['config']['normalization_manifest']).read_text())
    exposed=set(manifest['train_roots']+manifest['selection_roots'])|{s['lineage'] for s in training['sources'] if s['pilot_fit']}
    protected={s['lineage'] for s in plan['sources'] if s.get('validation_role',s['split']) in ('test','calibration')}
    if exposed&protected:raise ValueError(f'Protected encoder ancestry overlap: {exposed&protected}')
    if plan['scale']!=training['config']['scale']:raise ValueError('Material scale differs')
    cells={};releases=[]
    source_ids={s['id'] for s in plan['sources']}
    for release in config['reuse_plans']:
        p=resolve_path(release);release_plan=json.loads(p.read_text());cache=resolve_path(release_plan['config']['cache'])/'cells'
        candidates=[Path(e.path)/'complete.json' for e in os.scandir(cache) if e.is_dir()]
        candidates=[p for p in candidates if p.exists()]
        with ThreadPoolExecutor(max_workers=8) as pool:
            records=list(pool.map(lambda p:read_record(p,release_plan['identity']),candidates))
        for record in records:
            if record['source'] in source_ids and record['frame']%4==0 and record['frame']<=664:
                cells.setdefault((record['source'],record['frame']),record)
        releases.append(dict(plan=str(p),sha256=file_hash(p),complete_cells_at_freeze=len(records)))
    timeline={};chosen=[];used={}
    for source in plan['sources']:
        sid=source['id'];frames=[f for s,f in cells if s==sid]
        h=histories(frames,plan['anchors'],config['history_frames'],round(config['maximum_history_ps']/.75))
        if not h:raise ValueError(f'No causal archived histories for source {sid}')
        timeline[str(sid)]={str(a):f for a,f in h.items()}
        for frame in sorted({f for hist in h.values() for f in hist}):used[f'{sid}-{frame}']=cells[(sid,frame)]
        chosen.append(source)
    plan['reuse_config']=config;plan['relaxed_config']=config
    plan['relaxed_encoder']=dict(checkpoint=str(checkpoint),sha256=file_hash(checkpoint),selection=selection,protected_overlap=[])
    plan['config']=dict(plan['config'],output=config['output'])
    plan['structured_config']=dict(plan['structured_config'],output=config['output'],context_cache=config['context_cache'])
    plan['archive_cells']=used;plan['observed_histories']=timeline;plan['releases']=releases
    plan['structured_identity']=digest(dict(config=config,cells=used,histories=timeline,encoder=plan['relaxed_encoder']))
    plan['identity']=digest(plan);save_json(path,plan)
    tasks=[]
    for domain in ('relaxed','observed'):
        for old in json.loads((reference.parent/'queue.json').read_text()):
            if old['encoder']!='mace':continue
            spec=copy.deepcopy(old)
            spec.update(name=f'{domain}-mace-{old["method"]}-reuse-E36',observation_domain=domain,
                target_encoder='reference_mace',history_ps=config['maximum_history_ps'],observed_frames=config['history_frames'])
            tasks.append(spec)
    save_json(root/'queue.json',tasks)
    save_json(root/'coverage.json',dict(sources=len(chosen),cells=len(used),source_origins=sum(map(len,timeline.values())),
        original_required_cells=29850,new_relaxations=0,history_frames=config['history_frames'],maximum_history_ps=config['maximum_history_ps']))
    return plan


def archived_positions(plan,record,source):
    path=Path(record['path'])
    if file_hash(path)!=record['sha256']:raise ValueError(f'Reused receipt changed: {path}')
    archive=Path(record['archive']);meta=json.loads((archive/'metadata.json').read_text())
    if meta['source_manifest_sha256']!=source['manifest_sha256'] or meta['source_frame']!=record['frame']:
        raise ValueError(f'Archive source mismatch: {archive}')
    expected=json.loads(resolve_path(plan['reuse_config']['relaxation_plan']).read_text())['config']['potential_sha256']
    if (meta['fmax_eV_per_A']>.01 or meta['settings']['minimizer']!='fire'
            or [meta['potential_checksums'][p] for p in meta['settings']['potential_files']]!=expected):
        raise ValueError(f'Archive relaxation protocol differs: {archive}')
    binary=TemporalLAMMPSBinaryTrajectory.load(archive/'relaxed_binary_float16')
    binary.verify_checksums()
    conversion=json.loads((archive/'conversion.json').read_text())
    if conversion['source_sha256']!=binary.manifest['source']['sha256']:
        raise ValueError('Conversion lineage mismatch')
    if int(binary.timesteps[0])!=meta['source_timestep']:raise ValueError('Archive timestep differs')
    box=binary.box_high[0].astype(float)-binary.box_low[0].astype(float)
    cold=np.mod(binary.positions[0].astype(float),box)
    return cold,box,binary.atom_ids,dict(metadata_sha256=file_hash(archive/'metadata.json'),
        binary_manifest_sha256=file_hash(binary.root/'manifest.json'),conversion_sha256=file_hash(archive/'conversion.json'),
        quantization=conversion['quantization'])


def cold_graphs(plan,source,frame,cold,box,raw):
    np.testing.assert_allclose(raw.box_high[frame].astype(float)-raw.box_low[frame].astype(float),box,atol=1e-6,rtol=0)
    centers=np.searchsorted(raw.atom_ids,source['center_atom_ids']);np.testing.assert_array_equal(raw.atom_ids[centers],source['center_atom_ids'])
    # Match actual atom identities to the original observed symmetric context.
    # Quenching and storage quantization change offsets, never slot membership.
    hot=np.mod(raw.positions[frame].astype(float),box)
    tree=cKDTree(hot,boxsize=box);queries=stencil(plan['structured_config']['shell_radii_A'])
    assignments=[representatives(hot,c,tree,box,queries,plan['structured_config']['max_query_offset_A']) for c in centers]
    ids=np.stack([a for a,r in assignments])
    relative=cold[ids]-cold[centers,None];relative-=box*np.rint(relative/box);relative=relative.astype(np.float32)
    unique,mapping=np.unique(ids,return_inverse=True)
    _,clouds,neighbors=paired_clouds(raw.positions[frame].astype(float),cold,box,unique)
    return dict(clouds=clouds,mapping=mapping.reshape(ids.shape),relative=relative,query_atom_ids=raw.atom_ids[ids],
        neighbor_atom_ids=raw.atom_ids[neighbors])


def extract_source(plan,source,model,deadline):
    cache=resolve_path(plan['reuse_config']['context_cache']);root=cache/str(source['id']);root.mkdir(parents=True,exist_ok=True)
    receipt=root/'complete.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['identity']!=plan['structured_identity'] or file_hash(root/'observations.npz')!=saved['sha256']:raise ValueError('Reused source features changed')
        return True
    raw=ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
    if file_hash(raw.root/'manifest.json')!=source['manifest_sha256']:raise ValueError('Raw source changed')
    frames=sorted({f for h in plan['observed_histories'][str(source['id'])].values() for f in h})
    records=[]
    for frame in frames:
        if time.time()>deadline-300:return False
        key=f'{source["id"]}-{frame}';out=root/f'frame-{frame}.npz';rec=out.with_suffix('.json')
        if rec.exists():
            saved=json.loads(rec.read_text())
            if saved['identity']!=plan['structured_identity'] or file_hash(out)!=saved['sha256']:raise ValueError('Partial relaxed source changed')
            records.append(saved);continue
        record=plan['archive_cells'][key];cold,box,ids,provenance=archived_positions(plan,record,source)
        np.testing.assert_array_equal(ids,raw.atom_ids)
        item=cold_graphs(plan,source,frame,cold,box,raw)
        features=encode(model,item['clouds'],plan['scale'],'cuda',plan['reuse_config']['extraction_batch'])[item['mapping']]
        information=observed_information(cold,box,np.searchsorted(raw.atom_ids,source['center_atom_ids']))
        np.savez(out,features=features,relative=item['relative'],information=information,query_atom_ids=item['query_atom_ids'])
        saved=dict(identity=plan['structured_identity'],frame=frame,sha256=file_hash(out),archive=record['archive'],provenance=provenance)
        save_json(rec,saved);records.append(saved)
    values={k:[] for k in ('features','relative','information')}
    for frame in frames:
        with np.load(root/f'frame-{frame}.npz') as a:
            for k in values:values[k].append(a[k])
    np.savez(root/'observations.npz',frames=np.array(frames),**{k:np.stack(v) for k,v in values.items()})
    save_json(receipt,dict(identity=plan['structured_identity'],sha256=file_hash(root/'observations.npz'),frames=frames,cells=records))
    return True


def precision(plan):
    """Compare exact same quenched cells before/after archived quantization."""
    from src.data.relaxed_targets.worker import verify_archive
    root=resolve_path(plan['reuse_config']['output'])/'technical';model=load_encoder(plan,'cuda')
    benchmark=resolve_path('output/hardware_benchmark/relaxation-cuda-20260921/technical/results')
    records=[];exact_all=[];half_all=[]
    for name in ('liquid-400K','liquid-520K','transition-520K','post-onset-520K'):
        result=json.loads((benchmark/f'h100-{name}-r0.json').read_text());archive=Path(result['archive']);verify_archive(archive)
        source=next(s for s in plan['sources'] if s['id']==result['case']['source']);frame=result['case']['frame']
        raw=ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
        cold,meta=read_relaxed(archive);box=raw.box_high[frame].astype(float)-raw.box_low[frame].astype(float)
        cold=np.mod(cold-raw.box_low[frame],box)
        half=TemporalLAMMPSBinaryTrajectory.load(archive/'relaxed_binary_float16');half.verify_checksums()
        quantized=np.mod(half.positions[0].astype(float),box)
        a=cold_graphs(plan,source,frame,cold,box,raw);b=cold_graphs(plan,source,frame,quantized,box,raw)
        z=encode(model,a['clouds'],plan['scale'])[a['mapping']];q=encode(model,b['clouds'],plan['scale'])[b['mapping']]
        # Fixed observed IDs isolate storage precision from query reassignment.
        exact_all.append(z.reshape(-1,128));half_all.append(q.reshape(-1,128))
        cosine=np.sum(z*q,-1)/np.maximum(np.linalg.norm(z,axis=-1)*np.linalg.norm(q,axis=-1),1e-12)
        records.append(dict(case=name,source=source['id'],frame=frame,median_cosine=float(np.median(cosine)),
            relative_rms=float(np.sqrt(np.mean((z-q)**2)/np.mean(z**2))),query_identity_agreement=float(np.mean(a['query_atom_ids']==b['query_atom_ids']))))
    z=np.concatenate(exact_all);q=np.concatenate(half_all);g=plan['reuse_config']['quantization_gate']
    relative=float(np.sqrt(np.mean((z-q)**2)/np.mean(z**2)))
    cosine=float(np.median(np.sum(z*q,-1)/np.maximum(np.linalg.norm(z,axis=-1)*np.linalg.norm(q,axis=-1),1e-12)))
    passed=relative<=g['feature_relative_rms_max'] and cosine>=g['median_cosine_min']
    save_json(root/'precision.json',dict(passed=passed,relative_rms=relative,median_cosine=cosine,cases=records,
        interpretation='Engineering tolerance, not proof of identical downstream predictive skill; matched archival precision is disclosed.'))
    if not passed:raise ValueError('Archived-coordinate feature fidelity exceeds the declared tolerance; inspect precision.json')


def report(plan):
    root=resolve_path(plan['reuse_config']['output']);rows=['# Reused relaxed-MACE symmetric contexts','',
        'Matched archived origins, three real observations with actual time offsets; original dense MD targets, one seed. No new relaxation or GATr.',
        '', '| Fit | State | Brier | AP at 12 ps | Detected-window timing MAE at 12 ps (ps) |','|---|---|---:|---:|---:|']
    for spec in json.loads((root/'technical/queue.json').read_text()):
        folder=root/'technical/runs'/spec['name'];status=folder/'status.json';m=folder/'metrics.json'
        state=json.loads(status.read_text())['state'] if status.exists() else 'pending'
        values=' | | '
        if m.exists():
            value=json.loads(m.read_text());ap=value['short_horizon']['classification']['12.0']['average_precision']
            timing=value['short_horizon']['timing']['12.0']['detected_timing_mae_ps']
            displayed='' if timing is None else f'{timing:.3f}'
            values=f'{value["dense_integrated_brier"]:.5f} | {ap:.5f} | {displayed}'
        rows.append(f'| {spec["name"]} | {state} | {values} |')
    (root/'RESULTS.md').write_text('\n'.join(rows)+'\n')


def worker(plan,lane,*,dispatch=False):
    from src.research.crystallization_transfer.runtime import setup
    from src.research.crystallization_paths.runtime import fit
    from .reuse_data import ReusePaths
    setup();root=resolve_path(plan['reuse_config']['output'])/'technical';deadline=deadline_for_job()
    def state(value,**kw):save_json(root/f'lane-{lane}.json',dict(state=value,pid=os.getpid(),updated_at=time.time(),**kw))
    try:
        if not json.loads((root/'precision.json').read_text())['passed']:raise ValueError('Archived precision preflight failed')
        model=None;cache=resolve_path(plan['reuse_config']['context_cache'])
        for source in plan['sources']:
            with lock(cache/str(source['id'])/'worker.lock') as acquired:
                if not acquired:continue
                if (cache/str(source['id'])/'complete.json').exists():continue
                if time.time()>deadline-600:state('checkpointed',stage='extraction');return
                if model is None:model=load_encoder(plan,'cuda')
                state('extracting',source=source['id'])
                if not extract_source(plan,source,model,deadline):state('checkpointed',stage='extraction');return
        del model;torch.cuda.empty_cache()
        if not all((cache/str(s['id'])/'complete.json').exists() for s in plan['sources']):
            state('preparation_pending');return
        if dispatch:launch_fit_workers(plan)
        data=None;domain=None
        for spec in json.loads((root/'queue.json').read_text()):
            folder=root/'runs'/spec['name'];status=folder/'status.json'
            if status.exists() and json.loads(status.read_text())['state']=='complete':continue
            with lock(folder/'worker.lock') as acquired:
                if not acquired:continue
                if time.time()>deadline-1800:state('checkpointed',stage='training');return
                if domain!=spec['observation_domain']:
                    del data;torch.cuda.empty_cache();state('loading',domain=spec['observation_domain'])
                    data=ReusePaths(plan,spec);domain=spec['observation_domain']
                state('training',fit=spec['name']);save_json(folder/'spec.json',spec)
                if not fit(plan,spec,data,deadline):state('checkpointed',stage='training',fit=spec['name']);return
                report(plan)
        state('finished_lane');report(plan)
    except Exception as exc:
        state('failed',error=repr(exc),traceback=traceback.format_exc());raise


def launch_fit_workers(plan):
    """Request fitting GPUs only once the shared feature preparation is complete."""
    config=plan['reuse_config'];root=resolve_path(config['output'])/'technical'
    with lock(root/'dispatch.lock') as acquired:
        if not acquired:return
        path=root/'fit-launches.json'
        records=json.loads(path.read_text()) if path.exists() else []
        code=(root/'code').resolve()
        save_json(root/'run-config.json',config)
        for lane in range(len(records),config['new_workers']-1):
            command=[sys.executable,'-u','-m','src.research.structured_context.reuse','worker','--config',str(root/'run-config.json'),'--lane',f'fit-{lane}']
            script=root/f'fit-{lane}.sbatch'
            lines=['#!/bin/bash',f'#SBATCH --job-name=relaxed-reuse-fit-{lane}',f'#SBATCH --partition={config["partition"]}',
                '#SBATCH --gres=gpu:1','#SBATCH --cpus-per-task=4','#SBATCH --mem=32G',f'#SBATCH --time={config["hours"]}:00:00',
                f'#SBATCH --output={root}/fit-{lane}-%j.log','set -euo pipefail','cd '+shlex.quote(str(code)),
                'export PCM_PROJECT_ROOT='+shlex.quote(str(code)),
                'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1',
                'exec '+shlex.join(command)]
            script.write_text('\n'.join(lines)+'\n');job=subprocess.check_output(['sbatch','--parsable',str(script)],text=True).strip()
            records.append(dict(lane=lane,job=job));save_json(path,records)


def submit(config_path,allocation):
    config=json.loads(resolve_path(config_path).read_text());plan=freeze(config);root=resolve_path(config['output'])/'technical'
    if (root/'launches.json').exists():raise FileExistsError('Reuse queue already submitted')
    if not json.loads((root/'validation.json').read_text())['passed']:raise ValueError('Reuse preflight must pass')
    if allocation is None:raise ValueError('Specify the existing allocation for feature preparation')
    code=snapshot(root)
    command=['srun',f'--jobid={allocation}','--overlap','--exact','-N1','-n1','-c4','--gres=gpu:1',
        sys.executable,'-u','-m','src.research.structured_context.reuse','worker','--config',str(code/config_path),'--lane','allocated','--dispatch']
    env=dict(os.environ,PCM_PROJECT_ROOT=str(code),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    with (root/'allocated.log').open('a') as log:
        process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    records=[dict(allocation=allocation,pid=process.pid,lane='allocated',dispatch_fitting_workers_after_features=True)]
    save_json(root/'launches.json',records)
    report(plan);print(json.dumps(records),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','precision','verify','worker','submit','report']);p.add_argument('--config',required=True);p.add_argument('--lane',default='manual');p.add_argument('--allocation',type=int);p.add_argument('--dispatch',action='store_true');a=p.parse_args()
    config=json.loads(resolve_path(a.config).read_text());plan=freeze(config);torch.set_num_threads(1)
    if a.stage=='precision':precision(plan)
    elif a.stage=='verify':
        from .reuse_verify import verify
        verify(plan)
    elif a.stage=='worker':worker(plan,a.lane,dispatch=a.dispatch)
    elif a.stage=='submit':submit(a.config,a.allocation)
    elif a.stage=='report':report(plan)

if __name__=='__main__':main()
