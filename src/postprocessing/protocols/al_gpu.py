"""Audited Al additions: GPU descriptors, CPU lineage, then catalogue publication.

Input populations come from the ordinary production inventory. This adapter
preserves their native observations and runs the existing GPU and lineage APIs.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import REPO, machine, resolve_path
from src.postprocessing.io import read
from src.postprocessing import extraction, search
from src.postprocessing.datasets import births
from src.postprocessing.protocols.ni_gpu import reviewed_backend, verify_cache


def baseline(config):
    """Record existing identities so publication cannot silently relabel them."""
    with births.NucleiBirthDataset('010', config=config) as data:
        return dict(fingerprint=data.fingerprint,
            established=[dict(row) for row in data.db.execute(
                'SELECT * FROM events WHERE established=1 ORDER BY event_id')],
            tasks=[dict(row) for row in data.db.execute('SELECT * FROM tasks ORDER BY task_id')],
            native030=[dict(row) for row in data.db.execute(
                "SELECT * FROM views WHERE cadence='030' ORDER BY view_id")])


def prepare(c):
    backend = reviewed_backend(c)
    output = resolve_path(c['search']['output']); tech = output/'technical'
    if (tech/'plan.json').exists():
        raise ValueError('Already prepared; use the frozen preparation')
    check_metric_docs(family=search.FAMILY)
    sources=[]; coverage=[]; inputs=[]; seen=set(); parents=[]
    fields=('cadences','detectors','ptm_rmsd_cutoff','neighbor_cutoff_A','bond_order',
            'minimum_tracked_atoms','chunk_frames')
    for reference in c['input_plans']:
        path=resolve_path(reference); parent=births.checked_plan(path)
        if parents and any(parent['config'][k]!=parents[0]['config'][k] for k in fields):
            raise ValueError(f'Incompatible extraction definitions: {path}')
        for source in parent['sources']:
            if source['input_identity'] in seen:
                raise ValueError(f'Duplicate source input: {source["id"]}')
            if set(source['views']) != {'010'} or abs(source['native_cadence_ps']-.1)>1e-8:
                raise ValueError(f'Expected a native 0.10-ps source: {source["id"]}')
            seen.add(source['input_identity']); sources.append(source)
        coverage.extend(parent['coverage']); parents.append(parent)
        inputs.append(dict(path=str(path),sha256=sha(path),identity=parent['identity']))
    if len(sources)!=c['expected_sources']:
        raise ValueError(f'Population differs: {len(sources)} vs {c["expected_sources"]}')
    pc=dict(parents[0]['config'], output=c['search']['parent_output'],cache=c['cache'])
    pc['protocol']='al_cluster_structure_gpu_additions_v1'
    for key in ('registry','exclude_plan','production_inventory'):
        pc.pop(key,None)
    pc['input_contract']=c['search']['input_contract']
    parent=dict(config=pc,sources=sources,coverage=coverage,input_plans=inputs,
                descriptor_backend=backend,jobs=[],gpu_assignments=[])
    loads=[0]*c['gpu_lanes']
    for source in sorted(sources,key=lambda s:-len(s['sampled_frames'])*s['atom_count']):
        lane=min(range(c['gpu_lanes']),key=loads.__getitem__)
        loads[lane]+=len(source['sampled_frames'])*source['atom_count']
        parent['gpu_assignments'].append(dict(source=source['id'],lane=lane))
        parent['jobs'].extend(dict(source=source['id'],cadence='010',detector=d)
                              for d in pc['detectors'])
    parent['identity']=digest(parent)
    write_json(resolve_path(pc['output'])/'technical/plan.json',parent)
    write_json(tech/'search-config.json',c['search'])
    write_json(tech/'baseline.json',baseline(resolve_path(c['dataset_config'])))
    plan=search.prepare(c['search'])
    write_json(tech/'preparation.json',dict(config=c,parent_identity=parent['identity'],
        identity=plan['identity'],sources=len(sources),frames=sum(len(s['sampled_frames']) for s in sources),
        ancestors=len({s['ancestry_group'] for s in sources}),gpu_lane_atom_frames=loads,
        backend=backend,prepared_at=datetime.now(timezone.utc).isoformat()))
    return dict(identity=plan['identity'],sources=len(sources),frames=sum(len(s['sampled_frames']) for s in sources))


def load(c):
    tech=resolve_path(c['search']['output'])/'technical'
    if read(tech/'preparation.json')['config']!=c:
        raise ValueError('Changed prepared workflow; use the frozen configuration')
    p,parent=search.load(tech/'search-config.json')
    if parent['descriptor_backend']!=reviewed_backend(c):
        raise ValueError('GPU backend differs from reviewed preparation')
    for reference in parent['input_plans']:
        if sha(reference['path'])!=reference['sha256']:
            raise ValueError(f'Input population changed: {reference["path"]}')
    return tech,p,parent


def extract(c,index):
    from src.postprocessing.gpu_extract import extract as gpu_extract
    import torch
    tech,p,parent=load(c); ids={r['source'] for r in parent['gpu_assignments'] if r['lane']==index}
    sources=[s for s in parent['sources'] if s['id'] in ids]
    destination=resolve_path(parent['config']['output'])/'technical'
    bindings=[]
    with recorded_stage(tech/f'gpu-{index}.json',total_sources=len(sources),completed_sources=0,
                        gpu=torch.cuda.get_device_name(),job=os.environ.get('SLURM_JOB_ID')) as status:
        for source in sources:
            arguments=dict(source=source,frames=source['sampled_frames'],config=parent['config'],
                output=str(extraction.source_root(parent,source)),chunk_frames=parent['config']['chunk_frames'],
                prefetch=c['prefetch'],compression_workers=c['compression_workers'],
                ptm_ordering=c['ptm_ordering'],execution=c['execution'])
            write_json(tech/'gpu-configs'/(source['id']+'.json'),arguments)
            done=gpu_extract(**arguments)
            bindings.append(verify_cache(dict(parent=parent),source))
            status.update(completed_sources=len(bindings),last_source=source['id'],
                          last_frames=done['frames'],last_seconds=done['seconds'])
            print(json.dumps(dict(source=source['id'],frames=done['frames'],seconds=done['seconds'])),flush=True)
        write_json(destination/f'gpu-lane-{index}.json',dict(state='complete',parent_identity=parent['identity'],sources=bindings))


def extraction_receipts(c,parent):
    destination=resolve_path(parent['config']['output'])/'technical'; bindings={}
    for index in range(c['gpu_lanes']):
        done=read(destination/f'gpu-lane-{index}.json')
        if done['state']!='complete' or done['parent_identity']!=parent['identity']:
            raise ValueError(f'Incomplete GPU lane {index}')
        expected={r['source'] for r in parent['gpu_assignments'] if r['lane']==index}
        if {r['source'] for r in done['sources']}!=expected:
            raise ValueError(f'GPU lane population mismatch: {index}')
        bindings.update({r['source']:r for r in done['sources']})
    if set(bindings)!={s['id'] for s in parent['sources']}:
        raise ValueError('Incomplete GPU population')
    return bindings


def track(c,index):
    tech,p,parent=load(c); bindings=extraction_receipts(c,parent)
    selected={t['source'] for t in p['jobs'] if t['lane']==index}
    for source in parent['sources']:
        if source['id'] in selected and verify_cache(dict(parent=parent),source)!=bindings[source['id']]:
            raise ValueError(f'Changed GPU input for tracking: {source["id"]}')
    search.lane(tech/'search-config.json',index)


def collect(c):
    tech,p,parent=load(c); bindings=extraction_receipts(c,parent)
    # Every source cache was verified before tracking; retain the exact bindings.
    write_json(resolve_path(parent['config']['output'])/'technical/gpu-complete.json',
               dict(state='complete',parent_identity=parent['identity'],sources=bindings,
                    backend=parent['descriptor_backend']))
    search.collect(tech/'search-config.json')


def publish(c):
    tech,p,parent=load(c)
    completion=read(tech/'complete.json')
    if completion['identity']!=p['identity'] or completion['jobs']!=len(p['jobs']):
        raise ValueError('Search is not complete; catalogue publication refused')
    config=Path(__file__).resolve().parents[3]/c['dataset_config']; previous=read(tech/'baseline.json')
    with recorded_stage(tech/'publication.json',previous_fingerprint=previous['fingerprint']) as status:
        manifest=births.refresh(read(config))
        with births.NucleiBirthDataset('010',config=config) as data:
            data.verify()
            for row in previous['tasks']:
                actual=data.db.execute('SELECT * FROM tasks WHERE task_id=?',(row['task_id'],)).fetchone()
                if actual is None or dict(actual)!=row: raise ValueError('Historical task changed')
            for row in previous['established']:
                actual=data.db.execute('SELECT * FROM events WHERE event_id=?',(row['event_id'],)).fetchone()
                if actual is None or dict(actual)!=row: raise ValueError('Historical established record changed')
            current=[dict(r) for r in data.db.execute("SELECT * FROM views WHERE cadence='030' ORDER BY view_id")]
            if current!=previous['native030']:raise ValueError('Native 0.30-ps population changed')
            datasets={r[0] for r in data.db.execute('SELECT dataset FROM sources')}
            if any(s['dataset'] not in datasets for s in parent['sources']):
                raise ValueError('New source missing from published catalogue')
        status.update(fingerprint=manifest['fingerprint'],summary=manifest['summary'],
                      verified=True,historical_records_unchanged=True,native030_unchanged=True)


def submit(c):
    tech,p,parent=load(c)
    if (tech/'gpu-launch.json').exists():raise ValueError('Already submitted; consult existing launch receipt')
    bundle=ExecutionBundle.freeze(REPO,tech/'gpu-code',c,directories=('src','configs','docs/metrics'),
                                 files=((REPO/c['numerical_audit'],c['numerical_audit']),))
    settings=machine(); settings['catalog']=str(bundle.root/'configs/datasets.json')
    write_json(tech/'gpu-machine.json',settings)
    env=dict(PCM_PROJECT_ROOT=str(REPO),PCM_MACHINE_CONFIG=str(tech/'gpu-machine.json'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1',
        QT_QPA_PLATFORM='offscreen',CUDA_HOME=c['cuda_home'],PYTHONPATH=str(bundle.root),MAX_JOBS='4',
        PATH=c['cuda_home']+'/bin:'+str(Path(os.sys.executable).parent)+':/usr/bin:/bin',
        TORCH_EXTENSIONS_DIR=str(resolve_path(c['torch_extensions'])))
    receipt=dict(identity=p['identity'],parent_identity=parent['identity'],code=str(bundle.root),jobs={},
                 workflow='GPU Al descriptors -> unchanged CPU lineage -> verified catalogue publication',
                 submitted_at=datetime.now(timezone.utc).isoformat())
    queue=SlurmQueue(tech,bundle,'src.postprocessing.protocols.al_gpu',env,tech/'gpu-launch.json',receipt,'AL-BIRTH-GPU')
    with queue.submission():
        gpu=queue.submit('extract',[f'--array=0-{c["gpu_lanes"]-1}','--gres=gpu:1','--cpus-per-task=4',
            '--mem=32G',f'--time={c["gpu_walltime"]}'],partition=c['gpu_partitions'])
        tracking=queue.submit('track',[f'--array=0-{c["search"]["lanes"]-1}',
            f'--cpus-per-task={c["search"]["workers_per_lane"]}','--mem=64G',f'--time={c["search"]["time_limit"]}'],dependency='afterok:'+gpu)
        collection=queue.submit('collect',['--cpus-per-task=1','--mem=64G','--time=06:00:00'],dependency='afterok:'+tracking)
        queue.submit('publish',['--cpus-per-task=1','--mem=64G','--time=06:00:00'],dependency='afterok:'+collection)
    return receipt


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('prepare','submit','extract','track','collect','publish'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    args=parser.parse_args(); c=read(args.config)
    result=(globals()[args.stage](c,args.index) if args.stage in ('extract','track') else globals()[args.stage](c))
    if result is not None:print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
