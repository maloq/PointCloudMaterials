"""Original-birth transfer versus appearance-supervised combined-pool readouts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import resolve_path
from . import appearance_data as data
from .data import read, load
from .features import descriptors, seal_descriptors, load_bank
from .feature_families import domain_config
from .merged_fit import fit
from .temporal import design
from .temporal_inputs import packet, context

FAMILY='birth_appearance'


def original_config(c, domain):
    m=read(resolve_path(c['original_comparison_config']))
    return domain_config(m,next(d for d in m['domains'] if d['name']==domain))


def specs(c):
    for domain in c['domains']:
        for model,arm in [('linear','rich_linear'),('catboost','rich_gbdt')]:
            for observation in c['observations']:
                yield dict(domain=domain,model=model,arm=arm,observation=observation,
                           bank='descriptors',families=['geometry','bond_order','cna','tda'])


def reference(c, spec):
    return resolve_path(c['original_comparison_output'])/'analyses/fits-v1'/spec['domain']/spec['arm']/spec['observation']['name']


def bind(c):
    check_metric_docs(family=FAMILY)
    prepared=data.prepare(c)
    refs=[]
    for spec in specs(c):
        root=reference(c,spec)/'technical'
        complete=read(root/'complete.json')
        names=['complete.json','binding.json','selection.json','predictions.npz',
               'model.joblib' if spec['model']=='linear' else 'model.cbm']
        if sha(root/'predictions.npz')!=complete['predictions_sha256']:raise ValueError('Changed original predictions')
        refs.append(dict(spec=spec,path=str(root),files={f:sha(root/f) for f in names}))
    result=dict(config=c,data_identity=prepared['identity'],references=refs,
                producer_sha256=sha(Path(__file__)))
    result['identity']=digest(result)
    path=resolve_path(c['output'])/'technical/prepared.json'
    if path.exists() and read(path)!=result:raise ValueError('Changed appearance experiment binding')
    write_json(path,result)
    return result


def inputs(c,spec):
    old=original_config(c,spec['domain']);new=data.cache_config(c,spec['domain'])
    _,a,_=load(old);_,b,_=load(new)
    oldbank,columns=load_bank(old,'descriptors');newbank,names=load_bank(new,'descriptors')
    if names!=columns:raise ValueError('Different physical descriptor schemas')
    select=np.arange(len(columns));obs=spec['observation']
    xx=packet(oldbank,a,select,obs,c['seed']);yy=packet(newbank,b,select,obs,c['seed'])
    keys=['id','source','event','pair','role','atom','label','weight']
    rows={k:np.concatenate([a[k],b[k]]) for k in keys}
    rows['stratum']=np.r_[np.full(len(a['id']),'established'),b['stratum']]
    if len(set(rows['id']))!=len(rows['id']):raise ValueError('Original/transient history overlap')
    train=rows['role']=='train';folds=design(c)['binding']['source_folds']
    assignments=np.asarray([folds[str(s)] for s in rows['source'][train]])
    sources=read(resolve_path(c['output'])/'technical/data-plan.json')['original_plan']['sources']
    train_ancestors={s['lineage'] for s in sources if s['role']=='train'}
    test_ancestors={s['lineage'] for s in sources if s['role']!='train'}
    if train_ancestors & test_ancestors:raise ValueError('Ancestry crosses training/evaluation')
    ctx=dict(encoder=None,predictor=dict(model=spec['model'],bank='442 fixed physical descriptors',
        geometry_features=99,bond_order_features=40,cna_features=45,tda_features=258,
        spatial_support='nearest 80 original atom IDs; descriptor radius 8 A; no external halo',
        **context(obs,.75)),conditions=[],species_input=False,velocity_input=False,
        relaxation=new.get('input_domain'),domain=spec['domain'],
        label='Any observed crystal appearance: established births and verified failed embryos are positive',
        sampling='Retrospective matched case/control; 4 liquid controls per case; equal total weight per event',
        selector='Original training-source five-fold CV NLL; no calibration',
        split='Original train; former selection/calibration/test merged; no new source assignment',
        tracking='Local frozen-descriptor diagnostic; no encoder training or W&B run')
    return old,rows,np.concatenate([xx,yy]),assignments,ctx,len(a['id'])


def run_fit(c,spec):
    b,rows,x,folds,ctx,_=inputs(c,spec)
    dest=resolve_path(c['output'])/'analyses/fits-v1'/spec['domain']/spec['arm']/spec['observation']['name']
    fit(c,b,rows,x,spec,dest,folds,ctx,family=FAMILY)


def support(c):
    coverage=read(resolve_path(c['output'])/'technical/eligibility.json')
    _,new,_=load(data.cache_config(c,'original'));_,old,_=load(original_config(c,'original'))
    records=[]
    for role in ['train','merged_test']:
        for name,rows,mask in [('established',old,np.ones(len(old['id']),bool)),
            ('failed_strong',new,new['stratum']=='failed_strong'),
            ('failed_other',new,new['stratum']=='failed_other'),('failed_all',new,np.ones(len(new['id']),bool))]:
            selected=mask & ((rows['role']=='train') if role=='train' else (rows['role']!='train'))
            positive=selected & (rows['label']==1)
            records.append(dict(role=role,stratum=name,rows=int(selected.sum()),positives=int(positive.sum()),
                controls=int((selected & (rows['label']==0)).sum()),sources=len(set(rows['source'][selected])),
                events=len(set(zip(rows['source'][positive],rows['event'][positive])))))
    root=resolve_path(c['output'])/'analyses/coverage-v1'
    write_metric_rows(records,root,family=FAMILY,name='support')
    write_metric_rows(coverage,root,family=FAMILY,name='candidate-eligibility')


def submit(path):
    c=read(path);p=bind(c);tech=resolve_path(c['output'])/'technical'
    if (tech/'launch.json').exists():raise ValueError('Already submitted; use frozen jobs to resume')
    repo=Path(__file__).resolve().parents[3]
    bundle=ExecutionBundle.freeze(repo,tech/'code',c,directories=('src','configs','docs/metrics'))
    record=dict(identity=p['identity'],jobs={},code=str(bundle.root),original_models=8,combined_fits=8,
                target='any crystal appearance; failed embryos positive')
    queue=SlurmQueue(tech,bundle,'src.research.birth_prediction.appearance',
        dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1'),
        tech/'launch.json',record,'APPEARANCE')
    with queue.submission():
        prep=queue.submit('prepare',[f'--array=0-{c["prepare_workers"]//2-1}','--cpus-per-task=2','--mem=16G','--time=04:00:00'],
            command_stage='group',arguments=['--worker-stage','prepare','--workers-per-group','2'])
        seal=queue.submit('seal',['--cpus-per-task=1','--mem=8G','--time=00:30:00'],dependency='afterok:'+prep)
        desc=queue.submit('descriptors-original',[f'--array=0-{c["prepare_workers"]//2-1}','--cpus-per-task=2','--mem=12G','--time=06:00:00'],dependency='afterok:'+seal,
            command_stage='group',arguments=['--worker-stage','descriptors-original','--workers-per-group','2'])
        ds=queue.submit('seal-original',['--cpus-per-task=1','--mem=8G','--time=00:30:00'],dependency='afterok:'+desc)
        relax=queue.submit('relax',[f'--array=0-{c["relaxation_workers"]//4-1}',f'--cpus-per-task={4*c["relaxation_ranks"]}','--mem=32G','--time=16:00:00'],dependency='afterok:'+seal,
            command_stage='group',arguments=['--worker-stage','relax','--workers-per-group','4'])
        rs=queue.submit('seal-relaxed',['--cpus-per-task=1','--mem=8G','--time=00:30:00'],dependency='afterok:'+relax)
        rd=queue.submit('descriptors-relaxed',[f'--array=0-{c["prepare_workers"]//2-1}','--cpus-per-task=2','--mem=12G','--time=06:00:00'],dependency='afterok:'+rs,
            command_stage='group',arguments=['--worker-stage','descriptors-relaxed','--workers-per-group','2'])
        rds=queue.submit('ready-relaxed',['--cpus-per-task=1','--mem=8G','--time=00:30:00'],dependency='afterok:'+rd)
        fits=[]
        for domain,dep in [('original',ds),('relaxed',rds)]:
            for model in ['linear','boost']:
                name=f'{model}-{domain}'
                options=['--cpus-per-task=8','--mem=16G','--time=04:00:00',f'--exclude={c["excluded_gpu_node"]}'] if model=='boost' else ['--cpus-per-task=8','--mem=16G','--time=04:00:00']
                if model=='boost':options.append('--gpus=1')
                fits.append(queue.submit(name,options,dependency='afterok:'+dep,
                    partition=c['gpu_partition'] if model=='boost' else 'CPU'))
        queue.submit('collect',['--cpus-per-task=4','--mem=12G','--time=01:00:00',f'--nodelist={c["collection_node"]}'],dependency='afterok:'+':'.join(fits))
    return record


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['bind','submit','prepare','seal','descriptors-original','seal-original','relax',
        'seal-relaxed','descriptors-relaxed','ready-relaxed','linear-original','boost-original','linear-relaxed','boost-relaxed','collect','group'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    parser.add_argument('--worker-stage',choices=['prepare','descriptors-original','descriptors-relaxed','relax'])
    parser.add_argument('--workers-per-group',type=int)
    a=parser.parse_args();c=read(a.config)
    if a.stage=='submit':print(json.dumps(submit(a.config),indent=2));return
    if a.stage=='bind':print(json.dumps(bind(c),indent=2));return
    name=f'group-{a.worker_stage}' if a.stage=='group' else a.stage
    with recorded_stage(resolve_path(c['output'])/f'technical/{name}-{a.index}.json',job=os.environ.get('SLURM_JOB_ID')) as progress:
        if a.stage=='group':
            ranks=c['relaxation_ranks'] if a.worker_stage=='relax' else 1
            cpus=sorted(os.sched_getaffinity(0));n=a.workers_per_group
            if len(cpus)<ranks*n:raise RuntimeError('Insufficient CPU affinity for independent workers')
            children=[]
            for slot in range(n):
                index=a.index*n+slot;mask=','.join(map(str,cpus[slot*ranks:(slot+1)*ranks]))
                command=['taskset','-c',mask,sys.executable,'-u','-m',__package__+'.appearance',
                         a.worker_stage,'--config',str(Path(a.config).absolute()),'--index',str(index)]
                log=resolve_path(c['output'])/f'technical/{a.worker_stage}-worker-{index}.log'
                with log.open('a') as stream:children.append((index,subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)))
            codes={index:child.wait() for index,child in children}
            if any(codes.values()):raise RuntimeError(f'Grouped workers failed: {codes}')
        elif a.stage=='prepare':
            p=read(resolve_path(c['output'])/'technical/data-plan.json')
            for item in p['original_plan']['sources'][a.index::c['prepare_workers']]:
                progress.update(source=item['id']);r=data.prepare_source(c,p,item)
                print(json.dumps(r),flush=True)
        elif a.stage=='seal':
            data.seal(c);support(c);p=data.relaxation_plan(c);progress.update(cells=len(p['tasks']))
        elif a.stage.startswith('descriptors-'):descriptors(data.cache_config(c,a.stage.split('-')[1]),a.index)
        elif a.stage=='seal-original':seal_descriptors(data.cache_config(c,'original'))
        elif a.stage=='ready-relaxed':seal_descriptors(data.cache_config(c,'relaxed'))
        elif a.stage=='relax':
            p=data.relaxation_plan(c)
            for i,task in enumerate(p['tasks'][a.index::c['relaxation_workers']]):
                progress.update(task=task['id'],completed=i);r=data.cell(p,task)
                print(json.dumps(dict(task=task['id'],seconds=r['seconds'])),flush=True)
        elif a.stage=='seal-relaxed':data.seal_relaxed(c)
        elif a.stage=='collect':
            from .appearance_analysis import collect
            collect(c,progress)
        else:
            model,domain=a.stage.split('-')
            for spec in specs(c):
                if spec['domain']==domain and spec['model']==('catboost' if model=='boost' else 'linear'):
                    progress.update(treatment=spec);run_fit(c,spec)


if __name__=='__main__':main()
