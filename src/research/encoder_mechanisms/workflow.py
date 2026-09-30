"""Frozen Slurm queue for the declared encoder-mechanism interventions."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import sha,digest,write_json


def load(path):
    c=json.loads(Path(path).read_text())
    if c['protocol']!='encoder_mechanisms_v1' or c['external_inputs']!=[]:raise ValueError('Wrong protocol/input contract')
    if c['batch_size']!=256 or c['microbatch']!=256:raise ValueError('This matched study requires batch/microbatch 256')
    if len(c['seeds'])!=3 or len(set(c['seeds']))!=3:raise ValueError('Require three declared independent seeds')
    if c['pretraining']['epochs']!=24:raise ValueError('Fixed 24-epoch endpoint required')
    from src.research.supervised_onset.tracking import require_online
    require_online(c['wandb'])
    return c


def bind(config,root):
    repo=Path(__file__).resolve().parents[3]
    packages=['encoder_mechanisms','encoder_context','encoder_quality','supervised_onset','mace_epi']
    files=[p for package in packages for p in (repo/'src/research'/package).glob('*.py')]
    files += [repo/p for p in ('src/models/encoders/spatial_mace.py','src/models/encoders/mace_backend.py',
        'src/models/encoders/graph_bank.py','src/data/structural_pretraining/native_dataset.py',
        'src/training_methods/regularizers.py','src/training_methods/structural_pretraining/objective.py')]
    record=dict(config=config,implementation={str(p.relative_to(repo)):sha(p) for p in files})
    path=root/'technical/identity.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text())!=record:raise ValueError(f'Frozen mechanism run changed: {root}')
        else:write_json(path,record)
    return digest(record)


def treatment_study(c,seed,arm):
    root=resolve_path(c['output'])/'runs'/str(seed)/arm['name']
    config=dict(c,seed=seed,output=str(root),pretraining=dict(c['pretraining'],objective=arm['objective']))
    config['shared_initialization']=str(resolve_path(c['output'])/'runs'/str(seed)/'initial.pt')
    identity=bind(config,root)
    write_json(root/'technical/prediction-context.json',dict(
        encoder_inputs=['current centered normalized coordinates','constant atom channel','center indicator'],
        predictors='none during label-free training; frozen exported state in subsequent readouts',
        external_inputs=[],spatial=dict(candidate_atoms=80,radius_A=8,edge_cutoff_A=5,blocks=2,halo=False),
        history_frames=1,motion=False,conditions=[],inference_relaxation=False,
        training_only_teacher='same-time observed/relaxed views; frozen initial random geometric reference for Epi',
        material_scale='fixed material preprocessing; never an input channel',
        selection='fixed 24 epochs primary, fixed 12 secondary; no label-based encoder choice',
        fixed_dataset=c['fixed_dataset'],structural_dataset=c['structural_dataset']))
    return SimpleNamespace(config=config,identity=identity,root=root,technical=root/'technical')


def quality_config(c,output):
    from src.research.encoder_quality.common import load as load_quality
    q=load_quality(resolve_path(c['quality_template']))
    q.update(output=str(output),feature_cache=str(resolve_path(c['feature_cache'])),
        seed=c['evaluation_seed'],wandb=dict(c['wandb']),batch_size=256)
    q['models']=[]
    return q


def evaluate_checkpoint(c,path,kind,name,output,device,seed):
    from src.research.encoder_quality.common import bind as bind_quality
    from src.research.encoder_quality.run import run
    q=quality_config(c,output)
    spec=dict(name=name,checkpoint=str(path),checkpoint_sha256=sha(path),kind=kind,domain='hot',
        producer=str(Path(__file__).resolve().parents[3]),origin=str(path.parent),
        seed=seed,encoder_training='likelihood adaptation' if kind=='adapted' else 'label-free' if kind=='pretrained' else 'initialization',
        inputs='geometry only; constant atom channel; no covariates; observed inference')
    initialized=kind=='initial' or name.startswith('scratch-')
    spec['training_context']=dict(pretraining='none' if initialized else 'paired same-time observed and relaxed structural patches',
        teacher='fixed initial random geometry projection' if not initialized and not name.startswith('R2-') else None,
        supervised_teacher=None,current_encoder_training=False,
        pool_normalization='same-seed fixed train structural rows, retained at initialization',
        likelihood_adaptation=kind=='adapted' and not name.startswith('frozen-'))
    q['models']=[spec];write_json(output/'technical/config.json',q)
    identity=bind_quality(q);run(q,identity,spec,device)
    if kind=='adapted':
        import numpy as np
        import torch
        from torch import nn
        from src.research.encoder_quality.common import corpus,probe_study
        from src.research.encoder_quality.metrics import score
        from src.research.local_predictability.metrics import cumulative_risk
        from src.research.equivariant_context.cache import RetainedCache
        from src.experiment_runner.metric_docs import write_metric_table
        saved=torch.load(path,map_location=device,weights_only=False)
        head=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,5)).to(device)
        head.load_state_dict({k.removeprefix('hazard.'):v for k,v in saved['model'].items() if k.startswith('hazard.')})
        key=digest(dict(identity=identity,checkpoint=spec['checkpoint_sha256']))
        with RetainedCache(q['feature_cache'],6).lease(key,deadline=time.time()+3600,
                metadata=dict(artifact='encoder-quality',model=name,checkpoint_sha256=spec['checkpoint_sha256'])) as cache:
            z=np.load(cache/'population.npy')
        with torch.no_grad():risk=cumulative_risk(head(torch.as_tensor(z,device=device))).cpu().numpy()
        data=corpus(q);metrics,calibrated=score(data,risk)
        dest=output/'analyses/joint-head';dest.mkdir(parents=True,exist_ok=True)
        np.savez(dest/'predictions.npz',risks=risk,calibrated=calibrated,
            **{k:data.pop[k] for k in ('sample_id','source','event','role')})
        write_json(dest/'technical/metrics.json',metrics)
        write_metric_table(metrics,dest,family='encoder_mechanisms')
    from .rearrangement import evaluate as displacement_evaluate
    if not (output/'analyses/displacement/technical/metrics.json').exists():
        displacement_evaluate(c,q,spec,output,device)
    return q


def evaluate_checkpoint_isolated(c,path,kind,name,output,device,seed,*,displacement_only=False):
    """Keep each checkpoint's compiled models in a fresh process."""
    request=output/'technical/evaluation-request.json'
    write_json(request,dict(config=c,path=str(path),kind=kind,name=name,output=str(output),
        device=device,seed=seed,displacement_only=displacement_only))
    subprocess.run([sys.executable,'-u','-m','src.research.encoder_mechanisms.recovery',
        'checkpoint','--request',str(request)],check=True)


def adapt(c,seed,pretrain_study,device,deadline,*,mode=None):
    from src.research.supervised_onset.common import Study
    from src.research.supervised_onset.data import Corpus
    from src.research.supervised_onset.train import make_banks,fit
    root=resolve_path(c['output'])
    for mode in (('frozen','finetune','scratch') if mode is None else (mode,)):
        cfg=json.loads(resolve_path(c['supervised_template']).read_text())
        dest=root/'runs'/str(seed)/f'adapt-{mode}'
        epoch=0 if mode=='scratch' else 24
        cfg.update(seed=seed,output=str(dest),cache=c['population_cache'],wandb=dict(c['wandb'],
            display_name=f'Mechanisms | {mode} | seed {seed}'),
            initial_encoder=dict(checkpoint=str(pretrain_study.root/f'pretraining/epi_variance/technical/epoch-{epoch:03d}.pt'),
                method='epi_variance',epoch=epoch))
        steps=math.ceil(c['prediction_train_rows']/256)
        cfg['training'].update(batch_size=256,microbatch=256,epochs=24,maximum_updates=24*steps,
            screen_updates=24*steps,evaluate_every=steps,save_every=steps,checkpoint_epochs=[12,24],
            minimum_selection_epoch=12,freeze_encoder=mode=='frozen',record_initial_state=True)
        cfg['baselines'].update(batch_size=256,epochs=24,updates=24*steps,minimum_selection_epoch=12)
        path=dest/'technical/config.json';write_json(path,cfg)
        study=Study(path);study.bind();data=Corpus(study)
        if len(data.split['train'])!=c['prediction_train_rows']:raise ValueError('Matched prediction population changed')
        state=study.technical/'runs/O-NLL/training-state.json'
        if not state.exists() or json.loads(state.read_text())['state']!='update_budget_complete':
            banks=make_banks(study,data,device)
            result=fit(study,data,banks,'O-NLL',until=deadline,max_updates=24*steps,device=device)
            del banks
            if result['state']!='update_budget_complete':raise TimeoutError('Adaptation saved for resume')
        import torch
        initial=torch.load(study.technical/'runs/O-NLL/epoch-000.pt',map_location='cpu',weights_only=False)
        frozen_init=pretrain_study.root/f'pretraining/epi_variance/technical/epoch-{epoch:03d}.pt'
        original=torch.load(frozen_init,map_location='cpu',weights_only=False)['encoder']
        for key,value in original.items():
            if not torch.equal(value,initial['model']['encoder.'+key]):raise ValueError('Adaptation initialization drifted')
        if mode=='frozen':
            final=torch.load(study.technical/'runs/O-NLL/last.pt',map_location='cpu',weights_only=False)
            for key,value in original.items():
                if not torch.equal(value,final['model']['encoder.'+key]):raise ValueError('Frozen encoder changed during head fitting')
        if mode!='frozen':
            matched=torch.load(root/f'runs/{seed}/adapt-frozen/technical/runs/O-NLL/epoch-000.pt',map_location='cpu',weights_only=False)
            for key,value in initial['model'].items():
                if key.startswith('hazard.') and not torch.equal(value,matched['model'][key]):raise ValueError('Fresh head initialization differs')
        write_json(study.technical/'matched-input-checks.json',dict(initial_encoder_exact=True,
            frozen_encoder_unchanged=mode=='frozen',same_seed_fresh_head=True,external_inputs=[]))
        for checkpoint in ('best','epoch-024'):
            if time.time()>deadline-600:raise TimeoutError('Adaptation evaluation awaits resumed allocation')
            evaluate_checkpoint_isolated(c,study.technical/f'runs/O-NLL/{checkpoint}.pt','adapted',f'{mode}-{checkpoint}',
                root/f'analyses/adaptation/{seed}/{mode}/{checkpoint}',device,seed)
            from src.research.supervised_onset.tracking import update_training_summary
            metrics=root/f'analyses/adaptation/{seed}/{mode}/{checkpoint}/analyses/joint-head/technical/metrics.json'
            update_training_summary(study,'O-NLL',
                {f'evaluation/{checkpoint}':json.loads(metrics.read_text())},evaluation=checkpoint)


def check(c):
    """Real-data gradient/identity preflight; deliberately no online debug run."""
    import numpy as np
    import torch
    from src.data.structural_pretraining.native_dataset import NativeStructuralDataset
    from src.research.supervised_onset.model import CapacityEncoder
    from src.research.encoder_context.geometry import graph
    from src.research.mace_epi.objective import Objective
    data=NativeStructuralDataset(c['structural_dataset']['root'],'train',paired=True,normalization=c['structural_dataset']['normalization'])
    if data.identity!=c['structural_dataset']['identity'] or len(data)!=c['structural_train_rows']:raise ValueError('Wrong paired release')
    batch=data.batch(np.arange(256));data.close()
    if set(batch)!={'hot','cold'}:raise ValueError('Unexpected data fields')
    torch.manual_seed(c['seeds'][0]);model=CapacityEncoder(**c['encoder'],d0=2.8,n_ref=80.).cuda()
    values={k:torch.as_tensor(v,device='cuda') for k,v in batch.items()}
    with torch.no_grad():
        pooled=model.pooled_graph(graph(values['hot'],model))
        model.pooled_mean.copy_(pooled.mean(0));model.pooled_scale.copy_(pooled.std(0,unbiased=False).clamp_min(1e-5))
    result=[]
    for arm in c['arms']:
        torch.cuda.synchronize();started=time.monotonic()
        model.zero_grad(set_to_none=True)
        z=torch.stack([model(graph(values[d],model)) for d in ('hot','cold')],1)
        projection=torch.randn(128,64,device='cuda')/128**.5
        obj=Objective(**arm['objective']).cuda()
        loss,terms=obj(None,z.reshape(-1,128),dict(index=np.arange(256),reservoir=z.detach()@projection))
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError(f'Invalid real-batch gradient: {arm["name"]}')
        result.append(dict(arm=arm['name'],loss=float(loss.detach()),alignment=float(terms['alignment'].detach()),finite_gradients=True,
            seconds=time.monotonic()-started,timing_scope='Uncompiled real-batch gradient check; excludes frozen-reference forwards'))
    root=resolve_path(c['output'])
    write_json(root/'technical/checks.json',dict(passed=True,arms=result,rows=256,encoder_parameters=sum(p.numel() for p in model.parameters()),
        returned_fields=list(batch),wandb_runs=0,identity=bind(c,root),torch_version=torch.__version__,
        numpy_version=np.__version__,gpu=torch.cuda.get_device_name(0)))


def report(c):
    root=resolve_path(c['output']);technical=root/'technical'
    rows=[]
    for index in range(9):
        path=technical/f'alignment-{index}.json'
        state=json.loads(path.read_text())['state'] if path.exists() else 'queued'
        seed=c['seeds'][index//3];arm=c['arms'][index%3]['name']
        rows.append(f'| {seed} | {arm} | {state} | [Milestones](analyses/alignment/{seed}/{arm}/) |')
    text=['# Encoder mechanism queue','',
        'Nine label-free fits (24 epochs, three treatments × three seeds), followed by nine likelihood fits from the predeclared R1 initialization.',
        'Self-supervised selection is the fixed endpoint; supervised selection is validation hazard NLL after epoch12. AP is diagnostic only.','',
        '[Submission receipt](technical/launch.json) · [Scientific config](technical/code/configs/encoder_mechanisms/alignment_readout_20260926.json)',
        '[Readout controls](controls/analyses/) · [Adaptation comparisons](analyses/adaptation/) · [Birth coverage](analyses/birth-availability/)','',
        '| Seed | Treatment | Pipeline state | Analysis |','| --- | --- | --- | --- |',*rows,'',
        'Pipeline completion includes milestone evaluation; R1 also includes frozen/fine-tuned/scratch likelihood comparisons. ',
        'Birth prediction and linear VAMP remain gated on diagnostic evidence. Missing metrics are pending, not zeros.','']
    with (technical/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);(root/'README.md').write_text('\n'.join(text))


def worker(c,stage,index):
    import torch
    from src.training_methods.shared_pretraining.queue import deadline_for_job
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    root=resolve_path(c['output']);technical=root/'technical'
    identity=bind(c,root/'runs' if stage=='alignment' else root)
    state=technical/f'{stage}-{index}.json'
    with (technical/f'{stage}-{index}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write_json(state,dict(state='running',stage=stage,index=index,job=os.environ.get('SLURM_JOB_ID')))
            if stage=='controls':
                from .controls import run
                from src.research.encoder_quality.common import load as load_quality
                q=load_quality(resolve_path(c['quality_template']));previous=q['output']
                q.update(output=str(root/'controls'),feature_cache=str(resolve_path(c['feature_cache'])),wandb=c['wandb'])
                (Path(q['output'])/'technical').mkdir(parents=True,exist_ok=True)
                run(q,identity,previous,'cuda')
            elif stage=='alignment':
                from src.research.encoder_context.stream_pretrain import run
                seed=c['seeds'][index//len(c['arms'])];arm=c['arms'][index%len(c['arms'])]
                study=treatment_study(c,seed,arm);deadline=deadline_for_job()
                run(study,arm['method'],'cuda',deadline)
                # Checkpoints are predeclared; evaluation never promotes an arm.
                for epoch in c['pretraining']['checkpoint_epochs']:
                    if time.time()>deadline-900:raise TimeoutError('Milestone evaluation awaits resumed allocation')
                    path=study.root/f'pretraining/{arm["method"]}/technical/epoch-{epoch:03d}.pt'
                    evaluate_checkpoint_isolated(c,path,'initial' if epoch==0 else 'pretrained',f'{arm["name"]}-epoch{epoch:03d}',
                        root/f'analyses/alignment/{seed}/{arm["name"]}/epoch-{epoch:03d}','cuda',seed)
                if arm['name']=='R1':adapt(c,seed,study,'cuda',deadline)
            else:raise ValueError(stage)
            write_json(state,dict(state='complete',stage=stage,index=index,finished_at=time.time()))
            report(c)
        except BaseException as error:
            write_json(state,dict(state='checkpointed' if isinstance(error,TimeoutError) else 'failed',error=repr(error),traceback=traceback.format_exc()))
            report(c)
            if isinstance(error,TimeoutError) and int(os.environ.get('SLURM_RESTART_COUNT','0'))<2:
                subprocess.run(['scontrol','requeue',os.environ['SLURM_JOB_ID']],check=True)
            raise


def submit(path):
    from src.training_methods.shared_pretraining.queue import snapshot
    c=load(path);root=resolve_path(c['output']);technical=root/'technical';technical.mkdir(parents=True,exist_ok=True)
    receipt=technical/'launch.json'
    if receipt.exists():raise FileExistsError('Queue already submitted; use its receipts for resume')
    code=snapshot(technical);frozen=code/Path(path).resolve().relative_to(Path.cwd())
    environment=['PCM_PROJECT_ROOT='+str(code),'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
        'TORCHINDUCTOR_COMPILE_THREADS=4','PYTORCH_ALLOC_CONF=expandable_segments:True','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1']
    records=[]
    def job(name,module,args,*,gpu,hours,cpus,memory,dependency=None,array=None,node=None):
        script=technical/f'{name}.sbatch'
        lines=['#!/bin/bash',f'#SBATCH --job-name={name}',f'#SBATCH --partition={"RTX6000PRO" if gpu else "CPU"}',
            f'#SBATCH --cpus-per-task={cpus}',f'#SBATCH --mem={memory}',f'#SBATCH --time={hours}:00:00',
            f'#SBATCH --output={technical}/{name}-%A_%a.log',f'#SBATCH --chdir={code}','#SBATCH --requeue']
        if gpu:lines.append('#SBATCH --gres=gpu:1')
        if node:lines.append('#SBATCH --nodelist='+node)
        if array:lines.append('#SBATCH --array='+array)
        if dependency:lines.append('#SBATCH --dependency=afterok:'+dependency)
        command=[sys.executable,'-u','-m',module]+args
        lines+=['set -euo pipefail','exec env '+shlex.join(environment)+' '+shlex.join(command),'']
        script.write_text('\n'.join(lines))
        ident=subprocess.check_output(['sbatch','--parsable',str(script)],text=True).strip()
        records.append(dict(stage=name,job=ident,script=str(script),dependency=dependency,array=array))
        write_json(receipt,dict(state='submitting',code=str(code),config=str(frozen),jobs=records))
        return ident
    check_job=job('mechanisms-check','src.research.encoder_mechanisms.workflow',['check','--config',str(frozen)],
        gpu=True,hours=1,cpus=4,memory='24G',node='node58')
    controls=job('mechanisms-controls','src.research.encoder_mechanisms.workflow',['worker','--stage','controls','--config',str(frozen)],
        gpu=True,hours=8,cpus=4,memory='32G',dependency=check_job,node='node58')
    job('mechanisms-alignment','src.research.encoder_mechanisms.workflow',['worker','--stage','alignment','--config',str(frozen)],
        gpu=True,hours=24,cpus=4,memory='32G',dependency=controls,array='0-8%3')
    job('mechanisms-birth','src.research.encoder_mechanisms.birth',['--config',str(frozen)],
        gpu=False,hours=24,cpus=12,memory='64G',array='0-2%1')
    record=dict(state='submitted',code=str(code),config=str(frozen),jobs=records,submitted_at=time.time(),
        scientific_fits=dict(label_free=9,adaptation_encoder_updates=6,adaptation_head_only=3),
        gated='VAMP and birth-prediction fits require the declared diagnostic/coverage evidence; not submitted')
    write_json(receipt,record);report(c);return record


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['submit','check','worker'])
    p.add_argument('--config',required=True);p.add_argument('--stage',choices=['controls','alignment'])
    p.add_argument('--index',type=int,default=None);a=p.parse_args();c=load(a.config)
    if a.action=='submit':print(json.dumps(submit(a.config),indent=2))
    elif a.action=='check':check(c)
    else:worker(c,a.stage,a.index if a.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))


if __name__=='__main__':main()
