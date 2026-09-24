"""Frozen parameter-search workers using native fit and fixed assay producers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from src.project_runtime.paths import resolve_path
from src.research.encoder_screen.common import write,sha,load_config
from src.training_methods.shared_pretraining.queue import snapshot,deadline_for_job


def read(path):
    c=json.loads(Path(path).read_text())
    if c['protocol']!='matched_encoder_parameters_v1': raise ValueError('Wrong campaign protocol')
    for key in ('output','screen_config'):
        c[key]=str(resolve_path(c[key]).resolve())
    return c


def task_for(config,item,milestone):
    root=Path(config['output']);code=Path(__file__).resolve().parents[3]
    screen=load_config(config['screen_config'])
    kind='geoframe' if item['family']=='geoframe' else 'geometry'
    template=next(t for t in screen['tasks'] if t['kind']==kind)
    task=dict(template)
    task.pop('encoder_sha256',None)
    name=f'{item["name"]}-{milestone:04d}'
    if kind=='geoframe':
        file='initial.ckpt' if milestone==0 else f'epoch-{milestone-1:03d}.ckpt'
        path=root/'technical/fits'/item['name']/'technical/training'/file
        step=milestone*61
    else:
        study=json.loads(Path(item['training_config']).read_text())
        file='initial.pt' if milestone==0 else f'step-{milestone}.pt'
        path=resolve_path(study['output'])/'technical/fits'/item['arm']/file
        step=milestone
    files={str(code/'src'/p.split('/src/',1)[1]):sha(code/'src'/p.split('/src/',1)[1]) for p in template['producer_files']}
    task.update(name=name,checkpoint=str(path.resolve()),checkpoint_sha256=sha(path),
                producer=str(code),producer_files=files,step=step)
    return task


def evaluation_config(config):
    screen=load_config(config['screen_config'])
    screen['output']=config['output']
    return screen


def child(command,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        return subprocess.run([sys.executable,'-u',*command],stdout=stream,stderr=subprocess.STDOUT).returncode


def evaluate(config,item,milestone):
    from src.research.encoder_screen.run import run
    from .assess import supplement
    c=evaluation_config(config);task=task_for(config,item,milestone)
    root=Path(config['output']);folder=root/'technical/evaluations'/task['name']
    if (folder/'complete.json').exists():
        old=json.loads((folder/'complete.json').read_text())
        if old['task']!=task: raise ValueError('Completed evaluation task changed')
    else: run(c,task)
    destination=root/'technical/supplements'/task['name']
    if not (destination/'technical/metrics.json').exists(): supplement(folder,c['reference'],destination)


def worker(config_path,lane):
    c=read(config_path);root=Path(c['output']);tech=root/'technical';deadline=deadline_for_job()
    status=tech/f'lane-{lane}.json'
    for item in c['fits']:
        folder=tech/'tasks'/item['name'];folder.mkdir(parents=True,exist_ok=True)
        with (folder/'task.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: continue
            if (folder/'complete.json').exists() or (folder/'failed.json').exists(): continue
            if time.time()>deadline-1200:
                write(status,dict(state='checkpointed',reason='allocation reserve'));return
            stage='training'
            try:
                write(status,dict(state='running',task=item['name'],stage=stage,pid=os.getpid()))
                if item['family']=='geoframe':
                    command=['-m','src.research.encoder_parameter_search.geoframe_fit','--config',item['training_config'],
                        '--output',str(tech/'fits'/item['name']),'--passes',str(c['geoframe_passes'])]
                else:
                    command=['-m','src.research.encoder_parameter_search.queue','mace-fit','--config',str(config_path),'--name',item['name']]
                rc=child(command,folder/'training.log')
                if rc==75:
                    write(status,dict(state='checkpointed',task=item['name'],stage=stage));return
                if rc: raise RuntimeError(f'Training exit {rc}; see {folder}/training.log')
                for milestone in item['milestones']:
                    if time.time()>deadline-900:
                        write(status,dict(state='checkpointed',task=item['name'],stage='evaluation'));return
                    stage=f'evaluation-{milestone}'
                    write(status,dict(state='running',task=item['name'],stage=stage,pid=os.getpid()))
                    rc=child(['-m','src.research.encoder_parameter_search.queue','evaluate','--config',str(config_path),
                        '--name',item['name'],'--milestone',str(milestone)],folder/f'evaluate-{milestone}.log')
                    if rc: raise RuntimeError(f'Evaluation exit {rc}; see {folder}/evaluate-{milestone}.log')
                    with (tech/'report.lock').open('a') as report_lock:
                        fcntl.flock(report_lock,fcntl.LOCK_EX)
                        rc=child(['-m','src.research.encoder_parameter_search.report','--config',str(config_path)],tech/'report.log')
                        if rc: raise RuntimeError('Report failed; inspect technical/report.log')
                stage='final dense plots'
                write(status,dict(state='running',task=item['name'],stage=stage,pid=os.getpid()))
                rc=child(['-m','src.research.encoder_parameter_search.queue','figures','--config',str(config_path),
                    '--name',item['name'],'--milestone',str(item['milestones'][-1])],folder/'figures.log')
                if rc:raise RuntimeError(f'Plotting exit {rc}; see {folder}/figures.log')
                write(folder/'complete.json',dict(state='complete',item=item))
            except Exception as exc:
                write(folder/'failed.json',dict(state='failed',stage=stage,error=repr(exc),traceback=traceback.format_exc()))
    write(status,dict(state='finished',pid=os.getpid()))
    child(['-m','src.research.encoder_parameter_search.report','--config',str(config_path)],tech/'report.log')


def launch(config_path):
    c=read(config_path);root=Path(c['output']);tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    if (tech/'launch.json').exists():raise FileExistsError('Campaign already launched')
    receipt=json.loads((tech/'preflight.json').read_text())
    if not receipt['passed'] or receipt['campaign_sha256']!=sha(config_path):raise ValueError('Need matching preflight')
    for p,h in receipt['files'].items():
        if sha(p)!=h:raise ValueError(f'Implementation/config changed after preflight: {p}')
    screen=load_config(c['screen_config']);old=Path(screen['output'])
    # Real copies; reuse the already-computed fixed inputs and physical labels.
    for part in ('inputs','dense8-inputs'):
        target=tech/part
        if not target.exists():shutil.copytree(old/'technical'/part,target)
    code=snapshot(tech);relative=Path(config_path).resolve().relative_to(Path.cwd())
    cfg=json.loads((code/relative).read_text())
    # Runtime paths remain in the immutable runtime config, not active recipes.
    for item in cfg['fits']:item['training_config']=str(code/item['training_config'])
    frozen=tech/'campaign.json';write(frozen,cfg)
    env=dict(os.environ,TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_ALLOC_CONF='expandable_segments:True',PCM_PROJECT_ROOT=str(code))
    import torch
    count=torch.cuda.device_count()
    if count!=2:raise ValueError(f'Expected the current two-GPU allocation; got {count}')
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1').split(',')
    launch=dict(job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,code=str(code),config=str(frozen),workers=[],deadline=deadline_for_job())
    write(tech/'launch.json',launch)
    for lane in range(2):
        command=[sys.executable,'-u','-m','src.research.encoder_parameter_search.queue','worker','--config',str(frozen),'--lane',str(lane)]
        log=tech/f'worker-{lane}.log'
        with log.open('a') as stream:
            proc=subprocess.Popen(command,cwd=code,env=dict(env,CUDA_VISIBLE_DEVICES=devices[lane]),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        launch['workers'].append(dict(lane=lane,pid=proc.pid,command=command,gpu=devices[lane],log=str(log)))
        write(tech/'launch.json',launch)
    print(json.dumps(launch,indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['launch','worker','mace-fit','evaluate','figures'])
    p.add_argument('--config',required=True);p.add_argument('--name');p.add_argument('--lane');p.add_argument('--milestone',type=int)
    a=p.parse_args();c=read(a.config)
    if a.stage=='launch':launch(a.config)
    elif a.stage=='worker':worker(a.config,a.lane)
    else:
        item=next(i for i in c['fits'] if i['name']==a.name)
        if a.stage=='evaluate':evaluate(c,item,a.milestone)
        elif a.stage=='figures':
            from src.research.encoder_screen.dense import evaluate as dense
            dense(evaluation_config(c),task_for(c,item,a.milestone))
        else:
            import torch
            from src.research.structural_state.common import Study
            from src.research.structural_state.runtime import train
            torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            s=Study(item['training_config']);s.bind()
            if not train(s,item['arm'],deadline=deadline_for_job()):raise SystemExit(75)


if __name__=='__main__':main()
