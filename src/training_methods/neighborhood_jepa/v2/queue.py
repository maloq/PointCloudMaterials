"""Claim locked scientific tasks across allocated nodes; preserve frozen source and manifests."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,file_hash
from src.training_methods.shared_pretraining.queue import snapshot,deadline_for_job
from .contracts import variants,RequiredViewPlan


def task_list(config,code):
    root = resolve_path(config['output'])
    tasks = [dict(type='train',name=s['name'],spec=s) for s in variants(config)]
    old = [dict(name='vicreg-mace-local',kind='mace',
                checkpoint='output/shared_pretraining/mace-local-20260918/technical/best.pt',
                producer_code='output/shared_pretraining/local-structure-campaign-20260918/technical/code'),
           dict(name='vicreg-gatr-local',kind='gatr',
                checkpoint='output/shared_pretraining/gatr-local-bond-20260918/technical/best.pt',
                producer_code='output/shared_pretraining/gatr-local-bond-campaign-20260918/technical/code'),
           dict(name='vicreg-mace-expanded',kind='mace',
                checkpoint='output/shared_pretraining/mace-expanded-dual-20260919/technical/best.pt',
                producer_code='output/shared_pretraining/mace-dual-optimized-campaign-20260919/technical/code')]
    for campaign,tag,config_name in [('mace-20260920','v1-b256','mace_20260920.json'),
              ('mace-b1024-lr0008-20260920','v1-b1024','mace_b1024_lr0008_20260920.json')]:
        base = Path('output/neighborhood_jepa')/campaign/'technical'
        for p in sorted((base/'runs').glob('*/status.json')):
            status = json.loads(p.read_text())
            if status['state']!='complete' and not (p.parent.name.endswith('-long') and status['state']=='running'): continue
            old.append(dict(name=tag+'-'+p.parent.name,kind='v1',checkpoint=str(p.parent/'best.pt'),
                producer_code=str(base/'code'),training_config=str(base/'code/configs/neighborhood_jepa'/config_name),
                completion_status=str(p.resolve())))
    for item in old:
        item['checkpoint'] = str(Path(item['checkpoint']).resolve())
        item['producer_code'] = str(Path(item['producer_code']).resolve())
        if 'training_config' in item: item['training_config']=str(Path(item['training_config']).resolve())
    # Old VICReg references and cheap controls are measured early; new fits remain first priority.
    probes = [dict(type='probe',name=i['name'],item=i) for i in old]
    for spec in variants(config):
        item = dict(name='v2-'+spec['name'],kind='v2',checkpoint=str(root/'technical/runs'/spec['name']/'best.pt'),producer_code=str(code))
        probes.append(dict(type='probe',name=item['name'],item=item,requires=spec['name']))
    baselines = [dict(type='probe',name=name,item=dict(name=name,kind='baseline')) for name in ('condition-baseline','geometry-baseline')]
    return tasks+baselines+probes


def worker(config_path,lane,mode='all'):
    config = json.loads(Path(config_path).read_text())
    root = resolve_path(config['output'])/'technical'
    tasks = json.loads((root/'tasks.json').read_text())
    deadline = deadline_for_job()
    # All three legacy long runs must finish before reusing this node. No process is killed.
    dependency = config.get('wait_legacy_on_node59',[])
    if os.uname().nodename.split('.')[0]=='node59':
        while time.time()<deadline-300:
            states = [json.loads(Path(p).read_text())['state'] if Path(p).exists() else 'pending' for p in dependency]
            if all(s=='complete' for s in states): break
            if any(s=='failed' for s in states): raise RuntimeError(f'Legacy dependency failed: {states}')
            save_json(root/f'lane-{lane}.json',dict(state='waiting_for_legacy',states=states,pid=os.getpid()))
            time.sleep(30)
    while time.time()<deadline-300:
        pending = False
        for task in tasks:
            if mode!='all' and task['type']!=mode: continue
            folder = root/('runs' if task['type']=='train' else 'crystallization')/task['name']
            folder.mkdir(parents=True,exist_ok=True)
            status = folder/'status.json'
            if task['type']=='probe' and task['item'].get('completion_status'):
                completed = json.loads(Path(task['item']['completion_status']).read_text())
                if completed['state']!='complete':
                    pending = True
                    continue
            if task.get('requires'):
                dependency_status = root/'runs'/task['requires']/'status.json'
                if not dependency_status.exists() or json.loads(dependency_status.read_text())['state']!='complete':
                    pending = True
                    continue
            with (folder/'worker.lock').open('a') as lock:
                try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    pending = True
                    continue
                if status.exists():
                    prior = json.loads(status.read_text())
                    if prior['state']=='complete': continue
                    if prior['state']=='failed':
                        # Preserve failed tasks and continue independent scientific work.
                        continue
                pending = True
                save_json(root/f'lane-{lane}.json',dict(state='running',task=task['name'],pid=os.getpid(),time=time.time()))
                try:
                    if task['type']=='train':
                        from .runtime import run
                        import torch
                        torch._dynamo.reset()
                        done = run(config,task['spec'],deadline)
                    else:
                        from .probe import run
                        done = run(config,task['item'],deadline)
                except Exception as error:
                    save_json(status,dict(state='failed',error=repr(error),traceback=traceback.format_exc()))
                    print(traceback.format_exc(),flush=True)
                    continue
                if not done:
                    save_json(root/f'lane-{lane}.json',dict(state='checkpointed',task=task['name']))
                    return
                import gc,torch
                gc.collect()
                torch.cuda.empty_cache()
        if not pending:
            save_json(root/f'lane-{lane}.json',dict(state='complete'))
            return
        time.sleep(10)
    save_json(root/f'lane-{lane}.json',dict(state='allocation_deadline'))


def coordinate(config_path,lane_offset):
    config = json.loads(Path(config_path).read_text())
    root = resolve_path(config['output'])/'technical'
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(devices)!=2: raise ValueError(f'Expected exactly two allocated GPUs, got {devices}')
    processes = []
    for i,device in enumerate(devices):
        lane = lane_offset+i
        with (root/f'lane-{lane}.log').open('a') as log:
            processes.append(subprocess.Popen([sys.executable,'-u','-m',__package__+'.queue','worker',
                '--config',config_path,'--lane',str(lane)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=subprocess.STDOUT))
    codes = [p.wait() for p in processes]
    if any(codes): raise RuntimeError(f'V2 workers failed: {codes}')


def submit(config_path):
    config = json.loads(Path(config_path).read_text())
    root = resolve_path(config['output'])/'technical'
    if (root/'launch.json').exists(): raise FileExistsError('Already submitted; use recorded frozen source')
    from .probe import prepare_population
    prepare_population(config)
    code = snapshot(root)
    tasks = task_list(config,code)
    save_json(root/'tasks.json',tasks)
    save_json(root/'view-counts.json',{s['name']:len(RequiredViewPlan.from_spec(s).views) for s in variants(config)})
    frozen_config = str(code/config_path)
    env = dict(os.environ,PCM_PROJECT_ROOT=str(code),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
               OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    # Separate node52 batch allocation; the user's explicit node52 request authorizes this submission.
    script = root/'node52.sbatch'
    command = [sys.executable,'-u','-m',__package__+'.queue','coordinate','--config',frozen_config,'--lane-offset','0']
    exports = '\n'.join(f'export {k}={shlex.quote(env[k])}' for k in ('PCM_PROJECT_ROOT','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTORCH_ALLOC_CONF'))
    script.write_text('#!/bin/bash\n#SBATCH --job-name=nj-v2-al\n#SBATCH --partition=L40S\n#SBATCH --nodelist=node52\n#SBATCH --gres=gpu:2\n#SBATCH --cpus-per-task=16\n#SBATCH --mem=96G\n#SBATCH --time=08:00:00\n#SBATCH --output='+str(root/'node52-slurm.log')+'\nset -euo pipefail\n'+exports+'\ncd '+shlex.quote(str(code))+'\n'+shlex.join(command)+'\n')
    submission = subprocess.run(['sbatch','--parsable',str(script)],check=True,capture_output=True,text=True).stdout.strip()
    command59 = ['srun','--jobid=1000818','--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=12','--gres=gpu:2',
        sys.executable,'-u','-m',__package__+'.queue','coordinate','--config',frozen_config,'--lane-offset','2']
    with (root/'node59-coordinator.log').open('a') as log:
        process = subprocess.Popen(command59,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save_json(root/'launch.json',dict(node52_job=submission,node59_allocation='1000818',node59_pid=process.pid,
        code=str(code),config=frozen_config,tasks=len(tasks),submitted_at=time.time()))
    print(json.dumps(dict(node52_job=submission,node59_pid=process.pid,tasks=len(tasks))))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['submit','worker','coordinate'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--lane',type=int,default=0)
    parser.add_argument('--lane-offset',type=int,default=0)
    args = parser.parse_args()
    if args.command=='submit': submit(args.config)
    elif args.command=='coordinate': coordinate(args.config,args.lane_offset)
    else: worker(args.config,args.lane)
