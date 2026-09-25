"""Three matched arms with overnight budgets and automatic checkpoint evaluation."""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from src.project_runtime.paths import resolve_path
from src.training_methods.bcr.runtime import train,load_data
from src.training_methods.shared_pretraining.queue import snapshot,deadline_for_job
from .data import prepare
from .compare import checkpoint,compare


def recipes(config):
    manifest=json.loads((resolve_path(config['data'])/'manifest.json').read_text());base=json.loads(Path('configs/bcr/real_overfit.json').read_text())
    root=resolve_path(config['output']);training=base['training'];training.update(updates=config['updates'],batch_size=config['batch_size'],microbatch=config['microbatch'],seed=config['seed'],
        gate_receipt=str(root/'technical/preflight/technical/gate.json'))
    training['encoder'].update(d0=manifest['d0'],n_ref=manifest['n_ref']);base.update(data=config['data'],output=str(root/'technical/preflight'),training=training)
    base.pop('preparation');base.update(overfit_updates=1000,overfit_level_ids=[3,4])
    path=Path('configs/bcr/pilot_20260921/verification.json');path.write_text(json.dumps(base,indent=2)+'\n')
    for arm in ('bcr','unconditional','frozen_random'):
        cfg=copy.deepcopy(base);cfg['training']['arm']=arm;cfg['output']=str(root/'technical/runs'/arm)
        Path(f'configs/bcr/pilot_20260921/{arm}.json').write_text(json.dumps(cfg,indent=2)+'\n')
    return base


def arm(config,name,device='cuda'):
    root=resolve_path(config['output']);recipe=json.loads(Path(f'configs/bcr/pilot_20260921/{name}.json').read_text());deadline=deadline_for_job()
    if name!='bcr':
        train(recipe['training'],resolve_path(config['data']),root/'technical/runs'/name,device,deadline=deadline);return
    for step in (1000,3000,10000):
        if time.time()>deadline-180:return
        state=root/'technical/runs/bcr/technical/status.json'
        previous=json.loads(state.read_text())['step'] if state.exists() else 0
        if previous<step:train(recipe['training'],resolve_path(config['data']),root/'technical/runs/bcr',device,stop_after=step,deadline=deadline)
        if not checkpoint(root,'bcr',step).exists():return
        for milestone in ((0,1000) if step==1000 else (step,)):
            while not all(checkpoint(root,a,milestone).exists() for a in ('unconditional','frozen_random')):
                if time.time()>deadline-180:return
                for other in ('unconditional','frozen_random'):
                    failure=root/'technical'/f'failed-{other}.json'
                    if failure.exists():raise RuntimeError(f'Comparator failed: {failure}')
                time.sleep(20)
            compare(config,milestone,device)
    (root/'technical/campaign-complete.json').write_text(json.dumps(dict(state='complete',fits=3,checkpoints=[0,1000,3000,10000],held_out_roots=6))+'\n')


def submit(config,allocation=None):
    root=resolve_path(config['output'])/'technical'
    if (root/'launches.json').exists():raise FileExistsError('Pilot already submitted')
    timing=json.loads((root/'profile.json').read_text())
    # The allocation covers fits + checkpoint-wise reconstruction/probes, not
    # merely a short fitting fragment. Fail before submitting if underbudgeted.
    estimate=max(r['ten_thousand_update_hours'] for r in timing)*1.35+2
    hours=max(12,int(estimate)+2)
    if hours>23:raise ValueError(f'Pilot needs {hours}h per lane; revise budget explicitly before submission')
    remaining=None
    if allocation is not None:
        if str(allocation)!=os.environ.get('SLURM_JOB_ID'):
            raise ValueError('Submit from the requested existing allocation so its deadline and GPU access are explicit')
        remaining=(deadline_for_job()-time.time())/3600
        if remaining<estimate+.5:raise ValueError(f'Existing allocation has {remaining:.2f}h; need {estimate+.5:.2f}h including reserve')
        parity=json.loads((root/'h100-parity.json').read_text())
        if not parity['passed']:raise ValueError('Existing H100 needs passing cross-GPU FP32 parity')
    code=snapshot(root)
    partition='H100' if 'H100' in timing[0]['gpu'] else 'RTX6000PRO'
    launches=[]
    for name in ('unconditional','frozen_random','bcr'):
        script=root/f'{name}.sbatch'
        command=[sys.executable,'-u','-m','src.research.bcr_pilot.queue','arm','--config',str(code/'configs/bcr/pilot_20260921/study.json'),'--arm',name,'--device','cuda']
        if name=='bcr' and allocation is not None:
            environment=dict(os.environ,PCM_PROJECT_ROOT=str(code),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                PYTORCH_ALLOC_CONF='expandable_segments:True')
            launch=['srun',f'--jobid={allocation}','--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=6','--gres=gpu:1',*command]
            with (root/'h100-bcr.log').open('a') as log:
                process=subprocess.Popen(launch,cwd=code,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            launches.append(dict(arm=name,job=str(allocation),pid=process.pid,remaining_hours=remaining,partition='H100',existing_allocation=True))
            (root/'launches.json').write_text(json.dumps(launches,indent=2)+'\n')
            continue
        script.write_text(f'''#!/bin/bash
#SBATCH --job-name=bcr-g1-{name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time={hours}:00:00
#SBATCH --output={root}/{name}-%j.log
set -euo pipefail
cd {shlex.quote(str(code))}
export PCM_PROJECT_ROOT={shlex.quote(str(code))}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec {shlex.join(command)}
''')
        jid=subprocess.check_output(['sbatch','--parsable',str(script)],text=True).strip()
        launches.append(dict(arm=name,job=jid,hours=hours,partition=partition));(root/'launches.json').write_text(json.dumps(launches,indent=2)+'\n')
    (root/'budget.json').write_text(json.dumps(dict(measured_profile=timing,conservative_full_lane_hours=estimate,allocation_hours=hours,existing_allocation_remaining_hours=remaining,includes='preparation already complete; training, checkpoint comparisons and frozen probes'),indent=2)+'\n')
    print(json.dumps(launches),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','preflight','submit','arm','compare']);p.add_argument('--config',required=True);p.add_argument('--arm');p.add_argument('--step',type=int);p.add_argument('--device',default='cuda');p.add_argument('--allocation',type=int);a=p.parse_args()
    config=json.loads(resolve_path(a.config).read_text())
    if a.stage=='prepare':prepare(config);recipes(config)
    elif a.stage=='preflight':
        from src.training_methods.bcr.verify import verify
        from .profile import measure
        recipe=json.loads(Path('configs/bcr/pilot_20260921/verification.json').read_text())
        verify(recipe,resolve_path(config['data']),resolve_path(recipe['output']),a.device)
        measure(config,recipe['training'],a.device)
    elif a.stage=='submit':submit(config,a.allocation)
    elif a.stage=='arm':
        try:arm(config,a.arm,a.device)
        except Exception as error:
            (resolve_path(config['output'])/'technical'/f'failed-{a.arm}.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc()),indent=2)+'\n')
            raise
    else:compare(config,a.step,a.device)

if __name__=='__main__':main()
