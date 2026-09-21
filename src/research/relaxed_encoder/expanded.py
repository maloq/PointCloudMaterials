"""Submit a frozen paired-data expansion and matched encoder sweep."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json
from src.training_methods.shared_pretraining.queue import snapshot
from .prepare import freeze


def submit(config_path,allocation):
    config_path=Path(config_path);c=json.loads(config_path.read_text());plan=freeze(c)
    root=resolve_path(c['output'])/'technical'
    if (root/'launches.json').exists():raise FileExistsError('Expanded study already submitted')
    code=snapshot(root);config=code/config_path
    env=dict(PCM_PROJECT_ROOT=str(code),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    command=[sys.executable,'-u','-m','src.research.relaxed_encoder.queue'];records=[]
    def job(name,partition,cpus,mem,hours,flags,args):
        script=root/f'{name}.sbatch'
        text=f'#!/bin/bash\n#SBATCH --job-name=relaxed-expanded-{name}\n#SBATCH --partition={partition}\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task={cpus}\n#SBATCH --mem={mem}\n#SBATCH --time={hours}:00:00\n#SBATCH --output={root}/{name}-%A_%a.log\n'
        text+=''.join('#SBATCH '+v+'\n' for v in flags)
        text+='set -euo pipefail\ncd '+shlex.quote(str(code))+'\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())
        text+='exec '+shlex.join([*command,*args,'--config',str(config)])
        if name=='cpu':text+=' --lane "$SLURM_ARRAY_TASK_ID" --ranks 32'
        script.write_text(text+'\n');jid=subprocess.check_output(['sbatch','--parsable',str(script)],text=True).strip()
        records.append(dict(stage=name,job=jid));save_json(root/'launches.json',records);return jid
    cpu=job('cpu','CPU',32,'8G',12,['--array=0-11%12'],['cpu'])
    build=job('build','CPU',4,'32G',12,[],['build'])
    for index in range(3):
        job(f'gpu-{index}','RTX6000PRO,H100',12,'64G',8,[f'--dependency=afterok:{build}','--kill-on-invalid-dep=yes','--gres=gpu:1'],['gpu','--lane',f'expanded-{index}'])
    # Existing allocation starts fits as soon as training caches are ready,
    # while the builder finishes independent held-out descriptor extraction.
    cmd=['srun',f'--jobid={allocation}','--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=12','--gres=gpu:1',*command,'gpu','--config',str(config),'--lane','expanded-h100']
    with (root/'h100.log').open('a') as log:
        process=subprocess.Popen(cmd,cwd=code,env=dict(os.environ,**env),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    records.append(dict(stage='gpu-existing',allocation=allocation,pid=process.pid));save_json(root/'launches.json',records)
    print(json.dumps(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--allocation',type=int,required=True);args=parser.parse_args()
    submit(args.config,args.allocation)
