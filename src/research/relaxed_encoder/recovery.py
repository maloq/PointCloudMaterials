"""Restart unconverged pilot cells from archived full-precision coordinates."""
import argparse
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
from src.data.relaxed_targets.worker import verify_archive
from src.training_methods.shared_pretraining.queue import snapshot,deadline_for_job
from .prepare import freeze,produce
from .queue import claim,build,worker


def retry_cells(plan,root,lane,ranks):
    recipe=json.loads((root/'plan.json').read_text());technical=resolve_path(plan['config']['output'])/'technical'
    deadline=deadline_for_job();failed=False
    for item in recipe['cells']:
        task=item['task'];status=root/'cells'/f'{task["id"]}.json'
        if status.exists() and json.loads(status.read_text())['state']=='complete':continue
        if time.time()>deadline-recipe['limits']['frame_timeout_seconds']-300:raise SystemExit(75)
        with claim(technical/'locks'/f'cell-{task["id"]}') as acquired:
            if not acquired:continue
            if status.exists() and json.loads(status.read_text())['state']=='complete':continue
            save_json(status,dict(state='running',lane=lane,task=task))
            try:
                verify_archive(Path(item['restart_dump']).parent)
                recovery=dict(name=recipe['name'],limits=recipe['limits'],restart_dump=item['restart_dump'],restart_sha256=item['restart_sha256'])
                result=produce(plan,task,ranks,recovery=recovery)
                previous=technical/'failures'/f'{task["id"]}.json'
                if previous.exists():
                    if file_hash(previous)!=item['failure_sha256']:raise ValueError('Original failure record changed')
                    dest=root/'resolved-failures'/previous.name;dest.parent.mkdir(exist_ok=True);previous.rename(dest)
                save_json(status,dict(state='complete',lane=lane,force=result['relaxation']['fmax_eV_per_A'],seconds=result['relaxation']['seconds']))
                print(json.dumps(dict(cell=task['id'],state='complete',force=result['relaxation']['fmax_eV_per_A'])),flush=True)
            except Exception as exc:
                failed=True;save_json(status,dict(state='failed',error=repr(exc),traceback=traceback.format_exc()));traceback.print_exc()
    if failed:raise RuntimeError('Some restarted quenches failed; inspect recovery cell receipts')


def rebuild(plan,root):
    technical=resolve_path(plan['config']['output'])/'technical'
    try:
        if list((technical/'failures').glob('*.json')):raise RuntimeError('Unresolved cell failures')
        old=technical/'build-failed.json'
        if old.exists():old.rename(root/'original-build-failed.json')
        save_json(root/'build-status.json',dict(state='running'))
        build(plan)
        if not (technical/'training-ready.json').exists() or not (technical/'assay/ready.json').exists():
            raise RuntimeError('Cache preparation did not finish before allocation deadline')
        save_json(root/'build-status.json',dict(state='complete'))
    except Exception as exc:
        save_json(root/'build-status.json',dict(state='failed',error=repr(exc),traceback=traceback.format_exc()));raise


def submit(config_path,name,allocation):
    config_path=Path(config_path);plan=freeze(json.loads(config_path.read_text()));c=plan['config']
    technical=resolve_path(c['output'])/'technical';root=technical/'restarts'/name
    if root.exists():raise FileExistsError(f'Recovery already exists: {root}')
    cells=[]
    for p in sorted((technical/'failures').glob('*.json')):
        task=json.loads(p.read_text())['task'];archive=resolve_path(c['archive'])/'failures'/task['id'];verify_archive(archive)
        if 'Stopping criterion = max iterations' not in (archive/'log.lammps').read_text():
            raise ValueError(f'Failure needs separate diagnosis: {task["id"]}')
        cells.append(dict(task=task,restart_dump=str(archive/'relaxed.dump'),restart_sha256=file_hash(archive/'relaxed.dump'),failure_sha256=file_hash(p)))
    if not cells:raise ValueError('No failed cells to recover')
    save_json(root/'plan.json',dict(name=name,original_identity=plan['identity'],cells=cells,
        limits=dict(max_iterations=50000,max_evaluations=250000,frame_timeout_seconds=7200),
        protocol='Same FIRE/fixed box/potential/0.01 force tolerance; restart at archived full-precision failed-quench coordinates; FIRE internal state resets.'))
    code=snapshot(root);config_file=code/config_path
    base=[sys.executable,'-u','-m',__package__+'.recovery','--config',str(config_file),'--name',name]
    exports=dict(PCM_PROJECT_ROOT=str(code),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    records=[]
    def batch(stage,partition,cpus,mem,hours,extra,command):
        path=root/f'{stage}.sbatch'
        content=f'#!/bin/bash\n#SBATCH --job-name=relaxed-retry-{stage}\n#SBATCH --partition={partition}\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task={cpus}\n#SBATCH --mem={mem}\n#SBATCH --time={hours}:00:00\n#SBATCH --output={root}/{stage}-%A_%a.log\n'
        content+=''.join('#SBATCH '+x+'\n' for x in extra)
        content+='set -euo pipefail\ncd '+shlex.quote(str(code))+'\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in exports.items())
        path.write_text(content+'exec '+shlex.join(command)+'\n')
        job=subprocess.check_output(['sbatch','--parsable',str(path)],text=True).strip();records.append(dict(stage=stage,job=job));save_json(root/'launches.json',records);return job
    cpu=batch('cpu','CPU',32,'8G',6,['--array=0-5%6'],[*base,'cpu'])
    cache=batch('build','CPU',4,'16G',8,[f'--dependency=afterok:{cpu}'],[*base,'build'])
    batch('gpu','RTX6000PRO,H100',12,'64G',8,[f'--dependency=afterok:{cache}','--gres=gpu:1'],
        [sys.executable,'-u','-m',__package__+'.queue','gpu','--config',str(config_file),'--lane',name+'-batch'])
    command=['srun',f'--jobid={allocation}','--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=12','--gres=gpu:1',*base,'gpu-wait']
    with (root/'h100.log').open('a') as out:
        process=subprocess.Popen(command,cwd=code,env=dict(os.environ,**exports),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    records.append(dict(stage='gpu-existing',allocation=allocation,pid=process.pid));save_json(root/'launches.json',records)
    print(json.dumps(records),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['submit','cpu','build','gpu-wait']);parser.add_argument('--config',required=True);parser.add_argument('--name',required=True);parser.add_argument('--allocation',type=int);args=parser.parse_args()
    if args.stage=='submit':submit(args.config,args.name,args.allocation);return
    plan=freeze(json.loads(Path(args.config).read_text()));root=resolve_path(plan['config']['output'])/'technical/restarts'/args.name
    if args.stage=='cpu':retry_cells(plan,root,os.environ['SLURM_ARRAY_TASK_ID'],32)
    elif args.stage=='build':rebuild(plan,root)
    else:
        deadline=deadline_for_job()
        while time.time()<deadline-1800:
            p=root/'build-status.json'
            if p.exists():
                state=json.loads(p.read_text())['state']
                if state=='failed':raise RuntimeError('Recovery cache build failed')
                if state=='complete':worker(plan,args.name+'-h100',Path(args.config));return
            if any(json.loads(p.read_text())['state']=='failed' for p in (root/'cells').glob('*.json')):
                raise RuntimeError('Cell recovery failed; GPU worker stopped')
            time.sleep(20)
        raise SystemExit(75)


if __name__=='__main__':
    code=0
    try:main()
    except SystemExit as exc:code=exc.code
    except BaseException:traceback.print_exc();code=1
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
