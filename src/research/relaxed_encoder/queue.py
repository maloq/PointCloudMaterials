"""Locked detached workers. CPU production and GPU fits have explicit dependencies."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from src.data.structural_pretraining.prepare import save_json
from src.project_runtime.paths import resolve_path
from src.training_methods.shared_pretraining.queue import deadline_for_job
from .prepare import freeze,produce,build_training,ARMS,arm_for_name
from .availability import fatal_failures,skipped_cells
from .selection import evaluation_exclusions,selected_runs

@contextmanager
def claim(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:yield False;return
        try:yield True
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def cpu(plan,lane,ranks):
    torch.set_num_threads(1);c=plan['config'];root=resolve_path(c['output'])/'technical';cache=resolve_path(c['cache']);deadline=deadline_for_job()
    for task in plan['tasks']:
        if time.time()>deadline-2500:break
        if (cache/'cells'/task['id']/'complete.json').exists():continue
        with claim(root/'locks'/f'cell-{task["id"]}') as acquired:
            if not acquired:continue
            if (cache/'cells'/task['id']/'complete.json').exists():continue
            if (root/'failures'/f'{task["id"]}.json').exists():continue
            save_json(root/f'cpu-{lane}.json',dict(state='relaxing',task=task,pid=os.getpid()))
            try:
                try:result=produce(plan,task,ranks)
                except RuntimeError:
                    from src.data.relaxed_targets.worker import verify_archive
                    from src.data.structural_pretraining.prepare import file_hash
                    failure=resolve_path(c['archive'])/'failures'/task['id']
                    if 'retry_relaxation' not in c or not (failure/'log.lammps').exists() or 'Stopping criterion = max iterations' not in (failure/'log.lammps').read_text():raise
                    verify_archive(failure)
                    result=produce(plan,task,ranks,recovery=dict(name='extended-budget',limits=c['retry_relaxation'],restart_dump=str(failure/'relaxed.dump'),restart_sha256=file_hash(failure/'relaxed.dump')))
                print(json.dumps(dict(cell=task['id'],seconds=result['relaxation']['seconds'],reused='reused_targets'in result or 'reused_paired'in result)),flush=True)
            except Exception as exc:
                save_json(root/'failures'/f'{task["id"]}.json',dict(task=task,error=repr(exc),traceback=traceback.format_exc()));traceback.print_exc()
                if isinstance(exc,subprocess.TimeoutExpired):skipped_cells(plan)
    save_json(root/f'cpu-{lane}.json',dict(state='finished',missing=sum(not (cache/'cells'/t['id']/'complete.json').exists() for t in plan['tasks'])))


def build(plan):
    from .assay import prepare
    root=resolve_path(plan['config']['output'])/'technical';deadline=deadline_for_job()
    while time.time()<deadline-300:
        if fatal_failures(plan):raise RuntimeError('Cell production failed; see technical/failures before restarting')
        if not (root/'training-ready.json').exists():build_training(plan)
        if prepare(plan):return
        time.sleep(20)


def training_config(plan,arm):
    c=plan['config'];return dict(c,cache=str(resolve_path(c['cache'])/arm),order_cache=str(resolve_path(c['cache'])/arm/'order-cache'),crystallization_plan=c['assay_plan'])


def execute(plan,name,phase):
    if phase=='fit':
        if name in evaluation_exclusions(plan):
            raise ValueError(f'Encoder training explicitly stopped: {name}')
        from src.training_methods.neighborhood_jepa.regularization.specs import variants
        from src.training_methods.neighborhood_jepa.regularization.runtime import run
        c=training_config(plan,arm_for_name(plan['config'],name));spec=next(s for s in variants(c) if s['name']=='sig-direct-raw-order');spec['name']=name
        if 'runs' in c:spec.update(next(r['settings'] for r in c['runs'] if r['name']==name))
        if not run(c,spec,deadline_for_job()):raise SystemExit(75)
    elif phase=='extract':
        from .assay import extract
        extract(plan,name)
    elif phase=='probe':
        from .assay import probes
        probes(plan,name)


def worker(plan,lane,config_path,*,evaluation_only=False):
    root=resolve_path(plan['config']['output'])/'technical';deadline=deadline_for_job()
    arms=[r['name'] for r in selected_runs(plan)] if 'runs' in plan['config'] else ARMS
    tasks=[] if evaluation_only else [('fit',a) for a in arms]
    tasks += [('extract',a) for a in (*arms,'parent_hot','parent_cold')]
    tasks += [('probe',a) for a in (*arms,'parent_hot','parent_cold','geometry_hot','geometry_cold','original_geometry','conditions')]
    while time.time()<deadline-300:
        unfinished=False
        for phase,name in tasks:
            status=root/'queue'/f'{phase}-{name}.json'
            if status.exists() and json.loads(status.read_text())['state'] in ('complete','failed','blocked'):continue
            unfinished=True
            if phase=='fit' and not (root/'training-ready.json').exists():continue
            if phase in ('extract','probe') and not (root/'assay/ready.json').exists():continue
            dependency=root/'queue'/f'{"fit" if phase=="extract" else "extract"}-{name}.json'
            if (phase=='extract' and name in arms) or (phase=='probe' and name in (*arms,'parent_hot','parent_cold')):
                if not dependency.exists():continue
                state=json.loads(dependency.read_text())['state']
                if state in ('failed','blocked'):save_json(status,dict(state='blocked',dependency=str(dependency)));continue
                if state!='complete':continue
            with claim(root/'locks'/f'{phase}-{name}') as acquired:
                if not acquired:continue
                if status.exists() and json.loads(status.read_text())['state']=='complete':continue
                save_json(status,dict(state='running',lane=lane,pid=os.getpid(),phase=phase,name=name));save_json(root/f'gpu-{lane}.json',dict(state='running',phase=phase,name=name))
                log=root/'logs'/f'{phase}-{name}.log';log.parent.mkdir(exist_ok=True)
                with log.open('a') as stream:
                    result=subprocess.run([sys.executable,'-m','src.research.relaxed_encoder.queue','execute','--config',str(config_path),'--name',name,'--phase',phase],stdout=stream,stderr=subprocess.STDOUT)
                save_json(status,dict(state='complete' if result.returncode==0 else 'checkpointed' if result.returncode==75 else 'failed',returncode=result.returncode,log=str(log)))
                if result.returncode==75:return
        if not unfinished:
            from .report import report
            with claim(root/'locks/report') as acquired:
                if acquired:report(plan)
            save_json(root/f'gpu-{lane}.json',dict(state='finished'));return
        if fatal_failures(plan) or (root/'build-failed.json').exists():
            save_json(root/f'gpu-{lane}.json',dict(state='failed',reason='preparation_failed'))
            raise RuntimeError('Preparation failed; GPU queue stopped, see technical/failures')
        save_json(root/f'gpu-{lane}.json',dict(state='waiting_for_dependency'));time.sleep(20)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','cpu','build','gpu','evaluate','execute','report','benchmark']);p.add_argument('--config',required=True);p.add_argument('--lane',default='0');p.add_argument('--ranks',type=int,default=32);p.add_argument('--name');p.add_argument('--phase');a=p.parse_args()
    config_path=resolve_path(a.config);plan=freeze(json.loads(config_path.read_text()));torch.set_num_threads(1)
    if a.stage=='cpu':cpu(plan,a.lane,a.ranks)
    elif a.stage=='build':
        try:build(plan)
        except Exception as e:
            save_json(resolve_path(plan['config']['output'])/'technical/build-failed.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    elif a.stage=='gpu':worker(plan,a.lane,config_path)
    elif a.stage=='evaluate':worker(plan,a.lane,config_path,evaluation_only=True)
    elif a.stage=='execute':execute(plan,a.name,a.phase)
    elif a.stage=='report':
        from .report import report
        report(plan)
    elif a.stage=='benchmark':
        from .benchmark import run
        run(plan,a.ranks)

if __name__=='__main__':
    # Mirror the maintained trainer queue: dataloader shared-tensor resource
    # trackers can otherwise hold a finished subprocess and its GPU indefinitely.
    code=0
    try:main()
    except SystemExit as exc:code=exc.code
    except BaseException:traceback.print_exc();code=1
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
