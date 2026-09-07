"""Serial MACE ablations gated on a completed workflow in an existing allocation."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo


def write_json(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def process_identity(pid):
    path=Path(f'/proc/{pid}/stat')
    if not path.exists():return None
    fields=path.read_text().rsplit(')',1)[1].split()
    return None if fields[0]=='Z' else fields[19]


def predecessor_ready(dependency):
    state=json.loads(Path(dependency['status']).read_text())
    if state['state'] in ('failed','superseded'):
        raise RuntimeError(f'Predecessor did not complete successfully: {state}')
    alive=process_identity(dependency['pid'])==dependency['process_identity']
    if not alive and state['state']!='complete':
        raise RuntimeError(f'Predecessor exited before training and analysis completed: {state}')
    return state['state']=='complete' and not alive


def deadline(plan):
    text=subprocess.check_output(['scontrol','show','job','-o',str(plan['allocation'])],text=True)
    fields=dict(item.split('=',1) for item in text.split() if '=' in item)
    if fields['JobState']!='RUNNING':raise RuntimeError(f'Allocation {plan["allocation"]} is {fields["JobState"]}')
    end=datetime.fromisoformat(fields['EndTime']).replace(tzinfo=ZoneInfo(plan['timezone']))
    return end.timestamp()-plan['allocation_reserve_seconds']


def stop_child(child):
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait(timeout=30)


def run_command(command, log_path, end):
    with log_path.open('a',buffering=1) as log:
        log.write(json.dumps(command)+'\n')
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                if time.time()>=end:raise TimeoutError(f'Allocation safety deadline reached during {command}; see {log_path}')
                time.sleep(5)
            if child.returncode:raise RuntimeError(f'Command exited {child.returncode}: {command}; see {log_path}')
        finally:stop_child(child)


def run(plan):
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(state='waiting_for_training_and_analysis',pid=os.getpid(),predecessor=plan['predecessor'],jobs={r['name']:'pending' for r in plan['runs']})
    def update(**values):
        status.update(values);status['updated_at']=datetime.now().astimezone().isoformat()
        write_json(out/'status.json',status)
    def interrupted(signum,frame):raise InterruptedError(f'Queue interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    update()
    try:
        while not predecessor_ready(plan['predecessor']):
            if time.time()>=deadline(plan):raise TimeoutError('Allocation deadline reached while waiting for predecessor analysis')
            time.sleep(30)
        for item in plan['runs']:
            end=deadline(plan)
            if time.time()+plan['estimated_training_seconds']+plan['estimated_probe_seconds']>=end:
                update(state='insufficient_allocation_time',pending_run=item['name']);break
            cfg=json.loads(Path(item['config']).read_text());directory=Path(cfg['output']);directory.mkdir(parents=True,exist_ok=True)
            # Training reads the immutable prepared manifest; no cache rewriting.
            shutil.copy2(Path(plan['prepared_output'])/'data_summary.json',directory/'data_summary.json')
            status['jobs'][item['name']]='training';update(state='training_ablation',current_run=item['name'],deadline=datetime.fromtimestamp(end).astimezone().isoformat())
            run_command([sys.executable,'-m','src.training_methods.pretrained_mace','--config',item['config'],'--stage','train'],directory/'run.log',end)
            summary=json.loads((directory/'training_summary.json').read_text())
            if summary['steps']!=plan['expected_steps'] or summary['partial_epoch']:
                raise RuntimeError(f'{item["name"]} did not receive the matched {plan["expected_steps"]}-step budget: {summary}')
            status['jobs'][item['name']]='probing';update(state='probing',current_run=item['name'])
            run_command([sys.executable,'-m','src.analysis.pretrained_mace_ablation','--plan',plan['_path'],'--run',item['name']],directory/'probe.log',end)
            status['jobs'][item['name']]='trained_and_probed';update()
        # All matched trainings finish before visualization work spends the remaining time.
        for item in plan['runs']:
            if status['jobs'][item['name']]!='trained_and_probed':continue
            end=deadline(plan)
            if time.time()+plan['estimated_analysis_seconds']>=end:
                update(state='analysis_pending_allocation_time',pending_run=item['name']);return
            cfg=json.loads(Path(item['config']).read_text());directory=Path(cfg['output'])
            status['jobs'][item['name']]='static_analysis';update(state='static_analysis',current_run=item['name'])
            run_command([sys.executable,'-m','src.training_methods.pretrained_mace','--config',item['config'],'--stage','analysis'],directory/'analysis.log',end)
            status['jobs'][item['name']]='complete';update()
            run_command([sys.executable,'-m','src.analysis.pretrained_mace_ablation','--plan',plan['_path'],'--collect'],out/'collect.log',end)
            if plan['remove_completed_inference_cache']:
                cache=directory/'static_analysis/analysis_inference_cache.npz'
                write_json(directory/'removed_cache.json',dict(path=str(cache),bytes=cache.stat().st_size,reason='Disposable inference cache removed after reports and comparison; reproduce with the retained encoder and analysis configuration.'))
                cache.unlink()
        finished=all(value=='complete' for value in status['jobs'].values())
        update(state='complete' if finished else 'pending_allocation_time',current_run=None)
    except BaseException:
        update(state='failed',traceback=traceback.format_exc());raise
    finally:lock.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text());plan['_path']=args.plan
    run(plan)


if __name__=='__main__':main()
