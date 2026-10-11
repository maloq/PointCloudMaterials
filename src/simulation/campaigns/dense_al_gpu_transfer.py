"""Move unstarted GPU dense sources to other GPU partitions at source boundaries.

This driver is frozen separately and invokes the original campaign's frozen
producers. Its reservation marker makes the old worker exit at its existing
next-source guard, after completing its current MD, conversion and publication.
No signal is sent to an active worker and no scientific input is rewritten.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import traceback

from src.project_runtime.paths import REPO, load_json, machine
from src.project_runtime.transfer import write_json
from src.simulation.campaigns.birth_sources import freeze_code, now, sha256
from src.simulation.campaigns.dense_al import manifest_at, sbatch, verify_code, run_source
from src.simulation.campaigns.dense_al_gpu import complete_handoff


def read(path):
    return json.loads(Path(path).read_text())


def check_marker(transfer, record):
    value = read(Path(record['previous_directory'])/'status.json')
    if value['state'] != 'gpu_transferred' or value['transfer'] != str(transfer):
        raise ValueError(f'Transfer reservation changed: {record["source_id"]}')


def submit(c):
    transfer = Path(c['transfer'])
    prepared = read(transfer/'preparation.json')
    parent = Path(prepared['parent_launch'])
    plan = manifest_at(parent)
    original = read(parent/'submission.json')
    if prepared['source_ids'] != c['source_ids'] or prepared['partitions'] != c['partitions']:
        raise ValueError('Configuration differs from the captured handoff')
    if (transfer/'submission.json').exists():
        raise FileExistsError('This handoff has already been submitted')
    for r in plan['runs']:
        if r['source_id'] in c['source_ids']:
            check_marker(transfer, r)
            directory = Path(plan['root'])/r['run_dir']
            if (directory/'status.json').exists():
                raise ValueError(f'Transferred source already started: {directory}')
            for name, digest in r['input_sha256'].items():
                if sha256(directory/name) != digest:
                    raise ValueError(f'Prepared input changed: {directory/name}')
    write_json(transfer/'config.json', c)
    code = freeze_code(transfer)
    driver = code/'src/simulation/campaigns/dense_al_gpu_transfer.py'
    # Preserve the original physics, monitor and converter even if the working
    # repository changed since the initial campaign was submitted.
    cuda = machine()['execution']['gpu_meam_portable_cuda_root']
    environment = ['PCM_PROJECT_ROOT='+str(REPO), 'PYTHONPATH='+str(parent/'code'),
        'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'OVITO_THREAD_COUNT=1','QT_QPA_PLATFORM=offscreen','CUDA_HOME='+cuda,
        'PATH='+cuda+'/bin:'+str(Path(sys.prefix)/'bin')+':/usr/bin:/bin',
        'MAX_JOBS=4','TORCHINDUCTOR_COMPILE_THREADS=4',
        'UCX_TLS=tcp,self']
    command = [sys.executable,'-u',str(driver)]
    header = ['#!/bin/bash','#SBATCH --nodes=1','#SBATCH --ntasks=1',
              f'#SBATCH --chdir={parent}/code']
    worker = '\n'.join(header+['#SBATCH --job-name=al-dense-GPU-transfer',
        '#SBATCH --partition='+','.join(c['partitions']), '#SBATCH --cpus-per-task=4',
        '#SBATCH --gres=gpu:1','#SBATCH --mem=32G','#SBATCH --time='+c['walltime'],
        f'#SBATCH --array=0-{len(c["source_ids"])-1}', '#SBATCH --hold','#SBATCH --no-requeue',
        '#SBATCH --signal=B:USR1@600',f'#SBATCH --output={transfer}/worker-%A_%a.log',
        'set -euo pipefail','exec env '+shlex.join(environment+command+
            ['worker','--transfer',str(transfer)])+' --index "$SLURM_ARRAY_TASK_ID"',''])
    controller = str(original['cpu_controller'])
    old_collector = str(c.get('replaces_collector', original['reconciliation_job']))
    before = subprocess.check_output(['squeue','-h','-j',controller,'-o','%i %T %E'],text=True).strip()
    if not before.startswith(controller+' PENDING '):
        raise ValueError(f'CPU successor is no longer pending: {before}')
    old_dependencies = before.split(maxsplit=2)[2]
    dependencies = [s.split('(')[0].replace('_*','') for s in old_dependencies.split(',')]
    dependencies.remove('afterok:'+old_collector)
    subprocess.run(['scontrol','hold',controller],check=True)
    job = sbatch(worker,transfer/'workers.sbatch')
    collector_script = '\n'.join(header+['#SBATCH --job-name=al-dense-GPU-collect',
        '#SBATCH --partition=CPU','#SBATCH --cpus-per-task=1','#SBATCH --mem=4G',
        '#SBATCH --time=02:00:00',f'#SBATCH --dependency=afterany:{original["worker_job"]}:{job}',
        f'#SBATCH --output={transfer}/collect-%j.log','set -euo pipefail',
        'exec env '+shlex.join(environment+command+['reconcile','--transfer',str(transfer)]),''])
    collector = sbatch(collector_script,transfer/'reconcile.sbatch')
    dependency = ','.join(dependencies+['afterok:'+collector])
    subprocess.run(['scontrol','update','JobId='+controller,'Dependency='+dependency],check=True)
    receipt = dict(state='submitted',source_ids=c['source_ids'],partitions=c['partitions'],
        worker_job=job,reconciliation_job=collector,old_reconciliation_job=old_collector,
        original_worker_job=original['worker_job'],cpu_controller=controller,
        old_cpu_dependencies=old_dependencies,new_cpu_dependencies=dependency,
        parent_code_inventory_sha256=sha256(parent/'code_sha256.json'),
        source_driver_sha256=sha256(driver),submitted_at=now(),
        rtx_exit_policy='Existing frozen next-source reservation guard exits after current source publication; expected job failure is a queue handoff, not failed MD')
    write_json(transfer/'submission.json',receipt)
    # Only the superseded pending collector is canceled. Running workers finish.
    pending = subprocess.check_output(['squeue','-h','-j',old_collector,'-o','%T'],text=True).strip()
    if pending != 'PENDING':
        raise ValueError(f'Old collector changed: {pending}')
    subprocess.run(['scancel',old_collector],check=True)
    subprocess.run(['scontrol','release',job],check=True)
    subprocess.run(['scontrol','release',controller],check=True)
    return receipt


def verify_driver(transfer):
    inventory = read(transfer/'code_sha256.json')
    relative = 'src/simulation/campaigns/dense_al_gpu_transfer.py'
    if Path(__file__).resolve() != transfer/'code'/relative or sha256(__file__) != inventory[relative]:
        raise ValueError('Use the recorded frozen transfer driver')


def worker(transfer,index):
    import torch
    def interrupted(signum,frame):
        raise InterruptedError(f'Transferred GPU worker received signal {signum}')
    for signum in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
        signal.signal(signum,interrupted)
    verify_driver(transfer)
    c = read(transfer/'config.json')
    parent = Path(read(transfer/'preparation.json')['parent_launch'])
    verify_code(parent)
    plan = manifest_at(parent)
    source_id = c['source_ids'][index]
    record, = [r for r in plan['runs'] if r['source_id']==source_id]
    check_marker(transfer,record)
    directory = Path(plan['root'])/record['run_dir']
    audit = transfer/'technical'/f'source{source_id}'
    audit.mkdir(parents=True,exist_ok=True)
    try:
        device = torch.cuda.get_device_name()
        if not any(name in device for name in c['partitions']):
            raise ValueError(f'GPU outside the requested hardware: {device}')
        major,minor = torch.cuda.get_device_capability()
        arch = f'{major}.{minor}'
        cache = Path(c['compiler_cache'])/str(os.environ['SLURM_JOB_ID'])
        os.environ['TORCH_CUDA_ARCH_LIST'] = arch
        os.environ['TORCH_EXTENSIONS_DIR'] = str(cache/'extensions')
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache/'inductor')
        nvcc = Path(os.environ['CUDA_HOME'])/'bin/nvcc'
        hardware = dict(source_id=source_id,device=device,architecture=arch,
            gpu_count=torch.cuda.device_count(),torch_version=torch.__version__,
            torch_cuda=torch.version.cuda,nvcc=subprocess.check_output([str(nvcc),'--version'],text=True),
            job_id=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,checked_at=now())
        write_json(audit/'hardware.json',hardware)
        write_json(directory/'technical/hardware-transfer.json',dict(
            **hardware,transfer=str(transfer),physics_code=str(parent/'code')))
        check = [sys.executable,'-u','-m','src.research.meam_gpu.npt','validate',
                 '--directory',str(directory),'--output',str(audit/'validation'),'--steps','100']
        with (audit/'validation.log').open('x') as log:
            subprocess.run(check,check=True,stdout=log,stderr=subprocess.STDOUT)
        if read(audit/'validation/validation.json')['state']!='passed':
            raise ValueError('Hardware-specific numerical gate did not pass')
        # Archive the compiled extension and its build recipe. Inductor cache
        # stays disposable; scientific producer hashes are already frozen.
        shutil.copytree(cache/'extensions',audit/'compiled-extension')
        write_json(audit/'compiled-extension-sha256.json',{
            str(p.relative_to(audit/'compiled-extension')):sha256(p)
            for p in (audit/'compiled-extension').rglob('*') if p.is_file()})
        run_source(plan,record)
        complete_handoff(parent,plan,record)
        write_json(audit/'complete.json',dict(state='complete',source_id=source_id,finished_at=now()))
    except BaseException:
        write_json(audit/'failure.json',dict(state='failed',source_id=source_id,
            error=traceback.format_exc(),recorded_at=now()))
        # Leave the transferred marker in place so an old RTX worker cannot
        # retry this source. The final reconciler handles any failed attempt.
        raise


def reconcile(transfer):
    verify_driver(transfer)
    parent = Path(read(transfer/'preparation.json')['parent_launch'])
    verify_code(parent)
    plan = manifest_at(parent)
    c = read(transfer/'config.json')
    normalized = []
    for r in plan['runs']:
        if r['source_id'] not in c['source_ids']:
            continue
        old = Path(r['previous_directory'])
        if old.is_symlink():
            continue
        check_marker(transfer,r)
        # This collector depends on BOTH arrays ending. Restore the original
        # reservation solely for the original frozen failure/publication logic.
        write_json(old/'status.json',dict(state='gpu_reserved',launch=str(parent),
            transfer=str(transfer),created_at=now()))
        normalized.append(r['source_id'])
    write_json(transfer/'technical/reconciliation.json',dict(normalized_source_ids=normalized,
        started_at=now(),physics_reconciler=str(parent/'code')))
    subprocess.run([sys.executable,'-u','-m','src.simulation.campaigns.dense_al_gpu',
                    'reconcile','--launch',str(parent)],check=True)
    write_json(transfer/'completion.json',dict(state='reconciled',finished_at=now(),
        parent_completion=read(parent/'completion.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['submit','worker','reconcile'])
    p.add_argument('--config');p.add_argument('--transfer',type=Path);p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.action=='submit':print(json.dumps(submit(load_json(a.config)),indent=2))
    elif a.action=='worker':worker(a.transfer,a.index)
    else:reconcile(a.transfer)


if __name__=='__main__':main()
