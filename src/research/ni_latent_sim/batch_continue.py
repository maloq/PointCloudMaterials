"""Run the existing queue with its explicit RTX batch-continuation profile."""
from pathlib import Path
import os
import shutil
import signal
import subprocess
import sys
import time
from . import scale, train, batch_training
from .data import read, root, write_json, sha, digest
from .resilience import process_identity


def launch():
    """Replace only the two local workers identified by the current receipt."""
    import torch
    bundle = Path(__file__).resolve().parents[3]
    c = read(bundle/'config.json'); dest = bundle.parent
    previous = root(c)/'technical/scale-v1'
    original = read(previous/'launch.json')
    plan = read(root(c)/'technical/plan.json')
    if (dest/'launch.json').exists():
        raise FileExistsError('Batch continuation already launched; inspect its receipt')
    if read(dest/'rehearsal.json')['state'] != 'passed':
        raise ValueError('Continuation rehearsal has not passed')
    records = []
    for lane in [0, 1]:
        status = read(previous/f'supervisor-local-{lane}.json')
        worker = read(previous/f'worker-local-{lane}.json')
        supervisor_pid = original['local_pids'][lane]
        if process_identity(status['pid']) != status['pid_start']:
            raise ValueError('Supervisor identity changed')
        groups = []
        for pid in [supervisor_pid, status['worker_pid']]:
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            if (original['bundle'] not in command or 'src.research.ni_latent_sim.scale' not in command
                    or f'--slot local-{lane}' not in command or os.getpgid(pid) != pid):
                raise ValueError('Process is not the receipt-owned local worker group')
            groups.append(dict(pid=pid, start=process_identity(pid), command=command))
        records.append(dict(lane=lane, supervisor=groups[0], worker=groups[1],
                            supervisor_pid=status['pid'], status=worker))
    receipt = dict(owner=original['owner'], original_execution=str(previous/'launch.json'),
                   bundle=str(bundle), deadline_unix=original['deadline_unix'],
                   untouched_remote_jobs=original['jobs'], local_pids=[], lanes=records,
                   policy=read(bundle/'batch-profile.json'), state='prepared', started=time.time())
    write_json(dest/'handoff.json', receipt)
    paused = []; stopped = False
    try:
        for r in records:
            os.kill(r['supervisor_pid'], signal.SIGSTOP)
            paused.append(r['supervisor_pid'])
            os.killpg(r['worker']['pid'], signal.SIGSTOP)
        for r in records:
            worker = read(previous/f"worker-local-{r['lane']}.json")
            arm = next(a for a in c['arms'] if a['name'] == worker['arm'])
            folder = root(c)/'analyses'/arm['name']/'technical'
            old = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
            if old['identity'] != digest([plan['identity'], arm]):
                raise ValueError('Wrong checkpoint for owned-worker handoff')
            archive = dest/'before'/arm['name']; archive.mkdir(parents=True, exist_ok=True)
            for name in ['last.pt', 'best.pt', 'progress.json', 'complete.json', 'wandb.json']:
                if (folder/name).exists():
                    shutil.copy2(folder/name, archive/name)
            progress = read(folder/'progress.json')
            r.update(status=worker, checkpoint_update=old['update'],
                     checkpoint_sha256=sha(archive/'last.pt'), observed_update=progress['update'],
                     replayed_updates=max(0, progress['update']-old['update']))
        write_json(dest/'handoff.json', dict(receipt, state='checkpointed'))
        for r in records:
            scale.stop_group(r['worker']); scale.stop_group(r['supervisor'])
        stopped = True
    finally:
        if not stopped:
            for r in records:
                if process_identity(r['worker']['pid']) == r['worker']['start']:
                    os.killpg(r['worker']['pid'], signal.SIGCONT)
            for pid in paused:
                if process_identity(pid) is not None:
                    os.kill(pid, signal.SIGCONT)
    env = scale.environment(bundle)
    for lane in [0, 1]:
        with (dest/f'local-{lane}.log').open('a') as stream:
            command = ['timeout', '--signal=TERM', '--kill-after=60',
                       f"{int(receipt['deadline_unix']-time.time())}s", sys.executable, '-u', '-m',
                       'src.research.ni_latent_sim.batch_continue', 'supervise',
                       '--config', str(bundle/'config.json'), '--slot', f'local-{lane}']
            child = subprocess.Popen(command, cwd=bundle, env=dict(os.environ, **env, CUDA_VISIBLE_DEVICES=str(lane)),
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        receipt['local_pids'].append(child.pid)
        write_json(dest/'launch.json', dict(receipt, state='starting'))
    write_json(dest/'launch.json', dict(receipt, state='running'))
    write_json(dest/'handoff.json', dict(receipt, state='complete', allocation_stopped=False))
    write_json(root(c)/'technical/active-execution.json', dict(kind='shared_gpu_queue_with_batch_continuation',
               receipt=str(dest/'launch.json'), remote_receipt=str(previous/'launch.json')))
    print('Started batch-1024 local continuations:', receipt['local_pids'], flush=True)


def main():
    if sys.argv[1] == 'launch':
        launch()
        return
    policy = read(Path(__file__).resolve().parents[3]/'batch-profile.json')
    original = train.run

    def run(c, p, arm, deadline):
        if arm['name'] in policy['arms']:
            return batch_training.run(c, p, arm, deadline, policy)
        return original(c, p, arm, deadline)

    train.run = run
    scale.MODULE = 'src.research.ni_latent_sim.batch_continue'
    scale.main()


if __name__ == '__main__':
    main()
