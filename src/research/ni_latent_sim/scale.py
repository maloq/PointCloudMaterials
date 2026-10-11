"""Share an existing frozen Ni fit queue across local and Slurm GPU workers.

Only execution changes. Scientific code/config, identities, checkpoints, batch
sizes and metrics stay those of the original launch. Per-model NFS locks cover
training and all evaluations. A gated checkpoint handoff prevents old fixed
lanes and new dynamic workers from writing the same model concurrently.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, allocation_deadline
from .data import read, root, write_json, sha
from .resilience import process_identity, stage, safe_snapshot

MODULE = 'src.research.ni_latent_sim.scale'


def directory(c):
    return root(c)/'technical/scale-v1'


def profile(c):
    return read(directory(c)/'profile.json')


def verified_group(pid, bundle, expected_start=None):
    start = process_identity(pid)
    if start is None:
        return None
    if expected_start is not None and start != expected_start:
        raise ValueError('PID identity changed during owned-worker handoff')
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    if str(bundle) not in command or 'src.research.ni_latent_sim.workflow' not in command:
        raise ValueError('PID is not the original receipt-owned Ni worker')
    if os.getpgid(pid) != pid:
        raise ValueError('Expected the owned timeout process to lead its group')
    return dict(pid=pid, start=start, command=command)


def stop_group(record):
    if record is None or process_identity(record['pid']) != record['start']:
        return
    # The launch established this exact private process group. Never signal
    # the Slurm allocation, user shell, or jobs found by Unix username.
    os.killpg(record['pid'], signal.SIGTERM)
    os.killpg(record['pid'], signal.SIGCONT)
    end = time.time()+15
    while process_identity(record['pid']) == record['start'] and time.time() < end:
        time.sleep(.2)
    if process_identity(record['pid']) == record['start']:
        os.killpg(record['pid'], signal.SIGKILL)


def handoff(c, original):
    """Stop only receipt-owned lanes after verifying their saved checkpoints."""
    import torch
    out = root(c)/'technical'; dest = directory(c)
    records = []
    for lane in [0, 1]:
        status = read(out/f'supervision-{lane}.json')
        if process_identity(status['supervisor_pid']) != status['supervisor_start']:
            raise ValueError('Original supervisor identity changed')
        supervisor = verified_group(original['pids'][lane], original['bundle'])
        worker = verified_group(status['worker_pid'], original['bundle'])
        current = read(out/f'worker-{lane}.json')
        if current['stage'] != 'training':
            raise ValueError('Handoff expects a training checkpoint; original lane retained')
        folder = root(c)/'analyses'/current['arm']/'technical'
        saved = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
        if saved['update'] < 1:
            raise ValueError('No recoverable scientific checkpoint')
        records.append(dict(lane=lane, supervisor=supervisor, worker=worker,
                            supervisor_pid=status['supervisor_pid'], arm=current['arm'],
                            checkpoint_update=saved['update']))
    write_json(dest/'handoff.json', dict(state='verified', lanes=records, time=time.time()))
    # Pause the supervisors first so intentional termination is not interpreted
    # as an unexpected exit and restarted by the old queue.
    for r in records:
        os.kill(r['supervisor_pid'], signal.SIGSTOP)
    for r in records:
        os.killpg(r['worker']['pid'], signal.SIGSTOP)
        folder = root(c)/'analyses'/r['arm']/'technical'
        saved = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
        progress = read(folder/'progress.json')
        r.update(checkpoint_update=saved['update'], observed_update=progress['update'],
                 checkpoint_sha256=sha(folder/'last.pt'),
                 replayed_updates=max(0, progress['update']-saved['update']))
    write_json(dest/'handoff.json', dict(state='checkpointed', lanes=records, time=time.time()))
    for r in records:
        stop_group(r['worker']); stop_group(r['supervisor'])
    for r in records:
        with (out/f"lane-{r['lane']}.lock").open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(dest/'handoff.json', dict(state='complete', lanes=records, time=time.time(),
                                       allocation_stopped=False, scientific_change=False))


def environment(bundle):
    from src.project_runtime.paths import REPO
    return dict(PCM_PROJECT_ROOT=str(REPO), PYTHONPATH=str(bundle),
                PATH=str(Path(sys.executable).parent)+':/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin',
                CUDA_HOME='/usr/local/cuda', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1', OVITO_THREAD_COUNT='1', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
                UCX_TLS='self,sm,tcp')


def launch(spec):
    campaign = Path(spec['campaign']); original = read(campaign/'technical/launch.json')
    if not original['owner'].startswith('current chat: ni-autonomous-overnight-'):
        raise ValueError('Scale only this chat-owned autonomous campaign')
    if original['host'] != os.uname().nodename or original['allocation'] != os.environ['SLURM_JOB_ID']:
        raise ValueError('Checkpoint handoff must run in the original allocation')
    c = read(Path(original['bundle'])/'config.json'); dest = directory(c)
    if dest.exists():
        raise FileExistsError('Scale revision already exists; inspect its receipt')
    dest.mkdir(parents=True)
    # Copy the actual frozen producer, not the mutable research checkout.
    bundle = ExecutionBundle(dest/'code')
    shutil.copytree(original['bundle'], bundle.root)
    shutil.copy2(__file__, bundle.root/'src/research/ni_latent_sim/scale.py')
    write_json(dest/'profile.json', spec)
    write_json(dest/'implementation.json', dict(original_bundle=original['bundle'],
               orchestrator_sha256=sha(bundle.root/'src/research/ni_latent_sim/scale.py'),
               original_config_sha256=sha(Path(original['bundle'])/'config.json'),
               batch_size=c['training']['batch_size'], microbatch=c['training']['microbatch'],
               scientific_code_unchanged=True))
    receipt = dict(owner=original['owner'], identity=original['identity'], original_launch=str(campaign/'technical/launch.json'),
                   started=time.time(), deadline_unix=original['deadline_unix'], jobs={}, local_pids=[],
                   bundle=str(bundle.root), state='submitting', partitions=spec['partitions'],
                   original_allocation=original['allocation'], original_host=original['host'])
    env = environment(bundle.root)
    queue = SlurmQueue(dest, bundle, MODULE, env, dest/'launch.json', receipt, 'ni-motion')
    minutes = min(spec['job_hours']*60, math.floor((original['deadline_unix']-time.time())/60))
    with queue.submission():
        for index in range(spec['additional_workers']):
            slot = f'extra-{index}'
            queue.submit(slot, ['--gres=gpu:1', '--cpus-per-task=4', '--mem=48G',
                               f'--time={minutes}', '--no-requeue'],
                         partition=spec['partitions'], command_stage='supervise', arguments=['--slot', slot])
    # Accepted remote jobs wait at the handoff gate; no fit can be duplicated.
    handoff(c, original)
    for lane in [0, 1]:
        slot = f'local-{lane}'
        with (dest/f'{slot}.log').open('a') as stream:
            command = ['timeout', '--signal=TERM', '--kill-after=60',
                       f"{int(original['deadline_unix']-time.time())}s", sys.executable, '-u', '-m', MODULE,
                       'supervise', '--config', str(bundle.config_path), '--slot', slot]
            child = subprocess.Popen(command, cwd=bundle.root,
                                     env=dict(os.environ, **env, CUDA_VISIBLE_DEVICES=str(lane)),
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        receipt['local_pids'].append(child.pid)
        write_json(dest/'launch.json', receipt)
    receipt.update(state='running', handoff='complete')
    write_json(dest/'launch.json', receipt)
    write_json(campaign/'technical/active-execution.json', dict(kind='shared_gpu_queue', receipt=str(dest/'launch.json')))
    print(json.dumps(receipt, indent=2), flush=True)


def publish(c):
    """Incremental tables are always available; final export waits for all fits."""
    from . import dense_response
    safe_snapshot(c)
    dest = directory(c)
    with (dest/'report.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        tasks = [dest/'tasks'/(a['name']+'.json') for a in c['arms']]
        if not all(t.exists() for t in tasks):
            return
        states = [read(t)['state'] for t in tasks]
        if all(s == 'complete' for s in states):
            if not (root(c)/'analyses/dense-comparison-v1/technical/complete.json').exists():
                dense_response.report(c)
            write_json(dest/'complete.json', dict(state='complete', models=len(tasks), time=time.time()))


def worker(c, slot):
    import torch
    from . import train, evaluate, dense_response
    torch.set_num_threads(2)
    dest = directory(c); receipt = read(dest/'launch.json'); p = read(root(c)/'technical/plan.json')
    deadline = min(receipt['deadline_unix'], allocation_deadline(reserve_seconds=180))
    worker_path = dest/f'worker-{slot}.json'
    identity = dict(slot=slot, pid=os.getpid(), host=os.uname().nodename,
                    job=os.environ['SLURM_JOB_ID'], started=time.time(), deadline_unix=deadline)
    while not (dest/'handoff.json').exists() or read(dest/'handoff.json')['state'] != 'complete':
        write_json(worker_path, dict(identity, stage='waiting_for_checkpoint_handoff', time=time.time()))
        if time.time() > deadline-600:
            return
        time.sleep(5)
    # Validate sealed inputs before training on another node.
    dense_response.bank(c, p)
    arms = list(c['arms'])
    if slot.startswith('extra-'):
        index = int(slot.split('-')[-1]); long = [a for a in arms if a['train_steps'] == 60]
        arms = long[index:]+long[:index]+[a for a in arms if a['train_steps'] != 60]
    else:
        seed = c['seeds'][int(slot.split('-')[-1])]
        arms.sort(key=lambda a: (a['seed'] != seed, a['train_steps'], a['response']))
    (dest/'tasks').mkdir(exist_ok=True); (dest/'locks').mkdir(exist_ok=True)
    for arm in arms:
        if time.time() > deadline-600:
            break
        terminal = dest/'tasks'/(arm['name']+'.json')
        with (dest/'locks'/(arm['name']+'.lock')).open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if terminal.exists() and read(terminal)['state'] in ['complete', 'failed']:
                continue
            tech = root(c)/'analyses'/arm['name']/'technical'
            label = 'scale-v1/'+arm['name']
            write_json(worker_path, dict(identity, arm=arm['name'], stage='training', time=time.time()))
            trained = stage(c, label, 'training', lambda: train.run(c, p, arm, deadline), deadline)
            if not trained and time.time() > deadline-600:
                write_json(terminal, dict(state='checkpointed', worker=identity, time=time.time()))
                break
            success = trained
            if (tech/'best.pt').exists():
                for name, function in [('rollout', evaluate.run), ('births', evaluate.births),
                                       ('physical_path', dense_response.evaluate)]:
                    write_json(worker_path, dict(identity, arm=arm['name'], stage=name, time=time.time()))
                    ok = stage(c, label, name, lambda function=function: function(c, p, arm), deadline)
                    success = ok and success
            write_json(terminal, dict(state='complete' if success else 'failed', worker=identity, time=time.time()))
            publish(c)
    write_json(worker_path, dict(identity, stage='finished', time=time.time()))
    publish(c)


def supervise(c, slot):
    dest = directory(c); end = min(read(dest/'launch.json')['deadline_unix'], allocation_deadline(reserve_seconds=120))
    with (dest/f'supervisor-{slot}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bundle = Path(__file__).resolve().parents[3]
        for attempt in range(1, 4):
            if time.time() > end-600:
                return
            command = ['timeout', '--signal=TERM', '--kill-after=30', f'{int(end-time.time())}s',
                       sys.executable, '-u', '-m', MODULE, 'worker', '--config', str(bundle/'config.json'), '--slot', slot]
            child = subprocess.Popen(command, cwd=bundle)
            write_json(dest/f'supervisor-{slot}.json', dict(pid=os.getpid(), pid_start=process_identity(os.getpid()),
                       worker_pid=child.pid, attempt=attempt, deadline_unix=end, time=time.time()))
            code = child.wait()
            write_json(dest/f'exit-{slot}-{attempt}.json', dict(exit_code=code, time=time.time()))
            if code == 0:
                return
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['launch', 'supervise', 'worker'])
    parser.add_argument('--config', required=True); parser.add_argument('--slot')
    args = parser.parse_args(); c = read(args.config)
    if args.stage == 'launch':
        launch(c)
    elif args.stage == 'supervise':
        supervise(c, args.slot)
    else:
        try:
            worker(c, args.slot)
        except BaseException:
            write_json(directory(c)/f'failure-{args.slot}-{os.getpid()}.json',
                       dict(error=traceback.format_exc(), time=time.time()))
            raise


if __name__ == '__main__':
    main()
