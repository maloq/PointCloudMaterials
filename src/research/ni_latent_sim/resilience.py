"""Checkpoint recovery and independent stages for the two local dense-path lanes.

Scientific inputs/settings are unchanged. Failures remain visible, failed fits
are not called complete, and valid saved checkpoints can still be evaluated.
"""
import csv
import fcntl
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .data import read, root, write_json, resolve_path, FAMILY
from .artifacts import write_rows, table_path


def process_identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    fields = text[text.rfind(')')+2:].split()
    return None if fields[0] == 'Z' else fields[19]


def lane_busy(directory, lane):
    with (directory/f'lane-{lane}.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def snapshot(c):
    """Publish available results without requiring every model/table to finish."""
    out = root(c); technical = out/'technical'; states = []; metrics = []
    with (technical/'status.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        for index, arm in enumerate(c['arms']):
            folder = out/'analyses'/arm['name']; tech = folder/'technical'
            complete = read(tech/'complete.json') if (tech/'complete.json').exists() else None
            progress = read(tech/'progress.json') if (tech/'progress.json').exists() else {}
            partial = read(tech/'evaluation-training-status.json') if (tech/'evaluation-training-status.json').exists() else {}
            terminal = technical/'supervision'/f'lane-{index%2}'/'terminal.json'
            state = ('complete' if complete else 'partial_training' if partial else
                     'stopped_incomplete' if terminal.exists() else
                     'in_progress' if progress else 'pending')
            entry = dict(model=arm['name'], training_state=state,
                         updates=complete['updates'] if complete else progress.get('update', 0),
                         requested_updates=c['training']['updates'],
                         selected_update=complete['selected_update'] if complete else partial.get('selected_update'),
                         available_tables=[])
            for name in ['free_rollout', 'free_rollout_by_regime', 'dense_physical_path', 'nucleus_prediction']:
                file = table_path(folder, name)
                if not file.exists():
                    continue
                entry['available_tables'].append(name)
                with file.open() as stream:
                    for row in csv.DictReader(stream):
                        metrics.append(dict(row, table=name, training_state=state,
                                            completed_updates=entry['updates'], source_table=str(file)))
            states.append(entry)
        issues = sorted(str(p.relative_to(out)) for p in technical.glob('stage-attempts/**/attempt-*.json')
                        if read(p).get('state') in ['failed', 'budget_reached', 'interrupted'])
        write_json(technical/'live-status.json', dict(time=time.time(), models=states, issue_receipts=issues))
        report = out/'analyses/progress-v1'; report.mkdir(parents=True, exist_ok=True)
        lines = ['# Available experiment results', '',
                 'Updated incrementally. Partial training and missing evaluations are explicit.', '',
                 '| Model | Training | Updates | Available evaluations |',
                 '| --- | --- | --- | --- |']
        lines += [f"| {r['model']} | {r['training_state']} | {r['updates']}/{r['requested_updates']} | {', '.join(r['available_tables']) or 'pending'} |" for r in states]
        lines += ['', f'Failure/interruption receipts: {len(issues)}. See ../../technical/live-status.json.', '']
        temporary = report/'README.md.building'; temporary.write_text('\n'.join(lines)); temporary.replace(report/'README.md')
        if metrics:
            columns = list(dict.fromkeys(k for row in metrics for k in row))
            write_rows(metrics, report, family=FAMILY, name='available_metrics', columns=columns)


def safe_snapshot(c):
    """A dashboard/export failure must never turn successful training into failure."""
    try:
        snapshot(c)
    except Exception:
        error = traceback.format_exc()
        print('Incremental report failed; computation continues:', error, flush=True)
        try:
            write_json(root(c)/'technical'/f'snapshot-error-{os.getpid()}.json', dict(error=error, time=time.time()))
        except OSError:
            pass  # The error is still present in the detached process log.


def stage(c, label, name, function, deadline):
    """Two recorded attempts; an exhausted stage does not cancel sibling work."""
    directory = root(c)/'technical/stage-attempts'/label/name
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(directory.glob('attempt-*.json')):
        record = read(path)
        if record['state'] == 'running':
            record.update(state='interrupted', recovered_at=time.time())
            write_json(path, record)
        records.append(record)
    if any(r['state'] == 'complete' for r in records):
        return True
    for attempt in range(len(records)+1, 3):
        if time.time() > deadline-120:
            return False
        path = directory/f'attempt-{attempt}.json'
        write_json(path, dict(state='running', started=time.time(), pid=os.getpid()))
        try:
            result = function()
            state = 'budget_reached' if result is False else 'complete'
            write_json(path, dict(state=state, ended=time.time(), pid=os.getpid()))
            safe_snapshot(c)
            return state == 'complete'
        except TimeoutError:
            write_json(path, dict(state='budget_reached', ended=time.time(), error=traceback.format_exc()))
            safe_snapshot(c)
            return False
        except Exception:
            error = traceback.format_exc()
            write_json(path, dict(state='failed', ended=time.time(), error=error))
            print('Stage failed; retaining checkpoint and continuing:', label, name, attempt, error, flush=True)
            gc.collect()
            import torch
            try:
                if torch.cuda.is_initialized():torch.cuda.empty_cache()
            except RuntimeError:
                print('CUDA cleanup failed; process supervisor may need to restart this lane', flush=True)
            safe_snapshot(c)
            if attempt < 2 and time.time() < deadline-130:time.sleep(5)
    return False


def wait_predecessor(c, lane, deadline):
    if 'after_output' not in c['runtime']:
        return
    previous = resolve_path(c['runtime']['after_output'])/'technical'
    write_json(root(c)/'technical'/f'worker-{lane}.json',
               dict(stage='waiting_for_previous_gpu_lane', after=str(previous), lane=lane))
    while True:
        terminal = previous/'supervision'/f'lane-{lane}'/'terminal.json'
        completed = previous/f'worker-complete-{lane}.json'
        if (terminal.exists() or completed.exists()) and not lane_busy(previous, lane):
            # A failed predecessor fit does not invalidate the shared teacher
            # bank. Its missing comparison remains marked in partial reports.
            return
        status = previous/f'supervision-{lane}.json'
        watched = read(status) if status.exists() else None
        watched_alive = watched and process_identity(watched['supervisor_pid']) == watched['supervisor_start']
        failed = any((previous/f'{name}-{lane}.json').exists() for name in ['failed-worker', 'budget-stop'])
        if failed and not watched_alive and not lane_busy(previous, lane):
            write_json(root(c)/'technical'/f'predecessor-incomplete-{lane}.json', dict(previous=str(previous)))
            return
        if time.time() > deadline-600:
            raise TimeoutError('Deadline while waiting for preceding GPU lane')
        time.sleep(10)


def finish_reports(c, deadline):
    """Retry publication separately after all numerical stages have completed."""
    from . import dense_response
    out = root(c)/'technical'
    if not all((out/f'worker-complete-{i}.json').exists() for i in [0, 1]):
        return
    with (out/'report.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (root(c)/'analyses/dense-comparison-v1/technical/complete.json').exists():
            stage(c, 'campaign', 'comparison_report', lambda: dense_response.report(c), deadline)
        elif 'architecture_reference' in c and not (root(c)/'analyses/architecture-comparison-v1/technical/complete.json').exists():
            stage(c, 'campaign', 'architecture_report', lambda: dense_response.architecture_report(c), deadline)


def worker(c, lane):
    import torch
    from . import dense_response, train, evaluate
    torch.set_num_threads(2); out = root(c)/'technical'; p = read(out/'plan.json')
    deadline = read(out/'launch.json')['deadline_unix']; success = True
    with (out/f'lane-{lane}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wait_predecessor(c, lane, deadline)
        # Each worker verifies actual packets. Completion-marker presence alone
        # cannot let an incomplete shared bank through a restart barrier.
        def prepare():
            dense_response.collect(c, p, lane, deadline)
            while not all((dense_response.directory(c)/'packets'/(j['key']+'.json')).exists() for j in p['jobs']):
                if time.time() > deadline-600:raise TimeoutError('Dense bank incomplete at deadline')
                dense_response.collect(c, p, lane, deadline)
                time.sleep(5)
            dense_response.bank(c, p)
        if not stage(c, f'lane-{lane}', 'prepare', prepare, deadline):
            success = False
        else:
            for arm in c['arms'][lane::2]:
                if time.time() > deadline-600:
                    success = False
                    break
                name = arm['name']; tech = root(c)/'analyses'/name/'technical'
                write_json(out/f'worker-{lane}.json', dict(stage='training', arm=name, deadline_unix=deadline))
                trained = stage(c, name, 'training', lambda: train.run(c, p, arm, deadline), deadline)
                success = trained and success
                if not (tech/'best.pt').exists():
                    continue
                # Evaluate the best available likelihood-selected checkpoint,
                # even after a training failure; fitted() records partial status.
                for label, function in [('rollout', evaluate.run), ('births', evaluate.births),
                                        ('physical_path', dense_response.evaluate)]:
                    write_json(out/f'worker-{lane}.json', dict(stage=label, arm=name, deadline_unix=deadline))
                    ok = stage(c, name, label, lambda function=function: function(c, p, arm), deadline)
                    success = ok and success
        write_json(out/f'worker-finished-{lane}.json', dict(state='complete' if success else 'partial', time=time.time()))
        if success:write_json(out/f'worker-complete-{lane}.json', dict(state='complete', time=time.time()))
    safe_snapshot(c)
    finish_reports(c, deadline)


def supervise(c, lane, adopt_pid=None):
    """Adopt a receipt-owned worker untouched, then recover unexpected exits."""
    out = root(c)/'technical'; receipt = read(out/'launch.json'); deadline = receipt['deadline_unix']
    directory = out/'supervision'/f'lane-{lane}'; directory.mkdir(parents=True, exist_ok=True)
    with (directory/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if adopt_pid is not None and adopt_pid not in receipt['pids']:
            raise ValueError('Adopted PID is absent from this campaign launch receipt')
        if not receipt['owner'].startswith('current chat: '):
            raise ValueError('Missing current-chat launch ownership')
        pid = adopt_pid; identity = process_identity(pid) if pid else None
        if identity:
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'src.research.ni_latent_sim.workflow' not in command or str(out) not in command:
                raise ValueError('Receipt PID no longer runs this campaign')
        child = None; restarts = 0; last_snapshot = 0
        status = dict(supervisor_pid=os.getpid(), supervisor_start=process_identity(os.getpid()),
                      adopted_pid=adopt_pid, worker_pid=pid, restarts=0, deadline_unix=deadline)
        while time.time() < deadline:
            if child is not None:child.poll()
            alive = identity is not None and process_identity(pid) == identity
            status.update(worker_pid=pid, restarts=restarts, state='watching' if alive else 'recovering', time=time.time())
            write_json(out/f'supervision-{lane}.json', status)
            if time.time()-last_snapshot > 30:
                safe_snapshot(c)
                last_snapshot = time.time()
            if alive:
                time.sleep(10)
                continue
            if lane_busy(out, lane):
                time.sleep(5)
                continue
            finished = out/f'worker-finished-{lane}.json'; complete = out/f'worker-complete-{lane}.json'
            if finished.exists() or complete.exists():
                state = read(finished)['state'] if finished.exists() else 'complete'
                write_json(directory/'terminal.json', dict(state=state, time=time.time(), restarts=restarts))
                safe_snapshot(c)
                finish_reports(c, deadline)
                return
            if restarts >= 2 or time.time() > deadline-600:
                break
            restarts += 1
            # Preserve failures rather than letting stale flags cancel a peer.
            archive = directory/f'recovery-{restarts}'; archive.mkdir(exist_ok=True)
            for name in [f'failed-worker-{lane}.json', f'budget-stop-{lane}.json']:
                file = out/name
                if file.exists():file.replace(archive/name)
            bundle = Path(__file__).resolve().parents[3]
            command = ['timeout', '--signal=TERM', '--kill-after=60', f'{int(deadline-time.time())}s',
                       sys.executable, '-u', '-m', 'src.research.ni_latent_sim.workflow', 'worker',
                       '--config', str(bundle/'config.json'), '--lane', str(lane)]
            with (out/f'worker-{lane}.log').open('a') as log:
                child = subprocess.Popen(command, cwd=bundle, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            pid = child.pid; identity = process_identity(pid)
            write_json(archive/'launch.json', dict(pid=pid, bundle=str(bundle), time=time.time(), command=command))
        write_json(directory/'terminal.json', dict(state='partial', time=time.time(), restarts=restarts,
                   reason='Recovery limit or original deadline reached; saved artifacts retained'))
        safe_snapshot(c)
