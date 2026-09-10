"""Shared execution provenance for distinct training and simulation protocols.

Tracking observes a command; it never supplies scientific defaults or resumes it.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from importlib import metadata
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import tarfile
import time
import traceback

from .registry import sha256, write_json


@contextmanager
def tracked_run(output: Path, *, kind: str, configs: list[Path], command: list[str],
                question: str = '', success_state: str = 'command_succeeded'):
    """Record an immutable attempt; preserve all previous attempts on explicit retries."""
    repo = Path(__file__).resolve().parents[2]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tracking = output / 'tracking'
    tracking.mkdir(exist_ok=True)
    with (tracking / 'writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another tracked command owns {output}') from error
        now = datetime.now(timezone.utc)
        attempt = tracking / now.strftime('%Y%m%dT%H%M%S.%fZ')
        attempt.mkdir()
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        diff = subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=repo)
        (attempt / 'working_tree.patch').write_bytes(diff)
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True)
        (attempt / 'working_tree_status.txt').write_text(status)
        write_json(attempt / 'environment.json', {
            'python': sys.version, 'executable': sys.executable,
            'packages': sorted([{'name': dist.metadata['Name'], 'version': dist.version}
                                for dist in metadata.distributions()], key=lambda d: d['name'].lower())})
        # Include untracked research implementation as well as tracked files.
        source_paths = []
        for base in ('src', 'scripts', 'configs', 'experiments'):
            source_paths.extend(p for p in (repo / base).rglob('*')
                                if p.is_file() and not p.is_symlink()
                                and p.suffix in {'.py', '.sh', '.json', '.yaml', '.yml', '.md'})
        with tarfile.open(attempt / 'source.tar.gz', 'w:gz') as archive:
            for path in sorted(source_paths):
                archive.add(path, arcname=str(path.relative_to(repo)), recursive=False)
        snapshots = []
        for index, source in enumerate(configs):
            source = source.resolve()
            target = attempt / f'config_{index}_{source.name}'
            target.write_bytes(source.read_bytes())
            snapshots.append({'source': str(source), 'snapshot': str(target), 'sha256': sha256(target)})
        record = {'schema_version': 1, 'kind': kind, 'question': question,
                  'output': str(output), 'attempt': str(attempt),
                  'started_at': now.isoformat(), 'state': 'running',
                  'command': command, 'cwd': str(Path.cwd()), 'git_commit': commit,
                  'working_tree_dirty': bool(status),
                  'source_snapshot': str(attempt / 'source.tar.gz'),
                  'source_snapshot_sha256': sha256(attempt / 'source.tar.gz'),
                  'configs': snapshots, 'host': socket.gethostname(), 'pid': os.getpid(),
                  'pid_start_ticks': Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19],
                  'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'python': sys.executable,
                  'conda_env': os.environ.get('CONDA_DEFAULT_ENV')}

        def save():
            write_json(attempt / 'run_record.json', record)
            write_json(output / 'run_record.json', record)

        save()
        try:
            yield record
        except BaseException as error:
            record.update(state='interrupted' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'failed',
                          error=repr(error), traceback=traceback.format_exc(),
                          finished_at=datetime.now(timezone.utc).isoformat())
            save()
            raise
        else:
            record.update(state=success_state, finished_at=datetime.now(timezone.utc).isoformat())
            save()


def wait_for_dependencies(paths, *, until, status_path):
    """Wait for local tracked commands to finish, including their analysis stage."""
    try:
        while True:
            pending = []
            for path in paths:
                record = observed_record(Path(path))
                if record['state'] == 'command_succeeded':
                    continue
                if record['state'] != 'running' or record['observation'] != 'process alive':
                    raise RuntimeError(f'Dependency cannot complete successfully: {path}: {record["state"]}; {record["observation"]}')
                pending.append(path)
            if not pending:
                write_json(status_path, dict(state='dependencies_complete', updated_at=datetime.now(timezone.utc).isoformat()))
                return
            if until is None:
                raise RuntimeError(f'Dependencies have not succeeded: {pending}')
            if time.time() >= until:
                raise TimeoutError(f'Dependency deadline reached: {pending}')
            write_json(status_path, dict(state='waiting_for_dependencies', pending=pending,
                       pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
                       deadline=datetime.fromtimestamp(until, timezone.utc).isoformat()))
            time.sleep(min(30, until-time.time()))
    except BaseException as error:
        write_json(status_path, dict(state='failed', error=repr(error), updated_at=datetime.now(timezone.utc).isoformat()))
        raise


def execute_spec(path: Path, *, wait_for_dependencies_until: str | None = None) -> None:
    """Execute an explicit command list without shell interpolation or hidden resume."""
    spec = json.loads(path.read_text())
    if spec['kind'] not in {'training', 'simulation', 'analysis'}:
        raise ValueError(f'Unsupported run kind in {path}: {spec["kind"]}')
    if spec['completion'] not in {'process_exit', 'submission_only'}:
        raise ValueError(f'Explicit completion must be process_exit or submission_only in {path}')
    until = None
    if wait_for_dependencies_until is not None:
        deadline = datetime.fromisoformat(wait_for_dependencies_until)
        if deadline.tzinfo is None:
            raise ValueError('Dependency deadline must include its UTC offset')
        until = deadline.timestamp()
    output = Path(spec['output']).resolve()
    execution = output / 'execution'
    if (execution / 'run_record.json').exists():
        raise FileExistsError(f'Run already tracked at {output}; use a new output or the protocol-specific explicit resume command.')
    with tracked_run(execution, kind=spec['kind'], configs=[path] + [Path(p) for p in spec['configs']],
                     command=spec['command'], question=spec['question'],
                     success_state='submitted' if spec['completion'] == 'submission_only' else 'command_succeeded'):
        def interrupt_wait(signum, frame):
            raise InterruptedError(f'Dependency wait interrupted by signal {signum}')
        previous_term = signal.signal(signal.SIGTERM, interrupt_wait)
        try:
            wait_for_dependencies(spec['dependencies'], until=until, status_path=output/'queue_status.json')
        finally:
            signal.signal(signal.SIGTERM, previous_term)
        with (output / 'command.log').open('xb') as log:
            child = subprocess.Popen(spec['command'], cwd=spec['cwd'], stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            received_signal = None

            def forward(signum, frame):
                nonlocal received_signal
                received_signal = signum
                if child.poll() is None:
                    os.killpg(child.pid, signum)

            previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                code = child.wait()
                if received_signal is not None:
                    raise InterruptedError(f'Run interrupted by signal {received_signal}; forwarded to command process group')
                if code:
                    raise subprocess.CalledProcessError(code, spec['command'])
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)


def observed_record(path: Path) -> dict:
    record = json.loads(path.read_text())
    record['observation'] = 'recorded state; remote process not queried'
    if record['state'] == 'running' and record['host'] == socket.gethostname():
        stat = Path(f'/proc/{record["pid"]}/stat')
        try:
            fields = stat.read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            record['observation'] = 'interrupted/stale: recorded process no longer exists'
        else:
            record['observation'] = ('process alive' if fields[19] == record['pid_start_ticks'] and fields[0] != 'Z'
                                     else 'interrupted/stale: PID exited or was reused')
    return record
