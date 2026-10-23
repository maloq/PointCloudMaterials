"""Local integration checks: no cluster storage, training jobs, or online tracking."""

import importlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from omegaconf import OmegaConf

from .paths import REPO


def _command(command, log, *, input=None):
    result = subprocess.run(command, input=input, text=True, capture_output=True, cwd=REPO)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'{shlex.join(command)} failed ({result.returncode}); see {log}\n'
                           + result.stdout + result.stderr)


def _render_plans(plans, output):
    from src.experiment_runner.local import build_command
    from src.experiment_runner.plan import apply_plan_repeats, load_plan
    from src.experiment_runner.slurm import generate_sbatch_script

    commands, scripts = [], []
    for path in plans:
        plan = apply_plan_repeats(load_plan(path))
        for stage in plan.stages:
            for experiment in stage.experiments:
                arguments = dict(experiment=experiment, plan=plan, stage=stage,
                                 run_dir=output / path.stem / stage.name / experiment.name)
                command = build_command(**arguments)
                script = generate_sbatch_script(**arguments, repo_root=REPO, log_dir=output)
                invocation = script.split('\npython ', 1)[1].split('\necho "Finished:', 1)[0]
                invocation = invocation.replace('\\\n', '')
                if shlex.split('python ' + invocation) != ['python', *command[1:]]:
                    raise AssertionError(f'Local/Slurm commands differ: {path}, {experiment.name}')
                commands.append(dict(plan=str(path.relative_to(REPO)), command=command))
                scripts.append(script)
    content = '\n'.join(scripts)
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    (output / 'rendered.sh').write_text(content)
    _command(['bash', '-n'], output / 'bash-syntax.log', input=content)
    return dict(plans=len(plans), commands=len(commands))


def run(output=None):
    """Write a fresh check receipt; propagate failures with their original context."""
    parent = Path(output) if output else REPO / 'output/maintenance/checks'
    parent.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix='run-', dir=parent)).resolve()
    report = dict(output=str(destination), python=sys.version, checks={})
    try:
        _command([sys.executable, '-m', 'ruff', 'check', 'src', 'scripts'],
                 destination / 'ruff.log')
        report['checks']['lint'] = 'passed'

        sources = sorted(path for root in ('src', 'scripts')
                         for path in (REPO / root).rglob('*.py'))
        for path in sources:
            compile(path.read_text(), str(path), 'exec')
        report['checks']['python_syntax'] = len(sources)

        counts, plans = dict(json=0, yaml=0), []
        for root in ('configs', 'experiments'):
            for path in sorted((REPO / root).rglob('*')):
                if path.suffix == '.json':
                    json.loads(path.read_text())
                    counts['json'] += 1
                elif path.suffix in ('.yaml', '.yml'):
                    config = OmegaConf.load(path)  # Hydra resolvers belong to the real entry point.
                    counts['yaml'] += 1
                    if root == 'experiments' and 'train_script' in config and 'metrics' in config:
                        plans.append(path)
        report['checks']['configuration_syntax'] = counts

        modules = ('src.project_runtime.cli', 'src.experiment_runner.local',
                   'src.experiment_runner.slurm', 'src.experiment_runner.metric_docs',
                   'src.data.conversion.native_qv', 'src.models.encoders.mlp')
        for name in modules:
            importlib.import_module(name)
        report['checks']['imports'] = list(modules)
        report['checks']['command_rendering'] = _render_plans(plans, destination)

        for name, arguments in (('project-help', ['scripts/project.py', '--help']),
                                ('doctor', ['scripts/project.py', 'doctor']),
                                ('runner-help', ['scripts/run_experiments.py', '--help'])):
            _command([sys.executable, *arguments], destination / f'{name}.log')
        report['checks']['cli'] = 'passed'

        from .example import run as reference_experiment
        report['checks']['reference_experiment'] = reference_experiment(destination / 'example')
        report['ok'] = True
    except Exception as error:
        report.update(ok=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (destination / 'checks.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
