"""Project setup and explicitly selected storage transfer commands."""

import argparse
from contextlib import redirect_stdout
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .paths import REPO, catalog, load_json, machine


def _environment_profile(profile):
    """Check the maintained requirements and import the scientific runtime."""
    from packaging.requirements import Requirement

    report = dict(profile=profile, packages={}, imports={}, errors=[])
    files = [REPO / 'environments' / f'requirements-{profile}.txt']
    while files:
        path = files.pop()
        for line in path.read_text().splitlines():
            line = line.split('#', 1)[0].strip()
            if line.startswith('-r '):
                files.append(path.parent / line[3:].strip())
            elif line and not line.startswith('-'):
                requirement = Requirement(line)
                try:
                    version = importlib.metadata.version(requirement.name)
                except importlib.metadata.PackageNotFoundError:
                    version = None
                report['packages'][requirement.name] = version
                if version is None or version not in requirement.specifier:
                    report['errors'].append(f'{requirement}: installed {version or "missing"}')
    modules = ['numpy', 'torch', 'pytorch_lightning', 'mace.calculators', 'e3nn', 'ase.md.langevin']
    if profile == 'gpu':
        modules += ['torchvision', 'cuequivariance_torch', 'openequivariance']
    for name in modules:
        try:
            with redirect_stdout(sys.stderr):
                importlib.import_module(name)
            report['imports'][name] = 'ok'
        except Exception as error:
            report['imports'][name] = f'{type(error).__name__}: {error}'
            report['errors'].append(f'{name} import failed: {error}')
    if report['imports']['torch'] == 'ok':
        import torch

        available = torch.cuda.is_available()
        report['cuda'] = dict(build=torch.version.cuda, available=available)
        if profile == 'gpu' and not available:
            report['errors'].append('GPU profile requires a visible CUDA device; run inside a GPU allocation.')
        if profile == 'cpu' or available:
            try:
                device = 'cuda' if profile == 'gpu' else 'cpu'
                value = torch.ones((2, 2), device=device) @ torch.ones((2, 2), device=device)
                report['tensor_check'] = dict(device=device, sum=value.sum().item())
            except Exception as error:
                report['errors'].append(f'{device} tensor operation failed: {error}')
    report['ok'] = not report['errors']
    return report


def doctor(lammps=False, profile=None):
    settings = machine()
    report = dict(python=sys.version, executable=sys.executable, roots=settings['roots'],
                  execution=settings['execution'], packages={name: importlib.metadata.version(name)
                  for name in ('numpy', 'omegaconf', 'PyYAML')})
    if profile:
        report['environment'] = _environment_profile(profile)
    if lammps:
        executable = settings['execution']['lammps']
        found = shutil.which(executable)
        if found is None:
            raise FileNotFoundError(f'LAMMPS executable unavailable: {executable}; set execution.lammps in your machine file.')
        environment = dict(os.environ, **settings['execution']['mpi_environment'])
        result = subprocess.run([found, '-h'], text=True, capture_output=True, env=environment)
        if result.returncode:
            raise RuntimeError(f'LAMMPS startup failed ({result.returncode}): {found}\n{result.stderr}\nSet required runtime libraries in execution.mpi_environment.')
        required = ('meam', 'eam/alloy')
        missing = [style for style in required if style not in result.stdout.split()]
        if missing:
            raise RuntimeError(f'LAMMPS lacks required pair styles {missing}: {found}')
        report['lammps'] = dict(path=found, required_styles=list(required))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', type=Path, help='Machine YAML; otherwise use PCM_MACHINE_CONFIG, machine.local.yaml or portable defaults.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('doctor'); p.add_argument('--lammps', action='store_true')
    p.add_argument('--profile', choices=('cpu', 'gpu'), help='Check pinned package versions, scientific imports, and CPU/CUDA execution.')
    p = sub.add_parser('check', help='Run lightweight local development integration checks.')
    p.add_argument('--output', type=Path, help='Parent for fresh check receipt directories; defaults to output/maintenance/checks/.')
    p = sub.add_parser('example', help='Run the small CPU preparation/training/resume/evaluation example.')
    p.add_argument('--output', type=Path, required=True, help='New directory for the local example.')
    p.add_argument('--config', type=Path, help='Example recipe; defaults to configs/examples/cpu_pipeline.json.')
    sub.add_parser('paths')
    p = sub.add_parser('datasets')
    p.add_argument('--refresh', action='store_true', help='Build the evidence-backed dataset browser, cards, JSON and CSV.')
    p.add_argument('--output', type=Path, default=Path('docs/datasets'))
    from src.data.relaxation_inventory import add_arguments
    add_arguments(sub.add_parser('relaxations', help='Shared relaxed-cell and neighborhood index.'))
    p = sub.add_parser('simulations'); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--details', action='store_true', help='Measure trajectory schemas, timelines and owned disk sizes.')
    p = sub.add_parser('resolve'); p.add_argument('config', type=Path)
    p = sub.add_parser('snapshot'); p.add_argument('destination', type=Path)
    p = sub.add_parser('bundle'); p.add_argument('--plan', required=True, type=Path); p.add_argument('--destination', required=True, type=Path); p.add_argument('--apply', action='store_true')
    p = sub.add_parser('verify-bundle'); p.add_argument('root', type=Path)
    p = sub.add_parser('publish-simulation'); p.add_argument('source', type=Path); p.add_argument('--id', required=True); p.add_argument('--move', action='store_true')
    p = sub.add_parser('archive-failed-simulation'); p.add_argument('source', type=Path); p.add_argument('--id', required=True); p.add_argument('--inactive', action='store_true', required=True, help='Confirm all writers to this failed run have stopped.')
    args = parser.parse_args(argv)
    if args.machine:
        os.environ['PCM_MACHINE_CONFIG'] = str(args.machine.absolute())
    if args.command == 'doctor':
        result = doctor(args.lammps, args.profile)
    elif args.command == 'check':
        from .checks import run
        result = run(args.output)
    elif args.command == 'example':
        from .example import run
        result = run(args.output, args.config)
    elif args.command == 'paths':
        result = machine()
    elif args.command == 'datasets':
        if args.refresh:
            from .dataset_registry import build_registry
            result = build_registry(args.output)
        else:
            settings = machine()
            result = {key: dict(entry, location=str(Path(settings['roots'][entry['root']])/entry['path']),
                      available=(Path(settings['roots'][entry['root']])/entry['path']).exists())
                      for key, entry in catalog().items()}
    elif args.command == 'relaxations':
        from src.data.relaxation_inventory import dispatch
        result = dispatch(args)
    elif args.command == 'resolve':
        result = load_json(args.config)
    elif args.command == 'simulations':
        if args.details:
            from .simulation_details import export_details
            result = export_details(args.output)
        else:
            from .simulation_inventory import export_simulations
            result = export_simulations(args.output)
    else:
        from .transfer import archive_failed_simulation, bundle, publish_simulation, snapshot, verify_bundle
        if args.command == 'snapshot':
            result = snapshot(args.destination)
            result = {key: value for key, value in result.items() if key != 'files'}
        elif args.command == 'bundle':
            result = bundle(args.plan, args.destination, apply=args.apply)
        elif args.command == 'verify-bundle':
            result = verify_bundle(args.root)
        elif args.command == 'archive-failed-simulation':
            result = archive_failed_simulation(args.source, identifier=args.id)
            result = {key: value for key, value in result.items() if key != 'files'}
        else:
            result = publish_simulation(args.source, identifier=args.id, move=args.move)
            result = {key: value for key, value in result.items() if key != 'files'}
    print(json.dumps(result, indent=2))
    if args.command == 'doctor' and args.profile and not result['environment']['ok']:
        raise SystemExit(1)
