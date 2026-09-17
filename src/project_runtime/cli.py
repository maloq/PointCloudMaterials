"""Project setup and explicitly selected storage transfer commands."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .paths import REPO, catalog, dataset_path, load_json, machine, storage_path


def doctor(lammps=False):
    settings = machine()
    report = dict(python=sys.version, executable=sys.executable, roots=settings['roots'],
                  execution=settings['execution'], packages={name: importlib.metadata.version(name)
                  for name in ('numpy', 'omegaconf', 'PyYAML')})
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
    sub.add_parser('paths')
    p = sub.add_parser('datasets')
    p.add_argument('--refresh', action='store_true', help='Build the evidence-backed dataset browser, cards, JSON and CSV.')
    p.add_argument('--output', type=Path, default=Path('docs/datasets'))
    p = sub.add_parser('simulations'); p.add_argument('--output', required=True, type=Path)
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
        result = doctor(args.lammps)
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
    elif args.command == 'resolve':
        result = load_json(args.config)
    elif args.command == 'simulations':
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
