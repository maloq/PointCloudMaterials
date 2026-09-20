"""Rerun the conditional-information assay for a pinned pair of native encoders."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('all', 'prepare', 'extract', 'probe', 'report'), default='all')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    from .data import require_hardware
    require_hardware(config)
    from .comparison_data import freeze
    plan = freeze(config)
    if args.stage in ('all', 'prepare'):
        from .comparison_data import prepare
        prepare(config)
    if args.stage in ('all', 'extract'):
        worker = Path(__file__).with_name('comparison_extract.py').resolve()
        bootstrap = "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_path(sys.argv.pop(1),run_name='__main__')"
        for architecture, record in plan['checkpoints'].items():
            command = [sys.executable, '-c', bootstrap, record['producer_code'], str(worker),
                '--config', str(config_path), '--architecture', architecture]
            subprocess.run(command, check=True, env=dict(os.environ, PCM_PROJECT_ROOT=str(Path.cwd())))
    if args.stage in ('all', 'probe'):
        from .comparison_probe import run
        run(config)
    if args.stage in ('all', 'report'):
        from .comparison_report import report
        report(config)


if __name__ == '__main__':
    main()
