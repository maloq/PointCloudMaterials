"""Run a frozen GATr geometric-feature audit in the requested GPU allocation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('all', 'temporal', 'spatial', 'report'), default='all')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.stage == 'report':
        from .report import report
        parent = json.loads((Path(config['parent_audit'])/'technical/plan.json').read_text())
        report(config, parent)
    else:
        from .run import run
        run(config, args.stage)


if __name__ == '__main__':
    main()
