"""Run the matched structural trajectory stability assay."""
import argparse
import json
from pathlib import Path

from .prepare import freeze, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('prepare', 'encode', 'report', 'all'), default='all')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    plan = freeze(config)
    if args.stage in ('prepare', 'all'):
        prepare(plan)
    if args.stage in ('encode', 'all'):
        from .encode import encode
        encode(plan)
    if args.stage in ('report', 'all'):
        from .report import report
        report(plan)


if __name__ == '__main__':
    main()
