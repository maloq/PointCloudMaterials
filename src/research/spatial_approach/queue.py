"""Stages for the declared spatial approach experiment."""
import argparse
import traceback

from src.data.fixed_cohort.protocol import write_json
from .common import study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'worker', 'collect'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    s = study(args.config)
    state = s.technical/('preparation-state.json' if args.stage=='prepare' else 'state.json')
    try:
        write_json(state, dict(state='running', stage=args.stage, identity=s.identity))
        if args.stage == 'prepare':
            from .data import prepare
            prepare(args.config)
        elif args.stage == 'worker':
            from .train import worker
            worker(args.config)
        else:
            from .evaluate import collect
            collect(s)
    except BaseException:
        write_json(state, dict(state='failed', stage=args.stage, identity=s.identity, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
