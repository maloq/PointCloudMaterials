"""Prepare/run/inspect the expanded relaxed-TDA target release."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run', 'status'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', default='worker0')
    parser.add_argument('--hours', type=float, default=16)
    parser.add_argument('--ranks', type=int, default=32)
    parser.add_argument('--max-tasks', type=int)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        from .plan import prepare
        result = prepare(args.config)['counts']
    elif args.stage == 'run':
        from .worker import run
        result = run(args.config, args.worker, args.hours, args.ranks, args.max_tasks, args.retry_failed)
    else:
        from .worker import status
        result = status(args.config)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
