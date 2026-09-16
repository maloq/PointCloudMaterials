"""Run the distinct local phase-space encoder protocol."""
import argparse
from pathlib import Path
import traceback

from src.experiment_runner.registry import write_json
from src.project_runtime.paths import load_json


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['inventory','prepare','teacher','verify','train','evaluate',
                                        'data-prepare','data-study','data-smoke'])
    parser.add_argument('--config',default='configs/analysis/mace_velocity.json')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--variant',choices=['coordinates_velocity','coordinates_only'],default='coordinates_velocity')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();config=load_json(args.config)
    try:
        if args.stage=='data-prepare':
            from .data_amount_data import prepare
            prepare(config)
        elif args.stage in ('data-study','data-smoke'):
            from .data_amount import run
            run(config,args,smoke=args.stage=='data-smoke')
        elif args.stage=='inventory':
            from .inventory import inventory
            inventory(config)
        elif args.stage=='prepare':
            from .data import prepare
            prepare(config)
        else:
            from .train import teacher,verify,train,evaluate
            {'teacher':teacher,'verify':verify,'train':train,'evaluate':evaluate}[args.stage](config,args)
    except BaseException as error:
        write_json(Path(config['output'])/'technical'/f'{args.stage}-{args.variant}-failure.json',
                   dict(state='failed',error=repr(error),traceback=traceback.format_exc()))
        if args.stage in ('data-study','data-smoke'):
            root=Path(config['output'])
            if args.stage=='data-smoke': root=root/'technical/smoke'
            write_json(root/'technical/status.json',dict(state='failed',error=repr(error),
                       traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()
