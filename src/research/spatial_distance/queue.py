"""Prepare or train continuous spatial-distance readouts."""
import argparse
import traceback
from src.data.fixed_cohort.protocol import write_json
from .common import study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','worker'])
    parser.add_argument('--config',required=True)
    args=parser.parse_args();s=study(args.config)
    state=s.technical/f'{args.stage}-state.json'
    write_json(state,dict(state='running',identity=s.identity))
    try:
        if args.stage=='prepare':
            from .data import prepare
            prepare(args.config)
        else:
            from .train import worker
            worker(args.config)
        write_json(state,dict(state='complete',identity=s.identity))
    except BaseException:
        write_json(state,dict(state='failed',identity=s.identity,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
