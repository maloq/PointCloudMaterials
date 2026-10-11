"""Prepare local spatial-training targets with the existing ancestry/shard workflow.

No training or GPU benchmark is launched. Caches and plans retain the exact
input rows, fixed material length normalization and target producer identity.
"""
import argparse
import json
import os


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('selection','plan','prepare','seal'))
    parser.add_argument('--config')
    parser.add_argument('--packed-cache')
    parser.add_argument('--selection-output')
    parser.add_argument('--families',nargs='+',default=['bond_order','cna','tda'])
    parser.add_argument('--lane',type=int)
    args=parser.parse_args()
    if args.stage=='selection':
        if not args.packed_cache or not args.selection_output:parser.error('selection requires --packed-cache and --selection-output')
        from .descriptors.selection import freeze
        result=freeze(args.packed_cache,args.selection_output,args.families)
        print(json.dumps(dict(identity=result['identity'],columns=len(result['columns']))),flush=True)
        return
    if not args.config:parser.error('plan/prepare/seal require --config')
    from src.research.liquid_predictability.data import config
    from src.research.liquid_predictability.rich_multimaterial_data import plan,prepare,seal
    c=config(args.config)
    if args.stage=='plan':
        p=plan(c)
        print(json.dumps(dict(identity=p['identity'],counts=p['counts'],columns=len(p['columns']),
                              descriptors=p['descriptors'])),flush=True)
    elif args.stage=='prepare':
        lane=args.lane if args.lane is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
        prepare(c,lane)
    else:
        seal(c)


if __name__=='__main__':main()
