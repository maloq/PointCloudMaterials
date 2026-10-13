"""Thin command dispatcher for the phase-space experiment."""
import argparse
import torch
from .data import read, prepare


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['prepare','benchmark','launch','team','worker','train','evaluate','compare'])
    parser.add_argument('--config',required=True); parser.add_argument('--name')
    parser.add_argument('--lane',type=int,default=0); parser.add_argument('--deadline',type=float,default=float('inf'))
    args=parser.parse_args(); c=read(args.config); torch.set_num_threads(1)
    if args.stage=='prepare': print(prepare(c)['counts'])
    elif args.stage=='benchmark':
        from .learning import benchmark
        benchmark(c)
    else:
        from .campaign import dispatch
        dispatch(c,args)


if __name__=='__main__': main()
