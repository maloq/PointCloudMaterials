"""Hierarchy protocol with the shared tested optimizer and frozen Slurm queue."""
import argparse
import os
import sys
import torch
from src.research.robust_onset.queue import preflight,submit,worker
from src.research.robust_onset.evaluate import collect
from .common import Study


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['prepare','preflight','submit','worker','collect'])
    p.add_argument('--config',required=True);p.add_argument('--arm');p.add_argument('--device',default='cuda')
    args=p.parse_args();study=Study(args.config)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.action=='prepare':study.prepare()
    elif args.action=='preflight':preflight(study,args.device)
    elif args.action=='submit':submit(study)
    elif args.action=='worker':
        name=args.arm or study.config['arms'][int(os.environ['SLURM_ARRAY_TASK_ID'])]['name']
        return worker(study,name,args.device)
    else:study.bind();collect(study)
    return 0


if __name__=='__main__':sys.exit(main())
