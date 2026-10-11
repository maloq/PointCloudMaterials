"""Use an existing allocation for later queries of a frozen response campaign.

Only work ordering changes. Numerical inputs, replica seeds, batch sizes, parent
locks and durable publication remain those of the immutable scientific collector.
The helper works backward, far ahead of the two coordinator lanes, and stops
starting work15 minutes before its allocation expires. A128-query guard lets
the coordinator reach any partial helper query after its lock has been released.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--first-parent',type=int,required=True)
    parser.add_argument('--stop-parent',type=int,required=True)
    args=parser.parse_args()
    bundle=Path(args.bundle).resolve();os.chdir(bundle);sys.path.insert(0,str(bundle))
    from src.research.response_geometry import collection
    from src.research.response_geometry.common import read,root,bind,write_json,sha
    from src.research.response_geometry.data import prepare
    from src.experiment_runner.execution import allocation_deadline
    import torch

    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1:
        raise ValueError('An existing one-GPU allocation is required')
    c=read(bundle/'config.json');identity=bind(c)['identity'];states=prepare(c)
    gate=read(root(c)/'technical/gates/complete.json')
    if gate['state']!='complete' or gate['identity']!=identity:
        raise ValueError('Scientific gates have not passed')
    if not 0<=args.first_parent<args.stop_parent<=len(states):
        raise ValueError('Invalid helper query interval')
    lane='helper-'+os.environ['SLURM_JOB_ID'];tech=root(c)/'technical/execution-v1'
    tech.mkdir(parents=True,exist_ok=True);receipt=tech/(lane+'.json')
    details=dict(identity=identity,pid=os.getpid(),job=os.environ['SLURM_JOB_ID'],
        node=os.uname().nodename,device=torch.cuda.get_device_name(),lane=lane,bundle=str(bundle),
        first_parent=args.first_parent,stop_parent=args.stop_parent,
        helper_sha256=sha(__file__),scientific_producer_changed=False,started_at=time.time())
    stop_starting=allocation_deadline(reserve_seconds=900)
    details['stop_starting_at']=stop_starting

    class LaterQueries:
        def __len__(self):return len(states)

        def __iter__(self):
            for state in reversed(states[args.first_parent:args.stop_parent]):
                if time.time()>=stop_starting:
                    print('Allocation reserve reached; current query archived, stopping helper.',flush=True)
                    return
                progress=[read(root(c)/'technical'/f'collection-lane{i}.json')for i in [0,1]]
                if any(p['state']!='running'for p in progress):
                    print('Coordinator collection is no longer running; stopping helper.',flush=True)
                    return
                if state['index']-max(p['parent']for p in progress)<128:
                    print('Coordinator approaching helper interval; yielding remaining work.',flush=True)
                    return
                write_json(receipt,dict(details,state='running',next_parent=state['index'],updated_at=time.time()))
                yield state

    def stop(signum,frame):raise TimeoutError(f'Allocation signal {signum}; completed batches are archived')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    with (tech/(lane+'.lock')).open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(receipt,dict(details,state='starting'))
        try:
            # Work scheduling only; reuse the frozen collector and numerical contract.
            collection.prepare=lambda config:LaterQueries()
            collection.collect(c,lane)
            write_json(receipt,dict(details,state='complete',finished_at=time.time()))
        except BaseException:
            write_json(receipt,dict(details,state='failed',finished_at=time.time(),traceback=traceback.format_exc()))
            raise


if __name__=='__main__':main()
