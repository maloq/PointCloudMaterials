"""Join a frozen queue on two already allocated GPUs, with per-device completion gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--lane-offset',type=int,required=True)
    parser.add_argument('--gpu0-wait',required=True)
    args=parser.parse_args()
    root=Path(args.root)
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(devices)!=2: raise ValueError(f'Expected two allocated devices: {devices}')
    def launch(index):
        lane=args.lane_offset+index
        with (root/f'lane-{lane}.log').open('a') as log:
            return subprocess.Popen([sys.executable,'-u','-m','src.training_methods.neighborhood_jepa.v2.queue',
                'worker','--config',args.config,'--lane',str(lane)],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[index]),stdout=log,stderr=subprocess.STDOUT)
    free=launch(1)
    dependency=Path(args.gpu0_wait)
    while True:
        state=json.loads(dependency.read_text())['state']
        if state=='complete': break
        if state=='failed': raise RuntimeError(f'Existing GPU0 fit failed: {dependency}')
        status=root/f'lane-{args.lane_offset}.json'
        temp=status.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(state='waiting_for_legacy',dependency=str(dependency),pid=os.getpid())))
        temp.replace(status)
        time.sleep(30)
    busy=launch(0)
    codes=[free.wait(),busy.wait()]
    if any(codes): raise RuntimeError(f'Allocated queue workers failed: {codes}')


if __name__=='__main__':
    main()
