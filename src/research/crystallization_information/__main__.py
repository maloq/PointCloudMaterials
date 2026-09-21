"""CPU-friendly diagnostic runner using previously extracted frozen features."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json
from .data import prepare
from .runtime import run,tasks
from .report import report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--stage',choices=['all','prepare','fit','report'],default='all');parser.add_argument('--lane',type=int)
    args=parser.parse_args();config=json.loads(resolve_path(args.config).read_text());root=resolve_path(config['output'])
    if args.stage in ('all','prepare'):prepare(config)
    if args.stage in ('all','fit'):
        save_json(root/'technical/tasks.json',tasks(config))
        if args.lane is not None:run(config,args.lane,config['lanes'])
        else:
            with ProcessPoolExecutor(max_workers=config['lanes'],mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(run,config,i,config['lanes']) for i in range(config['lanes'])]
                for future in futures:future.result()
        report(config)
    if args.stage=='report':report(config)


if __name__=='__main__':main()
