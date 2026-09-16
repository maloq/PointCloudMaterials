"""Frozen local-state comparison, with explicit stages and retained provenance."""

import argparse
import json
from pathlib import Path
import sys
import time

from threadpoolctl import threadpool_limits

from src.experiment_runner.artifacts import result_folders, write_json
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.tracking import tracked_run
from src.project_runtime.paths import load_json
from .data import prepare, progress
from .compare import fit, evaluate
from .static import discover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['all', 'prepare', 'fit', 'evaluate', 'static',
        'smooth-all', 'smooth-prepare', 'smooth-fit', 'smooth-evaluate',
        'motion-all','motion-prepare','motion-fit','motion-evaluate'], default='all')
    parser.add_argument('--lane',type=int,default=None,help='GPU lane for motion protocol workers')
    args = parser.parse_args(); config = load_json(args.config)
    root = result_folders(config['output'])
    link = Path(config['local_output']); link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(root, target_is_directory=True)
    elif link.resolve() != root.resolve():
        raise ValueError(f'Local output points elsewhere: {link}')
    started = time.monotonic()
    if args.stage.startswith('motion-'):
        from .motion_workflow import run
        run(args,config,root)
        return
    if args.stage.startswith('smooth-'):
        from .smooth_data import prepare as smooth_prepare
        from .smooth import fit as smooth_fit, evaluate as smooth_evaluate
        if config['protocol'] != 'mace_local_smooth_v1':
            raise ValueError('Smooth stages require their distinct local-state protocol')
        with tracked_run(root/f'technical/execution-{args.stage}', kind='research',
            configs=[Path(args.config)], command=[sys.executable, *sys.argv],
            question='Can direct temporal regularization preserve instantaneous local physics with smaller jumps?'):
            snapshot_metric_docs(root, 'mace_local_smooth')
            try:
                for name,function in [('prepare',smooth_prepare),('fit',smooth_fit),('evaluate',smooth_evaluate)]:
                    if args.stage in ['smooth-all','smooth-'+name]: function(config,root)
                write_json(root/'technical/status.json',dict(state='complete',stage=args.stage,
                    elapsed_seconds=time.monotonic()-started))
            except BaseException as error:
                write_json(root/'technical/status.json',dict(state='failed',stage=args.stage,error=repr(error)))
                raise
        return
    with tracked_run(root/f'technical/execution-{args.stage}', kind='research',
        configs=[Path(args.config)], command=[sys.executable, *sys.argv],
        question='Can short-time coordinates or a learned group-physics distance yield informative, smooth local states with uncertainty?'):
        try:
            snapshot_metric_docs(root, 'mace_local_state')
            with threadpool_limits(limits=config['cpu_threads']):
                for name, function in [('prepare', prepare), ('fit', fit), ('evaluate', evaluate), ('static', discover)]:
                    if args.stage in ['all', name]:
                        progress(root, name)
                        function(config, root)
            write_json(root/'technical/status.json', dict(state='complete', stage=args.stage,
                       updated_unix=time.time(), elapsed_seconds=time.monotonic()-started))
        except BaseException as error:
            write_json(root/'technical/status.json', dict(state='failed', stage=args.stage,
                       error=repr(error), updated_unix=time.time(), elapsed_seconds=time.monotonic()-started))
            raise


if __name__ == '__main__':
    main()
