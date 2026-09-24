"""Evaluate every retained epoch in a separate process; fail with stage context."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from src.project_runtime.paths import resolve_path
from src.training_methods.shared_pretraining.queue import deadline_for_job
from .reference import write_json
from .report import collect


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());root=resolve_path(cfg['output'])
    deadline=deadline_for_job();status=root/'technical/evaluation-status.json'
    with (root/'technical/evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stage='waiting for classical reference'
        try:
            while not (root/'technical/reference/manifest.json').exists():
                if time.time()>deadline-300:
                    raise TimeoutError('Reference preparation did not finish before allocation deadline.')
                write_json(status,dict(state='waiting',stage=stage));time.sleep(20)
            items=[('archived-epoch34',resolve_path(cfg['reference_checkpoint']),True),
                   ('initial',root/'technical/training/initial.ckpt',False)]
            items += [(f'epoch-{i:03d}',root/f'technical/training/epoch-{i:03d}.ckpt',i in cfg['plot_epochs'])
                      for i in range(cfg['passes'])]
            for name,path,plots in items:
                stage=name
                complete=root/'technical/evaluations'/name/'metrics.json'
                if complete.exists():
                    receipt=json.loads((complete.parent/'provenance.json').read_text())
                    if receipt['checkpoint_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
                        raise ValueError(f'Previously evaluated checkpoint changed: {path}')
                    continue
                while not path.exists() or time.time()-path.stat().st_mtime<10:
                    if time.time()>deadline-300:
                        raise TimeoutError(f'Checkpoint {name} unavailable before allocation deadline.')
                    state_path=root/'technical/training/status.json'
                    if state_path.exists() and json.loads(state_path.read_text())['state']=='failed':
                        raise RuntimeError(f'Training failed before {name}; see {state_path}')
                    write_json(status,dict(state='waiting',stage=name));time.sleep(20)
                write_json(status,dict(state='running',stage=name))
                command=[sys.executable,'-u','-m','src.research.geoframe_evolution.evaluate',
                         '--checkpoint',str(path),'--output',str(root),'--name',name]
                if plots:command.append('--plots')
                with (root/'technical'/f'evaluate-{name}.log').open('a') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                collect(root)
            write_json(status,dict(state='complete',fresh_epochs=cfg['passes'],archived_reference=True))
            collect(root)
        except BaseException as exc:
            write_json(status,dict(state='failed',stage=stage,error=repr(exc),traceback=traceback.format_exc()))
            raise


if __name__=='__main__':
    main()
