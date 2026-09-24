"""Evaluate native embeddings with fixed classical references and future labels."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from src.experiment_runner.metric_docs import write_metric_table
from src.research.geoframe_evolution.evaluate import frame_metrics
from src.research.geoframe_evolution.metrics import perturbation
from src.research.geoframe_evolution import prediction
from .common import load_config,checked,sha,write


def run(config,task,smoke=False):
    root=Path(config['output']); ref=Path(config['reference'])
    base=root/'technical/evaluations'/task['name'];base.mkdir(parents=True,exist_ok=True)
    destination=base/('smoke' if smoke else 'embeddings')
    record=dict(task,inference_driver_sha256=sha(Path(__file__).with_name('native.py')),inputs=str((root/'technical/inputs').resolve()),destination=str(destination.resolve()))
    write(base/('smoke-task.json' if smoke else 'task.json'),record)
    checked(task['checkpoint'],task['checkpoint_sha256'])
    manifest=json.loads((root/'technical/inputs/manifest.json').read_text())
    checked(ref/'manifest.json',manifest['reference_sha256'])
    for f,h in manifest['files'].items():checked(root/'technical/inputs'/f,h)
    if not (destination/'extraction.json').exists():
        command=[sys.executable,'-u',str(Path(__file__).with_name('native.py').resolve()),'--record',str((base/('smoke-task.json' if smoke else 'task.json')).resolve())]
        if smoke:command.append('--smoke')
        with (base/('smoke.log' if smoke else 'inference.log')).open('w') as log:
            subprocess.run(command,cwd=task['producer'],stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=json.loads((destination/'extraction.json').read_text())
    if receipt['checkpoint_sha256']!=task['checkpoint_sha256']:raise ValueError('Cached embedding identity mismatch')
    if receipt['task_sha256'] != sha(base/('smoke-task.json' if smoke else 'task.json')) or receipt['inputs_sha256'] != sha(root/'technical/inputs/manifest.json'):
        raise ValueError('Cached extraction task/input identity changed; use a new output')
    if smoke:return receipt
    result={}; started=time.monotonic()
    with threadpool_limits(limits=1):
        for rec in manifest['frames']:
            if rec['material'] not in task['materials']:continue
            i=rec['frame_index'];a=np.load(ref/f'frame-{i:02d}.npz');features=np.load(destination/f'frame-{i:02d}.npz')
            for rep in receipt['representations']:
                z=features[rep]
                if z.shape[1]!=128:raise ValueError('Matched baseline requires exactly128 exported coordinates')
                if len(z)!=rec['count']:raise ValueError(f'Wrong reference rows {task["name"]}/{i}')
                metrics,clusters=frame_metrics(z,a,rec)
                metrics['continuity_fixed_candidates_center_fixed']={str(amplitude):perturbation(features[rep+'_control'],features[rep+'_'+str(amplitude)]) for amplitude in (1e-4,.01,.1)}
                result[f'frame_{i:02d}_{rec["material"]}_{rep}']=metrics
                np.save(base/f'clusters-{i:02d}-{rep}.npy',clusters)
        # Identical future target/scaling/splits/readout for every accepted snapshot.
        corpus,pc,_=prediction.prepare()
        if corpus.manifest['identity']!=manifest['future_identity']:raise ValueError('Changed future assay')
        features=np.load(destination/'future.npz')
        for rep in receipt['representations']:
            result['prediction_'+rep]=prediction.evaluate(features[rep],corpus,pc,base,rep)
    write(base/'metrics.json',result)
    write_metric_table(result,root,family='encoder_screen',name=task['name'])
    write(base/'complete.json',dict(state='complete',checkpoint_sha256=task['checkpoint_sha256'],
        reference_sha256=manifest['reference_sha256'],future_identity=manifest['future_identity'],
        feature_files={p.name:sha(p) for p in destination.glob('*.npz')},evaluation_seconds=time.monotonic()-started,
        extraction=receipt,task=task))
    print('complete',task['name'],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--name',required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();c=load_config(a.config);run(c,next(t for t in c['tasks'] if t['name']==a.name),a.smoke)
