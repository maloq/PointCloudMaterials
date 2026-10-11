"""TDA diagnostics for an immutable snapshot of an unfinished spatial run.

Run this file directly; inference imports only the run's original frozen model.
The training process and original final-checkpoint audit remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def infer(args):
    sys.path.insert(0, str(args.run / 'technical/code'))
    import torch
    from src.research.spatial_architecture.model import Spatial
    from src.models.encoders.esen_runtime import configure
    from src.research.structural_response.model import views

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    technical = args.output / 'technical'
    receipt = read(technical / 'snapshot.json')
    checkpoint = technical / 'checkpoint.pt'
    if sha(checkpoint) != receipt['sha256']:
        raise ValueError('Changed captured checkpoint')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    c, architecture = state['config'], state['architecture']
    if (state['identity'] != receipt['identity'] or state['update'] != receipt['update']
            or receipt['validation']['update'] != state['update']):
        raise ValueError('Checkpoint and validation must describe the same update')
    files = read(args.cache / 'complete.json')['files']
    for name in ('spatial-selection.npz', 'spatial.json'):
        if sha(args.cache / name) != files[name]:
            raise ValueError('Changed fixed spatial panel: ' + name)
    panel = np.load(args.cache / 'spatial-selection.npz')
    weights = np.array(read(args.cache / 'spatial.json')['weights'], dtype=np.float64)
    if panel['positions'].shape != (4096, 80, 3):
        raise ValueError('Expected original validation panel')
    torch.manual_seed(c['seed'])
    model = Spatial(c, architecture).cuda().eval().requires_grad_(False)
    model.load_state_dict(state['model'], strict=True)
    configure(model.geometry, architecture, c['execution_variants'][architecture])
    prediction = np.empty((4096, 2, len(weights)), np.float32)
    start_time = time.monotonic()
    with torch.no_grad():
        for start in range(0, 4096, 256):
            q = torch.from_numpy(panel['positions'][start:start+256]).cuda()
            for v, (x, keep) in enumerate(views(q, 'vicreg_descriptors', c['seed']+start)):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = model.geometry(x, keep)
                prediction[start:start+256, v] = model.descriptors(z.float()).cpu().numpy()
            print(architecture, state['update'], start+len(q), '/4096', flush=True)
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite descriptor prediction')
    mse = ((prediction.astype(np.float64)-panel['target'][:,None].astype(np.float64))**2).mean((0,1))
    reconstructed = float(.5*(mse*weights).sum())
    logged = receipt['validation']['descriptor_half_mse']
    if abs(reconstructed-logged)>2e-4:
        raise ValueError(f'Validation reconstruction differs: {reconstructed} versus {logged}')
    np.savez_compressed(technical/'predictions.npz', prediction=prediction,
                        target=panel['target'], rows=panel['rows'])
    result = dict(architecture=architecture, update=state['update'],
        epochs=state['update']/int(np.ceil(c['data']['training_rows']/c['spatial_training']['batch_size'])),
        planned_epochs=c['spatial_training']['epochs'], identity=state['identity'],
        checkpoint_sha256=receipt['sha256'], prediction_sha256=sha(technical/'predictions.npz'),
        panel_sha256=files['spatial-selection.npz'], metadata_sha256=files['spatial.json'],
        frozen_source=str(args.run/'technical/code'), model_source=str(sys.modules[Spatial.__module__].__file__),
        reconstructed_descriptor_half_mse=reconstructed, logged_descriptor_half_mse=logged,
        seconds=time.monotonic()-start_time, numerical_settings='original BF16 encoder / FP32 head, batch256, original two jitter views',
        local_diagnostic=True, checkpoint_selector='requested progress checkpoint, no validation/AP optimization')
    (technical/'inference.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)


def summarize(args):
    sys.path.insert(0, str(REPO))
    from src.experiment_runner.metric_docs import write_metric_rows
    from src.research.spatial_architecture.descriptor_audit import block_name

    technical = args.output/'technical'
    record = read(technical/'inference.json')
    if sha(technical/'predictions.npz') != record['prediction_sha256']:
        raise ValueError('Changed predictions')
    if sha(args.cache/'spatial.json') != record['metadata_sha256']:
        raise ValueError('Changed target metadata')
    values = np.load(technical/'predictions.npz')
    y, p = values['target'].astype(np.float64), values['prediction'].astype(np.float64)
    meta = read(args.cache/'spatial.json')
    weights = np.array(meta['weights']); active = weights>0
    mse = ((p-y[:,None])**2).mean((0,1)); variance=y.var(0); baseline=(y*y).mean(0)
    common=dict(model=record['architecture'],update=record['update'],epochs=record['epochs'],rows=len(y))
    features, blocks, groups = [], [], {}
    for index, column in enumerate(meta['descriptor_columns']):
        name=block_name(column);groups.setdefault(name,[]).append(index)
        features.append(dict(**common,feature=column['name'],block=name,active=bool(active[index]),
            mse=mse[index],baseline_mse=baseline[index],target_variance=variance[index],
            r2=1-mse[index]/variance[index] if active[index] and variance[index]>1e-12 else None))
    for name, indices in groups.items():
        ids=np.array(indices)[active[indices]]
        blocks.append(dict(**common,block=name,active_outputs=len(ids),mse=float(mse[ids].mean()),
            baseline_mse=float(baseline[ids].mean()),target_variance=float(variance[ids].mean()),
            r2=float(1-mse[ids].sum()/variance[ids].sum()),
            mean_baseline_skill=float(1-mse[ids].sum()/baseline[ids].sum())))
    tda=np.array([c['family']=='tda' for c in meta['descriptor_columns']])&active
    model=dict(**common,descriptor_half_mse=float(.5*(mse*weights).sum()),
        tda_weighted_mse=float((weights[tda]*mse[tda]).sum()/weights[tda].sum()),
        tda_r2=float(1-(weights[tda]*mse[tda]).sum()/(weights[tda]*variance[tda]).sum()),
        tda_mean_baseline_skill=float(1-(weights[tda]*mse[tda]).sum()/(weights[tda]*baseline[tda]).sum()),
        bond_order_mse=float(mse[:40].mean()))
    for name, rows in [('components',features),('blocks',blocks),('models',[model])]:
        write_metric_rows(rows,args.output,family='spatial_tda_progress',name=name)
    print(json.dumps(dict(model=model,blocks=blocks),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['infer','summarize'])
    parser.add_argument('--run',type=Path)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    (infer if args.stage=='infer' else summarize)(args)
