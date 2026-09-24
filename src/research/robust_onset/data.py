"""Reuse verified paired snapshots; no dynamics or relaxation simulation."""
import json
import numpy as np
from src.research.structural_state.data import Corpus, graph_arrays
from src.research.trajectory_stability.spectrum import source_weights
from .common import sha, digest, write_json
from .metrics import perturb_patch


def patches(arrays):
    return [arrays['positions'][a:b] for a,b in zip(arrays['offsets'][:-1], arrays['offsets'][1:], strict=True)]


def prepare(study):
    corpus = Corpus(study)  # includes all parent-file checksums and ancestry checks
    config = study.config
    folder = study.augmented
    signature = dict(parent=sha(study.cache/'manifest.json'), seed=config['seed'],
        fractions=config['noise']['training_rms_fractions'], producer=sha(__file__),
        metrics_producer=sha(__file__.replace('data.py','metrics.py')), cutoff=config['encoder']['cutoff'])
    if (folder/'manifest.json').exists():
        receipt = json.loads((folder/'manifest.json').read_text())
        if receipt['signature'] != signature:
            raise ValueError('Augmentation recipe changed; use a fresh cache')
        for f,h in receipt['files'].items():
            if sha(folder/f) != h: raise ValueError(f'Changed augmentation: {f}')
        return receipt
    folder.mkdir(parents=True, exist_ok=True)
    # Only fitting observations are augmented for optimization. Tuning and
    # development graphs are clean placeholders and never sampled by training.
    with np.load(study.cache/'observed-graphs.npz') as a: clean = patches(dict(a))
    files = {}; rng = np.random.default_rng(config['seed']+411)
    for view, fraction in enumerate(config['noise']['training_rms_fractions']):
        changed = list(clean); noise = []
        for i in corpus.split['fit']:
            changed[i], receipt = perturb_patch(clean[i], fraction, rng)
            noise.append(receipt)
        filename = f'view-{view}.npz'
        np.savez(folder/filename, **graph_arrays(changed, config['encoder']['cutoff']))
        files[filename] = sha(folder/filename)
        write_json(folder/f'view-{view}.json',dict(expected_relative_rms=fraction,
            actual_relative_rms=float(np.sqrt(np.mean([n['input_relative_mse'] for n in noise]))),
            fit_rows=corpus.split['fit'].tolist()))
        files[f'view-{view}.json'] = sha(folder/f'view-{view}.json')
    receipt = dict(state='complete', signature=signature, files=files,
        note='Fixed finite augmentation bank; rebuilt edges/radial/angular features. Center fixed; original candidate support.')
    write_json(folder/'manifest.json',receipt)
    return receipt


def targets(corpus):
    fit = corpus.split['fit']; sources=np.array([r['root'] for r in corpus.records])
    weights=source_weights(sources[fit]); values={}
    raw = dict(corpus.geometry, current=corpus.targets['current_order'],
        future=np.concatenate([corpus.targets[f'future_order_{t}']-corpus.targets['current_order'] for t in (3,9,12)],axis=1))
    scalers={}
    for name,x in raw.items():
        mean=weights@x[fit].astype(float)
        scale=np.sqrt(weights@np.square(x[fit]-mean)).clip(1e-5)
        values[name]=((x-mean)/scale).astype(np.float32)
        scalers[name]=dict(mean=mean.tolist(),scale=scale.tolist())
    ts=np.array([r['temperature_K'] for r in corpus.records]); temperatures=np.unique(ts[fit])
    if not np.isin(ts,temperatures).all():raise ValueError('Unseen temperature in fixed screen')
    conditions=(ts[:,None]==temperatures[None,:]).astype(np.float32)
    risk={s:ix[corpus.targets['at_risk'][ix]] for s,ix in corpus.split.items()}
    for s,ix in risk.items():
        bins=corpus.targets['event_bin'][ix]
        if not ((bins<5).any() and (bins==5).any()):raise ValueError(f'No both-class risk population: {s}')
    return values,scalers,conditions,risk,sources
