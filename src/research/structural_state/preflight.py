"""Bounded production-shape correctness checks, never called by training."""
import copy
import json
import tempfile

import numpy as np
import torch

from src.training_methods.bcr.model import Encoder
from src.training_methods.bcr.data import pack
from .common import write_json, sha, implementation, digest
from .data import Corpus, PairStream
from .runtime import train, setup, load_bank
from .evaluation import controls, nearest_neighbors, hazard_probe


def run(study, device):
    before = implementation()
    study.identity = 'preflight-' + digest(dict(config=study.config, implementation=before))
    receipt = study.technical / 'preflight.json'
    write_json(receipt, dict(passed=False, state='checking'))
    corpus = Corpus(study)
    pairs = PairStream(corpus.records, corpus.split['fit'], corpus.targets['phase'], study.config['seed']).draw(256)
    if len(pairs) != 256:
        raise AssertionError('Wrong production sampling shape')
    model = setup(study.config, corpus, 'observed', device)
    bank = load_bank(study, model, 'observed', device)
    with np.load(study.cache / 'observed-graphs.npz') as arrays:
        offsets, positions = arrays['offsets'], arrays['positions']
        patches = [positions[offsets[i]:offsets[i+1]] for i in pairs[:8]]
    with torch.no_grad():
        expected = Encoder.pooled(model.encoder, pack(patches, device))
        actual = model.encoder.pooled_graph(bank.batch(pairs[:8]))
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-4)
    del model, bank
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix='correctness-', dir=study.technical) as scratch:
        from pathlib import Path
        scratch = Path(scratch)
        for arm in study.config['arms']:
            complete = train(study, arm['name'], device, stop_after=3, directory=scratch / arm['name'])
            if complete:
                raise AssertionError('Disposable correctness updates cannot complete a scientific fit')
            state = torch.load(scratch / arm['name'] / 'last.pt', map_location='cpu', weights_only=False)
            if state['step'] != 3 or not all(torch.isfinite(v).all() for v in state['model'].values()):
                raise AssertionError('Production-shape objective failed')
            print(f'Production correctness passed: {arm["name"]}', flush=True)
            del state
            torch.cuda.empty_cache()
    feature_controls, _ = controls(corpus, 'observed')
    temperature = np.array([r['temperature_K'] for r in corpus.records])
    nearest_neighbors(feature_controls['descriptor'], corpus.split['fit'], corpus.split['development'],
                      temperature, corpus.targets['phase'])
    conditions = (temperature[:, None] == np.unique(temperature)[None]).astype(np.float32)
    short = copy.deepcopy(study.config)
    short['probes'].update(updates=2, evaluate_every=1)
    metrics, _, _ = hazard_probe(feature_controls['descriptor'], conditions, corpus, short, 'mlp', device, None)
    if not np.isfinite(metrics['nll']):
        raise AssertionError('Frozen onset probe failed')
    if implementation() != before:
        raise ValueError('Implementation changed during preflight')
    study.bind()
    write_json(receipt, dict(passed=True, identity=study.identity,
        production_updates_per_arm=3, cached_graph_matches_native=True,
        future_targets_excluded_from_encoder=study.config['protocol'] != 'fixed_geometry_future_relation_v3', device=str(device),
        actual_backend=study.config['encoder']['backend'], onset_probe_finite=True,
        population=corpus.manifest['onset_counts']))
    print(json.dumps(json.loads(receipt.read_text()), indent=2), flush=True)
