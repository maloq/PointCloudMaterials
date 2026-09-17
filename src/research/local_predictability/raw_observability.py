"""All-state current crystallinity from trainable native atomic observations."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from src.data.predictive_memory.prepare import file_hash, write_json
from src.project_runtime.paths import load_json, resolve_path
from .native_data import SourceSampler, selection_indices
from .native_model import NativeEncoder
from .native_preflight import seed_all
from .native_queue import configure_file_limit
from .supervised import prepare_rows, save
from .native_runtime import ObservationPrefetcher, peek_batch, evaluate
from .metrics import hazard_loss, source_weights
from src.models.encoders.mace_backend import with_mace_backend, mace_backend_metadata


class CurrentStateModel(nn.Module):
    def __init__(self, *, mace_backend='cueq', **options):
        super().__init__()
        self.encoder = NativeEncoder(variant='snapshot', **options)
        self.classifier = nn.Linear(135, 1)
        self.encoder = with_mace_backend(self.encoder, mace_backend)

    def forward(self, observations, conditions):
        state = self.encoder(observations)
        return dict(state=state, logits=self.classifier(torch.cat((state, conditions), -1)))


def current_targets(rows, release, cache):
    """Join by tracked source/atom/frame identities; never filter current crystals."""
    by_source = {}
    for index, row in enumerate(rows):
        by_source.setdefault(row['source_id'], []).append(index)
    target = np.full(len(rows), -1, dtype=np.int64)
    for source in release['sources']:
        shard = cache / source['shard']
        if file_hash(shard) != source['shard_sha256']:
            raise ValueError(f"Changed label producer shard: {source['id']}")
        with np.load(shard) as data:
            lookup = {int(cid): i for i, cid in enumerate(data['atom_ids'])}
            indices = by_source[source['id']]
            centers = [lookup[rows[i]['center_id']] for i in indices]
            anchors = [rows[i]['anchor'] for i in indices]
            target[indices] = (~np.isin(data['labels'][centers, anchors], [1, 2, 3])).astype(np.int64)
    if np.any(target < 0):
        raise ValueError('Some raw-atom rows have no current PTM label')
    return target


def fit(config, windows, cond, target, selection, identity, resume=False):
    root = resolve_path(config['output']) / 'technical/current-state'
    root.mkdir(parents=True, exist_ok=True)
    seed_all()
    model = CurrentStateModel(activation_checkpoint=False, max_spatial_edges=config['max_spatial_edges'],
                              mace_backend=config['mace_backend']).cuda()
    train = np.flatnonzero(np.array([r['split'] for r in windows.rows]) == 'train')
    weight = source_weights(np.array([windows.rows[i]['source_id'] for i in train]))
    positive = np.clip(weight @ (target[train].cpu().numpy() == 0), 1e-4, 1-1e-4)
    with torch.no_grad():
        model.classifier.bias.fill_(float(np.log(positive/(1-positive))))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    sampler = SourceSampler(windows.rows)
    step = 0
    best = float('inf')
    latest = root / 'latest.pt'
    if latest.exists():
        if not resume:
            raise FileExistsError(f'Interrupted raw-state fit requires --resume: {latest}')
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['identity'] != identity or state['stage'] != 'current-state':
            raise ValueError('Raw-state exact resume identity differs')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        sampler.load_state_dict(state['sampler']); step = state['step']; best = state['best']
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
    started = time.monotonic()
    deadline = datetime.fromisoformat(config['training_deadline_utc']).timestamp()
    with (root / 'training.jsonl').open('a' if resume else 'x') as log, ObservationPrefetcher(windows) as inputs:
        future = inputs.submit(peek_batch(sampler), 'snapshot') if step < config['updates_per_stage'] else None
        while step < config['updates_per_stage']:
            if time.time()+180 >= deadline:
                save(latest, model, optimizer, sampler, step, best, identity, 'current-state')
                raise TimeoutError(f'Raw-state training deadline reached at step {step}')
            indices = sampler.batch()
            observations = inputs.take(future, indices)
            future = (inputs.submit(peek_batch(sampler), 'snapshot')
                      if step+1 < config['updates_per_stage'] else None)
            optimizer.zero_grad(set_to_none=True)
            result = model(observations, cond[indices])
            loss = hazard_loss(result['logits'], target[indices]).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step(); step += 1
            if step == 1 or step % 50 == 0:
                record = dict(step=step, train_nll=float(loss.detach()), gradient_norm=float(norm),
                              elapsed_seconds=time.monotonic()-started, input_wait_seconds=inputs.wait_seconds)
                log.write(json.dumps(record)+'\n'); log.flush()
                print(json.dumps(record), flush=True)
                write_json(root / 'status.json', dict(state='running', **record))
            if step % 256 == 0 or step == config['updates_per_stage']:
                inputs.drain()
                score, _, _ = evaluate(model, windows, selection, cond, target, 8)
                if score < best:
                    best = score
                    save(root / 'best.pt', model, optimizer, sampler, step, best, identity, 'current-state')
                save(latest, model, optimizer, sampler, step, best, identity, 'current-state')
                print(json.dumps(dict(step=step, selection_nll=score, best=best)), flush=True)
    write_json(root / 'complete.json', dict(state='complete', step=step, identity=identity))
    del model, optimizer
    torch.cuda.empty_cache()
    return root / 'best.pt'


def run(config, resume=False):
    limits = configure_file_limit()
    torch.set_num_threads(config['torch_threads'])
    root = resolve_path(config['output']) / 'technical'
    root.mkdir(parents=True, exist_ok=True)
    status = root / 'raw_observability_status.json'
    write_json(status, dict(state='preparing', file_limit=limits))
    try:
        if config['seed'] != 20260919:
            raise ValueError('Only the declared single seed is allowed')
        windows, _, cond, _, _, _, base_identity = prepare_rows(config)
        cache = resolve_path(config['cache'])
        release = json.loads((cache / 'release.json').read_text())
        if file_hash(cache / 'release.json') != base_identity['release_sha256']:
            raise ValueError('Cached current labels differ from the native release')
        values = current_targets(windows.rows, release, cache)
        target = torch.tensor(values, device='cuda', dtype=torch.long)
        selection = selection_indices(windows.rows)
        np.savez(root / 'current_labels.npz', binary_target=values,
                 **{key: np.array([r[key] for r in windows.rows]) for key in ['source_id', 'center_id', 'anchor', 'split']})
        identity = dict(**base_identity, task='current PTM crystal state on all native rows',
                        raw_observability_sha256=file_hash(Path(__file__)),
                        mace_backend=mace_backend_metadata(config['mace_backend']),
                        backend_implementation={name: file_hash(Path(__file__).parents[2]/'models/encoders'/name)
                                                for name in ['mace_backend.py', 'mace_causal.py']},
                        native_runtime_sha256=file_hash(Path(__file__).with_name('native_runtime.py')),
                        current_labels_sha256=file_hash(root / 'current_labels.npz'))
        identity['selection_indices'] = selection
        write_json(root / 'identity.json', identity)
        write_json(status, dict(state='training', file_limit=limits))
        checkpoint = fit(config, windows, cond, target, selection, identity, resume)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = CurrentStateModel(activation_checkpoint=False, max_spatial_edges=config['max_spatial_edges'],
                                  mace_backend=config['mace_backend']).cuda()
        model.load_state_dict(state['model'])
        for split in ['selection', 'calibration', 'test', 'train']:
            write_json(status, dict(state='exporting', split=split, file_limit=limits))
            indices = np.flatnonzero(np.array([r['split'] for r in windows.rows]) == split)
            _, logits, z = evaluate(model, windows, indices, cond, target, 8)
            np.savez(root / f'{split}_predictions.npz', indices=indices,
                     probability=logits.sigmoid().numpy(), logits=logits.numpy(), embeddings=z.numpy(),
                     binary_target=values[indices],
                     **{key: np.array([windows.rows[i][key] for i in indices])
                        for key in ['source_id', 'center_id', 'anchor', 'split']})
            print(f'Saved raw-state {split} predictions: {len(indices)} rows', flush=True)
        write_json(status, dict(state='complete', steps=config['updates_per_stage'],
                               analysis='deferred until requested', file_limit=limits))
    except Exception as exc:
        write_json(status, dict(state='failed', error=repr(exc), file_limit=limits))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(load_json(args.config), args.resume)


if __name__ == '__main__':
    main()
