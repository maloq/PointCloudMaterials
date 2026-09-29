"""Frozen-encoder spatial likelihood probes and shared context predictors."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.data.fixed_cohort.protocol import sha, write_json, digest
from src.research.equivariant_context.data import ContextCorpus
from src.research.equivariant_context.features import FeatureExtractor, prepare_graphs
from src.research.equivariant_context.model import ContextPredictor, event_log_probabilities, context_fields
from src.research.equivariant_context.normalization import apply
from src.research.equivariant_context.cache import RetainedCache, encoder_cache
from src.research.supervised_onset.model import Model
from src.research.supervised_onset.tracking import tracked_run
from src.research.local_predictability.metrics import source_weights
from src.research.structured_context.geometry import stencil
from src.project_runtime.paths import resolve_path
from .common import study


class LocalPredictor(nn.Module):
    def __init__(self, field, width):
        super().__init__()
        self.field = field
        self.layers = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 5))

    def forward(self, batch):
        x = batch['descriptor'] if self.field == 'descriptor' else batch['z'][:, 0]
        # Same six-bin factorization as the context predictor, now in Angstroms.
        return event_log_probabilities(self.layers(x))


def path_features(s, cache, device):
    saved = torch.load(s.checkpoint, map_location=device, weights_only=False)
    if saved['arm']['input'] != 'hot' or saved['config']['objective'] != 'hazard_nll':
        raise ValueError('Require the declared observed, likelihood-trained encoder')
    model = Model(saved['encoder_config'], saved['arm']).to(device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval().requires_grad_(False)
    extractor = FeatureExtractor(model.encoder, device, s.config['batch_size'], compile=s.config['compile'])
    records = []
    for item in s.plan['sources']:
        folder = s.technical / 'sources' / str(item['id'])
        for record in json.loads((folder/'paths.json').read_text())['paths']:
            dest = cache/f'{item["id"]}-{record["path_id"]}.npz'
            receipt = dest.with_suffix('.json')
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if old['identity'] != s.identity or old['sha256'] != sha(dest) or old['geometry_sha256'] != record['sha256']:
                    raise ValueError(f'Changed scan feature cache: {dest}')
            else:
                path = folder/record['file']
                if sha(path) != record['sha256']:
                    raise ValueError(f'Changed scan geometry: {path}')
                with np.load(path) as a:
                    patches = [x[np.linalg.norm(x, axis=-1) < 8.] for x in a['positions']]
                    arrays = prepare_graphs(patches, model.encoder.cutoff, pin_memory=True)
                    exported = extractor(arrays)
                    values = {k: v[a['inverse']] for k, v in exported.items()}
                    for k in ('actual', 'descriptor', 'distance', 'travel_A', 'atom', 'step',
                              'visible_local', 'visible_context', 'ptm_local', 'ptm_context'):
                        values[k] = a[k]
                np.savez(dest, **values)
                write_json(receipt, dict(identity=s.identity, geometry_sha256=record['sha256'], sha256=sha(dest)))
            records.append(record | {'features': str(dest)})
            print(json.dumps(dict(stage='scan-features', completed=len(records), source=item['id'])), flush=True)
    del extractor, model
    torch.cuda.empty_cache()
    write_json(s.technical/'scan-features.json', dict(identity=s.identity, records=records))
    return records


def spatial_labels(s):
    n = len(s.pop['source'])
    values, seen = {}, np.zeros(n, bool)
    for item in s.plan['sources']:
        folder = s.technical/'sources'/str(item['id'])
        receipt = json.loads((folder/'complete.json').read_text())
        if receipt['identity'] != s.identity or sha(folder/'labels.npz') != receipt['files']['labels.npz']:
            raise ValueError(f'Changed spatial targets: {folder}')
        with np.load(folder/'labels.npz') as a:
            ids = a['rows']
            if seen[ids].any():
                raise ValueError('Duplicate spatial target rows')
            seen[ids] = True
            for k in a.files:
                if k == 'rows':
                    continue
                if k not in values:
                    values[k] = np.empty((n,)+a[k].shape[1:], a[k].dtype)
                values[k][ids] = a[k]
    if not seen.all():
        raise ValueError('Incomplete fixed spatial population')
    values['event'] = np.searchsorted(s.config['distance_bins_A'], values['distance'], side='left').astype(np.int64)
    return values


@torch.no_grad()
def predict(model, corpus, ids, batch):
    model.eval()
    return torch.cat([model(model_batch(model, corpus, ids[i:i+batch])) for i in range(0, len(ids), batch)])


def model_batch(model, corpus, ids):
    fields = (model.field,) if isinstance(model, LocalPredictor) else context_fields(model.variant)
    batch = {k: corpus.features[k][ids] for k in fields}
    if not isinstance(model, LocalPredictor):
        batch['nominal'] = corpus.nominal[None].expand(len(ids), -1, -1)
    return batch


def fit(s, variant, corpus, targets, records, descriptor_stats, device):
    c = s.config
    folder = s.root/variant
    tech = folder/'technical'; tech.mkdir(parents=True, exist_ok=True)
    if (tech/'complete.json').exists():
        receipt = json.loads((tech/'complete.json').read_text())
        if receipt['identity'] != s.identity or any(sha(tech/n) != h for n, h in receipt['files'].items()):
            raise ValueError(f'Changed completed spatial fit: {variant}')
        return
    torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    if variant == 'geometry_mlp':
        model = LocalPredictor('descriptor', targets['descriptor'].shape[1]).to(device)
    elif variant == 'mace_local':
        model = LocalPredictor('z', 128).to(device)
    else:
        model = ContextPredictor(variant, **c['predictor']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['training']['learning_rate'], weight_decay=1e-5)
    fit_ids, val_ids = corpus.split['train'], corpus.split['selection']
    train_w = torch.as_tensor(source_weights(s.pop['source'][fit_ids])*len(fit_ids), dtype=torch.float32, device=device)
    val_w = torch.as_tensor(source_weights(s.pop['source'][val_ids]), dtype=torch.float32, device=device)
    best, first_epoch, update = float('inf'), 0, 0
    latest = tech/'last.pt'
    if latest.exists():
        last = torch.load(latest, map_location=device, weights_only=False)
        if last['identity'] != s.identity:
            raise ValueError('Changed resumed spatial predictor')
        model.load_state_dict(last['model']); optimizer.load_state_dict(last['optimizer'])
        best, first_epoch, update = last['best'], last['epoch'], last['update']
    def save(path, epoch):
        temporary = path.with_suffix('.building.pt')
        torch.save(dict(identity=s.identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        best=best, epoch=epoch, update=update, scalers=corpus.scalers,
                        descriptor_stats=descriptor_stats, config=c, variant=variant), temporary)
        temporary.replace(path)
    tracking = SimpleNamespace(config=dict(c, wandb=dict(c['wandb'], display_name=f'Spatial approach | {variant} | frozen observed MACE')),
                               root=folder, technical=tech, identity=s.identity)
    with tracked_run(tracking, variant, job_type='predictor') as run:
        run.summary['objective'] = 'source-weighted six-bin spatial-distance NLL'
        run.summary['encoder/trainable'] = False
        run.summary['encoder/checkpoint_sha256'] = s.pointer['encoder_sha256']
        run.summary['model/predictor_parameters'] = sum(p.numel() for p in model.parameters())
        run.summary['population/fixed_rows'] = len(s.pop['source'])
        for epoch in range(first_epoch, c['training']['epochs']):
            order = np.random.default_rng(c['seed']+epoch).permutation(len(fit_ids))
            accumulated = 0.
            model.train()
            for begin in range(0, len(order), c['batch_size']):
                index = order[begin:begin+c['batch_size']]
                ids = fit_ids[index]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.zeros((), device=device)
                for offset in range(0, len(ids), c['microbatch']):
                    part = ids[offset:offset+c['microbatch']]
                    logp = model(model_batch(model, corpus, part))
                    value = -(logp[torch.arange(len(part), device=device), corpus.events[part]]*
                              train_w[index[offset:offset+len(part)]]).sum()/len(ids)
                    if not torch.isfinite(value):
                        raise FloatingPointError(f'Nonfinite spatial loss: {variant}, epoch {epoch}')
                    value.backward(); loss += value.detach()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step(); update += 1
                accumulated += float(loss.detach())*len(ids)
            val = predict(model, corpus, val_ids, c['batch_size'])
            nll = float(-(val_w*val[torch.arange(len(val_ids), device=device), corpus.events[val_ids]]).sum())
            if not np.isfinite(nll):
                raise FloatingPointError(f'Nonfinite validation NLL: {variant}')
            if epoch+1 >= c['training']['minimum_selection_epoch'] and nll < best:
                best = nll; save(tech/'best.pt', epoch+1)
                run.summary['checkpoint/epoch'] = epoch+1
                run.summary['checkpoint/validation_distance_nll'] = best
            record = {'optimizer_update': update, 'train/epoch': epoch+1,
                      'train/distance_nll': accumulated/len(fit_ids), 'validation/distance_nll': nll}
            run.log(record)
            with (tech/'training.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            save(latest, epoch+1)
            print(json.dumps(dict(stage='training', variant=variant, **record)), flush=True)
        selected = torch.load(tech/'best.pt', map_location=device, weights_only=False)
        model.load_state_dict(selected['model'])
        logp = predict(model, corpus, np.arange(len(corpus.events)), c['batch_size']).cpu().numpy()
        np.savez_compressed(tech/'predictions.npz', logp=logp, **s.pop,
                            **{k: v for k, v in targets.items() if k != 'descriptor'})
        paths = []
        for record in records:
            with np.load(record['features']) as a:
                batch = {k: torch.as_tensor(apply(a[k], corpus.scalers.get(k)), device=device) for k in corpus.required_fields}
                batch['descriptor'] = torch.as_tensor((a['descriptor']-descriptor_stats['mean'])/descriptor_stats['scale'], dtype=torch.float32, device=device)
                batch['nominal'] = torch.as_tensor(stencil(), device=device)[None].expand(len(a['distance']), -1, -1)
                with torch.no_grad():
                    p = model(batch).cpu().numpy()
                paths.append(dict(record=record, logp=p, **{k: a[k] for k in (
                    'distance', 'travel_A', 'visible_local', 'visible_context', 'ptm_local', 'ptm_context')}))
        # Numeric path arrays plus JSON indexing; no pickled model inputs.
        offsets = np.r_[0, np.cumsum([len(p['distance']) for p in paths])]
        np.savez_compressed(tech/'path-predictions.npz', offsets=offsets,
                            **{k: np.concatenate([p[k] for p in paths]) for k in paths[0] if k != 'record'})
        write_json(tech/'paths.json', [p['record'] for p in paths])
        from .evaluate import evaluate
        metrics = evaluate(s, variant)
        run.summary['test/distance_nll'] = metrics['test']['nll']
        run.summary['test/path_false_alarm_rate'] = metrics['paths']['test_control_false_alarm_rate']
        run.summary['test/warning_at_least_12A_recall'] = metrics['paths']['recall_at_12A']
        write_json(tech/'complete.json', dict(identity=s.identity, variant=variant, epoch=selected['epoch'],
                   files={n: sha(tech/n) for n in ('best.pt', 'predictions.npz', 'path-predictions.npz', 'paths.json', 'metrics.json')}))
    del model, optimizer
    torch.cuda.empty_cache()


def worker(config_path):
    s = study(config_path)
    while not (s.technical/'prepared.json').exists():
        state = s.technical/'preparation-state.json'
        if state.exists() and json.loads(state.read_text())['state'] == 'failed':
            raise RuntimeError('Spatial data preparation failed; see preparation-state.json')
        time.sleep(15)
    c = s.config
    if json.loads((s.technical/'prepared.json').read_text())['identity'] != s.identity:
        raise ValueError('Prepared spatial identity mismatch')
    device = 'cuda:0'
    torch.set_num_threads(1)
    deadline = time.time()+12*3600
    retained = RetainedCache(resolve_path(c['cache_policy']['features']), c['cache_policy']['encoders_kept'])
    metadata = {k: s.pointer[k] for k in ('identity', 'domain', 'encoder_sha256')}
    with retained.lease(s.pointer['key'], deadline=deadline, metadata=metadata, shared=True):
        with encoder_cache(s, 'spatial-paths', s.checkpoint, deadline) as cache:
            records = path_features(s, cache, device)
            if not any(r['role']=='calibration' and r['kind']=='away' for r in records):
                raise ValueError('No calibration control paths; inspect recorded path exclusions')
            original = SimpleNamespace(config=c, identity=s.pointer['identity'])
            corpus = ContextCorpus(original, 'hot', ('harmonic_hierarchy',), device, s.features)
            targets = spatial_labels(s)
            corpus.events = torch.as_tensor(targets['event'], device=device)
            ids = corpus.split['train']; x = targets['descriptor'][ids].astype(float)
            w = source_weights(s.pop['source'][ids])
            mean = w@x; scale = np.sqrt(w@((x-mean)**2)).clip(1e-5)
            stats = dict(mean=mean.tolist(), scale=scale.tolist())
            corpus.features['descriptor'] = torch.as_tensor((targets['descriptor']-mean)/scale, dtype=torch.float32, device=device)
            prior = np.bincount(targets['event'][ids], weights=w, minlength=6)
            write_json(s.technical/'training-prior.json', dict(probabilities=prior.tolist(), role='train'))
            for variant in c['variants']:
                fit(s, variant, corpus, targets, records, stats, device)
            from .evaluate import collect
            collect(s)
