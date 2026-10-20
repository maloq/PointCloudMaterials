"""Local frozen-encoder measurement readouts, including raw-state controls."""
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from src.experiment_runner.metric_docs import write_metric_rows
from src.research.equivariant_context.cache import RetainedCache
from .data import root, cache, read, sha, digest, atomic_save, write_json
from .history_model import HistoryProcess
from .births import invariant, fit_spaces
from .history_evaluation import birth_config
from .measurement_model import MeasurementHead
from .measurement_data import GROUPS, FAMILY
from .measurement_learning import group_loss


def diagnose(c, p, seed, deadline):
    out = root(c)/'diagnostics'/f'seed-{seed}'; tech = out/'technical'; tech.mkdir(parents=True, exist_ok=True)
    if (tech/'complete.json').exists():
        return True
    data = torch.load(cache(c)/'births.pt', map_location='cpu', weights_only=False)
    y = torch.load(cache(c)/'birth-measurements/measurements.pt', map_location='cpu', weights_only=False).reshape(-1, 4, 244)[:, -1]
    train = torch.zeros(len(y), dtype=torch.bool)
    ancestors = ['']*len(y)
    for index, row in zip(data['indices'], data['rows']):
        train[index] = row['role'] == 'train'; ancestors[index] = row['ancestor']
    mean = y[train].mean(0); std = y[train].std(0); scale = std.clamp_min(1e-5); active = (std > 1e-8).cuda()
    target = ((y-mean)/scale).cuda(); train_ids = torch.where(train)[0].numpy()
    test = ~train; ancestors = np.asarray(ancestors); test_groups = np.unique(ancestors[test.numpy()])
    def source_mean(value):
        return float(np.mean([float(value[ancestors == a].mean()) for a in test_groups]))
    spec = p['frozen_diagnostic_models'][f'attn4-full-s{seed}']
    if sha(spec['path']) != spec['sha256']:
        raise ValueError('Changed frozen diagnostic checkpoint')
    metadata = dict(kind='measurement-diagnostic', plan=p['identity'], checkpoint=spec['sha256'])
    pool = RetainedCache(cache(c)/'features', 6)
    with pool.lease(digest(metadata), deadline=deadline, metadata=metadata) as bank:
        if not (bank/'diagnostic.pt').exists():
            saved = torch.load(spec['path'], map_location='cuda', weights_only=False)
            model = HistoryProcess(p['config'], saved['arm']).cuda().eval(); model.load_state_dict(saved['model'])
            full = []; pooled = []
            with torch.no_grad():
                for h in data['history'].split(128):
                    h = h.cuda()/h.new_tensor([5., 5., 5., 10., 10., 10.]).cuda(); x = h[:, -1]
                    z = model.condition(h).encode(x); full.append(z.cpu()); pooled.append(invariant(z, x).cpu())
            atomic_save(bank/'diagnostic.pt', dict(full=torch.cat(full), pooled=torch.cat(pooled)))
            write_json(bank/'diagnostic.json', dict(sha256=sha(bank/'diagnostic.pt')))
            del model; torch.cuda.empty_cache()
        if sha(bank/'diagnostic.pt') != read(bank/'diagnostic.json')['sha256']:
            raise ValueError('Changed frozen diagnostic features')
        features = torch.load(bank/'diagnostic.pt', map_location='cpu', weights_only=False)
        raw = torch.zeros(len(y), 80, 8, 3)
        raw[:, :, :2] = (data['state']/torch.tensor([5., 5., 5., 10., 10., 10.])).reshape(-1, 80, 2, 3)
        features['raw'] = raw
        all_predictions = {}; settings = c['diagnostic']; rows = []
        for kind in ['pooled', 'full', 'raw']:
            folder = out/kind/'technical'; folder.mkdir(parents=True, exist_ok=True)
            x = features[kind].cuda()
            input_transform = None
            if kind == 'pooled':
                x = torch.asinh(x); xm = x[train].mean(0); xs = x[train].std(0).clamp_min(1e-5); x = (x-xm)/xs
                input_transform = dict(mean=xm.cpu(), scale=xs.cpu(), preprocessing='asinh')
            torch.manual_seed(seed)
            model = (nn.Sequential(nn.Linear(180, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 244))
                     if kind == 'pooled' else MeasurementHead()).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=1e-5)
            rng = np.random.default_rng(seed); start = 0; learning = []
            if (folder/'last.pt').exists():
                saved = torch.load(folder/'last.pt', map_location='cuda', weights_only=False)
                model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); start = saved['update']
                rng.bit_generator.state = saved['rng']; learning = saved['learning']
            def save(update):
                atomic_save(folder/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    update=update, rng=rng.bit_generator.state, learning=learning, mean=mean, scale=scale,
                    input_transform=input_transform))
            for update in range(start+1, settings['updates']+1):
                if time.time() > deadline-300:
                    save(update-1); return False
                warm = min(update/settings['warmup'], 1.)
                decay = .01+.99*.5*(1+np.cos(np.pi*max(0, update-settings['warmup'])/(settings['updates']-settings['warmup'])))
                for group in optimizer.param_groups:
                    group['lr'] = settings['lr']*warm*decay
                ids = rng.choice(train_ids, settings['batch_size']); optimizer.zero_grad(set_to_none=True); total = 0.
                for offset in range(0, len(ids), settings['microbatch']):
                    take = ids[offset:offset+settings['microbatch']]
                    loss = group_loss(model(x[take])-target[take], active)
                    (loss*len(take)/len(ids)).backward(); total += float(loss.detach())*len(take)/len(ids)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True); optimizer.step()
                learning.append(dict(update=update, loss=total))
                if update % 100 == 0:
                    save(update)
                write_json(tech/'progress.json', dict(kind=kind, update=update, total=settings['updates']))
            save(settings['updates']); model.eval()
            with torch.no_grad():
                predicted = torch.cat([model(block).cpu() for block in x.split(256)])
            all_predictions[kind] = predicted.numpy()
            error = (predicted-target.cpu()).square()
            for name, lo, hi in GROUPS:
                value = error[:, lo:hi][:, active[lo:hi].cpu()].mean(-1)
                rows.append(dict(seed=seed, representation=kind, group=name, normalized_mse=source_mean(value),
                    train_states=int(train.sum()), test_states=int(test.sum()), test_ancestors=len(test_groups)))
            write_metric_rows(learning, folder.parent, family=FAMILY, name='learning')
            del model, optimizer, x; torch.cuda.empty_cache()
        write_metric_rows(rows, out, family=FAMILY, name='reconstruction')
        atomic_save(tech/'predictions.pt', dict(prediction=all_predictions, mean=mean, scale=scale,
            train=train, ancestors=ancestors, target=y))
        failures = fit_spaces(birth_config(c), p, data, f'diagnostic-s{seed}', all_predictions, family=FAMILY)
        if failures:
            raise RuntimeError('Frozen diagnostic birth probes failed')
    write_json(tech/'complete.json', dict(state='complete', seed=seed, checkpoint=spec['sha256'],
        budget=settings, online_tracking=False, reason='Frozen diagnostic readouts stay local'))
    return True
