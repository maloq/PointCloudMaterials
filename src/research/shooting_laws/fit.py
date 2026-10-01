"""Matched diagnostic readouts selected only by proper predictive likelihood."""
import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.special import softmax

from src.data.fixed_cohort.protocol import sha, write_json
from .common import folder, result, plan, read
from .data import load
from .features import bank


class Head(nn.Module):
    def __init__(self, width, target_dim, mixtures, hidden):
        super().__init__()
        self.target_dim, self.mixtures = target_dim, mixtures
        output = mixtures * (1 + 2 * target_dim) if target_dim else 13
        self.net = (nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, output))
                    if width else nn.Linear(1, output))

    def forward(self, x):
        return self.net(x if x.shape[1] else torch.ones((len(x), 1), device=x.device))

    def parameters_for(self, x):
        values = self(x).reshape(len(x), self.mixtures, 1 + 2 * self.target_dim)
        log_weight = torch.log_softmax(values[:, :, 0], -1)
        mean = values[:, :, 1:1 + self.target_dim]
        # A declared 0.10 standardized-unit floor prevents singular mixture fits.
        scale = .10 + torch.nn.functional.softplus(values[:, :, 1 + self.target_dim:])
        return log_weight, mean, scale

    def loss(self, x, target):
        if not self.target_dim:
            logp = torch.log_softmax(self(x), -1)
            return -logp[:, None, :].expand(-1, target.shape[1], -1).gather(2, target.long()[:, :, None]).squeeze(-1).mean(1)
        logw, mu, sigma = self.parameters_for(x)
        residual = (target[:, :, None] - mu[:, None]) / sigma[:, None]
        normal = -.5 * residual.square().sum(-1) - sigma.log().sum(-1)[:, None] - .5 * self.target_dim * math.log(2 * math.pi)
        return -torch.logsumexp(normal + logw[:, None], -1).mean(1)


def source_weights(c, data):
    parents = plan(c)['parents']
    sources = np.array([parents[int(i)]['source'] for i in data['parent']])
    weights = data['weights'].astype(float).copy()
    # Each root/source has equal total mass, regardless of parent/center count.
    for s in np.unique(sources):
        keep = sources == s
        weights[keep] /= weights[keep].sum()
    return sources, weights


def moments(x, weights):
    w = weights / weights.sum()
    mean = np.einsum('n,nd->d', w, x)
    scale = np.sqrt(np.einsum('n,nd->d', w, (x - mean) ** 2))
    return mean, np.maximum(scale, 1e-5)


def problem(c, arm):
    data, manifest = load(c)
    x = bank(c, data, manifest, arm)
    roles = np.array([plan(c)['parents'][int(i)]['role'] for i in data['parent']])
    sources, weights = source_weights(c, data)
    mean, scale = moments(x[roles == 'train'], weights[roles == 'train'])
    x = ((x - mean) / scale).astype(np.float32)
    y = data['future'].reshape(len(x), 12, -1)
    train_y = y[roles == 'train'].reshape(-1, y.shape[-1])
    ym, ys = moments(train_y, np.repeat(weights[roles == 'train'] / 12, 12))
    y = ((y - ym) / ys).astype(np.float32)
    return data, manifest, x, y, roles, sources, weights, dict(x_mean=mean, x_scale=scale, y_mean=ym, y_scale=ys)


def train_head(c, x, y, roles, weights, *, seed, target_dim, root):
    torch.manual_seed(seed)
    torch.set_num_threads(c['fit_threads'])
    model = Head(x.shape[1], target_dim, c['mixtures'], c['hidden'])
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    tw = torch.as_tensor(weights, dtype=torch.float32)
    train = np.flatnonzero(roles == 'train')
    selection = np.flatnonzero(roles == 'selection')
    if not len(train) or not len(selection):
        raise ValueError('No train/selection observations after risk restriction')
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    best, best_state, stale, history = math.inf, None, 0, []
    generator = np.random.default_rng(seed)

    def evaluate(ids):
        with torch.no_grad():
            losses = torch.cat([model.loss(tx[ix], ty[ix]) for ix in np.array_split(ids, max(1, math.ceil(len(ids) / 256)))])
        return float((losses * tw[ids]).sum() / tw[ids].sum())

    for epoch in range(c['epochs']):
        model.train()
        shuffled = generator.permutation(train)
        for start in range(0, len(shuffled), 256):
            ix = shuffled[start:start + 256]
            # Normalize by the fixed full-train weight, not each minibatch's
            # random phase mixture; every original row is visited once/epoch.
            loss = (model.loss(tx[ix], ty[ix]) * tw[ix]).sum() * len(train) / (len(ix) * tw[train].sum())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite likelihood at epoch {epoch}: {root}')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
        model.eval()
        score = evaluate(selection)
        history.append(dict(epoch=epoch + 1, train_nll=evaluate(train), selection_nll=score))
        if score < best - 1e-6:
            best, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
            torch.save(dict(model=best_state, input_dim=x.shape[1], target_dim=target_dim, mixtures=c['mixtures'],
                            hidden=c['hidden'], epoch=epoch + 1, selection_nll=best), root / 'checkpoint.pt')
        else:
            stale += 1
        write_json(root / 'progress.json', history[-1])
        if stale >= c['patience']:
            break
    model.load_state_dict(best_state)
    write_json(root / 'history.json', history)
    return model.eval()


def fit(c, index):
    arm = c['arms'][index // len(c['fit_seeds'])]
    seed = c['fit_seeds'][index % len(c['fit_seeds'])]
    root = result(c) / 'analyses' / 'readouts-v1' / arm['name'] / str(seed)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'complete.json').exists():
        receipt = read(root / 'complete.json')
        if receipt['config'] != c or receipt['producer_sha256'] != sha(Path(__file__)):
            raise ValueError('Changed completed readout')
        return receipt
    data, manifest, x, y, roles, sources, weights, normalization = problem(c, arm)
    np.savez(root / 'normalization.npz', **normalization)
    outputs = {}
    for kind, target, valid in [('path', y, np.ones(len(x), bool)), ('event', data['event'], np.all(data['event'] >= 0, axis=1))]:
        out = root / kind
        out.mkdir(exist_ok=True)
        effective_weights = weights[valid].copy()
        for source in np.unique(sources[valid]):
            group = sources[valid] == source
            effective_weights[group] /= effective_weights[group].sum()
        model = train_head(c, x[valid], target[valid], roles[valid], effective_weights, seed=seed,
                           target_dim=y.shape[-1] if kind == 'path' else 0, root=out)
        tensor = torch.from_numpy(x)
        with torch.no_grad():
            if kind == 'path':
                parts = [model.parameters_for(tensor[i:i + 256]) for i in range(0, len(x), 256)]
                logw, mu, sigma = [torch.cat([v[j] for v in parts]).numpy() for j in range(3)]
                outputs.update(log_weight=logw, mean=mu, scale=sigma)
            else:
                logits = torch.cat([model(tensor[i:i + 256]) for i in range(0, len(x), 256)]).numpy()
                outputs['event_probability'] = softmax(logits, axis=-1)
    np.savez_compressed(root / 'predictions.npz', **outputs)
    receipt = dict(config=c, arm=arm, seed=seed, data_identity=manifest['identity'],
                   producer_sha256=sha(Path(__file__)), predictions_sha256=sha(root / 'predictions.npz'),
                   objective='joint Gaussian-mixture path NLL; separate 13-category event-time NLL',
                   selection='source-held-out NLL', tracking='local frozen diagnostic readout; no encoder fitting')
    write_json(root / 'complete.json', receipt)
    return receipt
