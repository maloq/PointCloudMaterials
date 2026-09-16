"""Frozen-state readouts and matched access-to-raw-history sufficiency tests."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.experiment_runner.registry import sha256, write_json
from .data import atomic_save, load
from .evaluate import encode, export
from .objective import PhysicalHeads, targets, task_objective
from .train import initialize, validation_score, variant_root
from .evaluate import physical_metrics
from .runtime import CausalRuntime


class LinearHeads(nn.Module):
    """Separate linear decoders at each horizon, from the exported z alone."""
    def __init__(self, dim, horizons, event_bins):
        super().__init__()
        self.horizons = horizons
        self.present = nn.Linear(dim, 169)
        self.future = nn.Linear(dim+1, horizons*340)
        self.hazard = nn.Linear(dim+1, event_bins) if event_bins else None

    def forward(self, z, temperature):
        condition = torch.cat((z, temperature[:, None]/1000.), -1)
        return dict(present=self.present(z), future=self.future(condition).reshape(-1, self.horizons, 340),
                    scale=None, hazard=None if self.hazard is None else self.hazard(condition))


def run(config, args, modes=None):
    """Train on train sources, select on val, and evaluate untouched test sources.

    `state_constant` and `state_history` have identical trainable architecture,
    parameter count and budget. The former receives one fixed training history
    for every example, so its only varying atomic information is frozen z.
    The latter also sees the original observed atomic history. Improvement in
    physical future skill is evidence that z discarded available information.
    """
    torch.set_num_threads(config['cpu_threads'])
    samples, identity = load(config)
    root = variant_root(config, args.variant)
    checkpoint = root/'technical/best.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['identity']['cache_sha256'] != identity['cache_sha256'] or saved['config'] != config:
        raise ValueError('Probe config/cache must match the frozen encoder checkpoint')
    frozen, _ = initialize(config, args.variant, args.device)
    frozen.load_state_dict(saved['encoder_state'], strict=True)
    frozen.eval().requires_grad_(False)
    norm = saved['normalization']
    runtime = CausalRuntime(samples, norm, frozen, args.device, config.get('runtime'))
    with torch.no_grad():
        z = runtime.encode(frozen, samples).detach()
    train_ids = [i for i, s in enumerate(samples) if s['split'] == 'train']
    by_source = {sid: [i for i in train_ids if samples[i]['source_id'] == sid]
                 for sid in sorted({samples[i]['source_id'] for i in train_ids})}
    constant = samples[train_ids[0]]
    spec = config['probes']
    modes = ('linear', 'nonlinear', 'state_constant', 'state_history') if modes is None else tuple(modes)
    if not modes or len(set(modes)) != len(modes) or set(modes)-{'linear', 'nonlinear', 'state_constant', 'state_history'}:
        raise ValueError(f'Invalid or duplicate probe modes: {modes}')
    for mode in modes:
        out = root.with_name(f'{root.name}-probe-{mode}')
        if out.exists():
            raise FileExistsError(f'Preserve existing probe: {out}')
        (out/'technical').mkdir(parents=True)
        torch.manual_seed(config['seed']+31)
        branch = None
        dim = frozen.invariant_dim
        if mode.startswith('state_'):
            branch, _ = initialize(config, 'D', args.device)
            dim *= 2
        torch.manual_seed(config['seed']+32)
        event_bins = config['events']['bin_edges_ps'] if config['events']['enabled'] else []
        if mode == 'linear':
            head = LinearHeads(dim, len(config['future_lags_ps']), len(event_bins)).to(args.device)
        else:
            head = PhysicalHeads(dim, config['future_lags_ps'], hidden=spec['hidden'],
                                 probabilistic=False, event_bins_ps=event_bins).to(args.device)
        parameters = list(head.parameters())+(list(branch.parameters()) if branch is not None else [])
        optimizer = torch.optim.AdamW(parameters, lr=spec['learning_rate'], weight_decay=0.)
        rng = np.random.default_rng(config['seed']+33)
        best, history = float('inf'), []

        def predict(indices, constant_features=None):
            state = z[indices]
            if branch is not None:
                if mode == 'state_constant':
                    feature = runtime.encode(branch, [constant]) if constant_features is None else constant_features
                    feature = feature.expand(len(indices), -1)
                else:
                    feature = runtime.encode(branch, [samples[i] for i in indices])
                state = torch.cat((state, feature), -1)
            y = runtime.target([samples[i] for i in indices])
            return head(state, y['temperature']), y

        @torch.no_grad()
        def evaluate(indices):
            head.eval()
            if branch is not None:
                branch.eval()
            constant_features = runtime.encode(branch, [constant]) if mode == 'state_constant' else None
            pieces = {k: [] for k in ('present', 'future', 'scale', 'hazard')}
            # Frozen readouts and the constant-history control need only tiny
            # dense heads after one fixed branch encoding per evaluation pass.
            chunk = runtime.batch_size if mode == 'state_history' else 1024
            for start in range(0, len(indices), chunk):
                pred, _ = predict(indices[start:start+chunk], constant_features)
                for key, value in pred.items():
                    if value is not None:
                        pieces[key].append(value)
            result = {k: torch.cat(v).cpu().numpy() if v else None for k, v in pieces.items()}
            result['z'] = z[indices].cpu().numpy()
            if any(v is not None and not np.isfinite(v).all() for v in result.values()):
                raise FloatingPointError(f'Nonfinite batched {mode} probe output')
            return result

        val_ids = [i for i, s in enumerate(samples) if s['split'] == 'val']
        for step in range(spec['steps']+1):
            if step:
                head.train()
                if branch is not None:
                    branch.train()
                source_ids = rng.choice(list(by_source), config['training']['batch_sources'], replace=True)
                indices = [int(rng.choice(by_source[s])) for s in source_ids]
                optimizer.zero_grad(set_to_none=True)
                pred, target = predict(indices)
                value, _ = task_objective(pred, target, **config['loss'])
                value.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizer.step()
            if step % spec['validation_every'] == 0 or step == spec['steps']:
                result = evaluate(val_ids)
                _, constraint = physical_metrics(result, [samples[i] for i in val_ids], norm, config['future_lags_ps'])
                score = validation_score(constraint, config, 'D')
                history.append(dict(step=step, validation_score=score))
                write_json(out/'technical/history.json', history)
                write_json(out/'technical/status.json', dict(state='training', step=step, validation_score=score))
                if score < best:
                    best = score
                    atomic_save(out/'technical/best.pt', dict(head=head.state_dict(),
                        branch=None if branch is None else branch.state_dict(), mode=mode, step=step,
                        config=config, normalization=norm, frozen_checkpoint_sha256=sha256(checkpoint),
                        parameter_count=sum(p.numel() for p in parameters)))
                print('CAUSAL PROBE', mode, step, score, flush=True)
        best_state = torch.load(out/'technical/best.pt', weights_only=False, map_location='cpu')
        head.load_state_dict(best_state['head'], strict=True)
        if branch is not None:
            branch.load_state_dict(best_state['branch'], strict=True)
        thresholds = None
        if config['events']['enabled']:
            from .events import thresholds_from_validation
            thresholds = thresholds_from_validation(evaluate(val_ids)['hazard'], [samples[i] for i in val_ids],
                                                     config['events']['maximum_false_alarm_rate'])
        for split in ('val', 'test'):
            ids = [i for i, s in enumerate(samples) if s['split'] == split]
            export(out, evaluate(ids), [samples[i] for i in ids], norm, config, split, thresholds)
        write_json(out/'technical/history.json', history)
        write_json(out/'technical/status.json', dict(state='complete', selected_step=best_state['step'],
                   frozen_checkpoint_sha256=best_state['frozen_checkpoint_sha256']))
