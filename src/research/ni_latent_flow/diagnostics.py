"""Fixed-budget, fixed-panel forecast diagnostics; never a model selector.

All distances use the shared train-normalized 244 measurements. The unconditional
reference samples complete saved training futures, without anchoring or stitching;
these are reference data samples, not independent physical MD continuations.
"""
from collections import defaultdict
import time

import numpy as np
import torch

from src.experiment_runner.metric_docs import write_metric_rows
from src.research.ni_kinetic_latent.assessment import (
    energy, estimate, scores, selected_rows, source_values,
)
from .data import load, root, read, sha, digest, atomic_save, write_json
from .learning import Model


def _panel(bank, settings, seed):
    """Rank row identities, without inspecting any future or measurement value."""
    ids, splits = [], []
    for split in ('train', 'test'):
        chosen = selected_rows(dict(bank, test_rows=bank[split+'_rows']),
                               settings['per_source_regime'], seed)
        ids.extend(chosen.tolist()); splits.extend([split]*len(chosen))
    return np.asarray(ids, dtype=np.int64), np.asarray(splits)


def _shuffle_donors(bank, ids, splits, seed):
    pools = defaultdict(list)
    for split in ('train', 'test'):
        for i in bank[split+'_rows']:
            row = bank['rows'][i]
            pools[split, row['ancestor'], row['regime']].append(int(i))
    donors = []
    for i, split in zip(ids, splits):
        row = bank['rows'][i]
        choices = [j for j in pools[split, row['ancestor'], row['regime']] if j != i]
        if not choices:
            raise ValueError('No independent history donor for '+row['id'])
        donors.append(min(choices, key=lambda j: digest(
            [seed, 'history-donor', row['id'], bank['rows'][j]['id']])))
    return np.asarray(donors, dtype=np.int64)


def _reference_donors(bank, ids, particles, seed):
    pools = defaultdict(list)
    for i in bank['train_rows']:
        pools[bank['rows'][i]['ancestor']].append(int(i))
    ancestors = sorted(pools)
    donors = []
    for i in ids:
        rng = np.random.default_rng(int(digest([seed, 'training-reference',
                                               bank['rows'][i]['id']])[:16], 16))
        # Every ancestor has equal probability, irrespective of window count.
        source = rng.integers(len(ancestors), size=particles)
        donors.append([int(rng.choice(pools[ancestors[s]])) for s in source])
    return np.asarray(donors, dtype=np.int64)


def _increment_scale(bank):
    # The old mixed-cohort training rows are shared by all data-size arms.
    # They fix the score geometry while the fitted predictor budget changes.
    rows = np.asarray(bank['reference_train_rows']).tolist()
    total = torch.zeros(244, dtype=torch.float64)
    count = 0
    for chunk in np.array_split(rows, max(1, (len(rows)+127)//128)):
        delta = bank['x'][chunk].double().diff(dim=1)
        total += delta.square().sum((0, 1)); count += delta.shape[0]*delta.shape[1]
    return (total/count).sqrt().clamp_min(.05).float(), rows


def _checkpoint(c, arm, bank, plan, checkpoint):
    folder = root(c)/'analyses'/arm['name']/'technical'
    path = folder/('final.pt' if checkpoint == 'final' else checkpoint+'.pt')
    checksum = sha(path)
    state = torch.load(path, map_location='cuda', weights_only=False)
    expected = digest([plan['identity'], arm])
    if state['identity'] != expected or state['data_identity'] != plan['data_identity']:
        raise ValueError('Diagnostic checkpoint identity mismatch: '+str(path))
    if checkpoint == 'final':
        receipt = read(folder/'complete.json')
        if receipt['identity'] != expected or receipt['checkpoint_sha256'] != checksum:
            raise ValueError('Changed final checkpoint: '+str(path))
    elif state['stage'] != 'dynamics' or state['stage_update'] != int(checkpoint.split('-')[1]):
        raise ValueError('Wrong fixed-update diagnostic checkpoint: '+str(path))
    model = Model(c, arm, bank).cuda()
    model.load_state_dict(state['model'])
    if arm['predictor'] == 'block':
        model.predictor.load_state_dict(state['predictor_ema'])
    return model.eval(), checksum, state['stage_update']


def _marginal_scores(samples, target, weights):
    """Marginal scores at each native 0.1-ps frame, not connected-path scores."""
    b, p, t, d = samples.shape
    marginal = samples.transpose(1, 2).reshape(b*t, p, 1, d)
    actual = target.reshape(b*t, 1, d)
    return dict(marginal_state_energy=energy(marginal, actual, weights).reshape(b, t),
        marginal_state_coverage90=(((target >= samples.quantile(.05, dim=1)) &
            (target <= samples.quantile(.95, dim=1))).float()*weights).sum(-1))


def _score(samples, target, initial, weights, increment_scale, *, teacher_forced=False):
    """Do not retain a surviving subset when any ensemble member is nonfinite."""
    result = {}
    for steps in (10, 30, 60):
        prediction, truth = samples[:, :, :steps], target[:, :steps]
        finite = torch.isfinite(prediction).all((1, 2, 3))
        # Avoid invalid arithmetic inside cdist/quantile. All affected row scores
        # below become undefined; substituted values never contribute a score.
        safe = torch.where(torch.isfinite(prediction), prediction, 0.)
        marginal = _marginal_scores(safe, truth, weights)
        values = {key:value.mean(1) for key, value in marginal.items()}
        if not teacher_forced:
            values.update(scores(safe, truth, initial, weights, increment_scale))
        for metric, value in values.items():
            result[steps, metric] = value.masked_fill(~finite, float('nan')).cpu().numpy()
        result[steps, 'nonfinite_path_fraction'] = (
            ~torch.isfinite(prediction).all((2, 3))).float().mean(1).cpu().numpy()
    return result


@torch.no_grad()
def _teacher_forced(model, observed, particles, seed):
    z = model.encode(observed)
    frames = torch.arange(3, 63, device=z.device)
    history = z[:, frames[:, None]+torch.arange(-3, 1, device=z.device)]
    history = history[:, None].expand(-1, particles, -1, -1, -1)
    rng = torch.Generator(device=z.device).manual_seed(seed)
    return model.decode(model.predictor.sample(history, rng))


def _combinations(c, arm):
    settings = c['diagnostics']
    grid = 32 if arm['predictor'] == 'block' else None
    items = [dict(checkpoint='update-'+str(n), gridpoints=grid, control='model')
             for n in settings['snapshot_updates']]
    items += [dict(checkpoint='final', gridpoints=n, control='model')
              for n in (settings['gridpoints'] if arm['predictor'] == 'block' else [None])]
    items += [dict(checkpoint='final', gridpoints=grid, control=control)
              for control in ('shuffled_history', 'training_trajectory_reference', 'persistence')]
    if arm['predictor'] == 'mdn':
        items.append(dict(checkpoint='final', gridpoints=None, control='teacher_forced'))
    return items


def _tables(arm, bank, ids, splits, packets, settings, seed):
    ancestors = np.asarray([bank['rows'][i]['ancestor'] for i in ids])
    cohorts = np.asarray([bank['rows'][i]['source_cohort'] for i in ids])
    regimes = np.asarray([bank['rows'][i]['regime'] for i in ids])
    records, sources = [], []
    for packet in packets:
        for (steps, metric), value in packet['values'].items():
            for split in ('train', 'test'):
                for cohort in ('all', *sorted(set(cohorts))):
                    for regime in ('all', *sorted(set(regimes))):
                        chosen = ((splits == split) & ((cohorts == cohort) if cohort != 'all' else True)
                                  & ((regimes == regime) if regime != 'all' else True))
                        if not chosen.any():
                            continue
                        base = dict(model=arm['name'], **packet['combination'],
                            dynamics_update=packet['dynamics_update'], split=split,
                            source_cohort=cohort, regime=regime, horizon_ps=steps*.1, metric=metric)
                        records.append(dict(base, rows=int(chosen.sum()),
                            **estimate(value[chosen], ancestors[chosen], seed, settings['bootstrap'])))
                        keys, means = source_values(value[chosen], ancestors[chosen])
                        sources.extend(dict(base, ancestor=str(key),
                            value=float(mean) if np.isfinite(mean) else None) for key, mean in zip(keys, means))
    return records, sources


@torch.no_grad()
def evaluate(c, arm, deadline, *, family):
    """Resume every fixed checkpoint/sampler/control independently, then export."""
    out = root(c)/'analyses'/arm['name']
    technical = out/'technical'
    folder = technical/'diagnostics'; folder.mkdir(parents=True, exist_ok=True)
    settings = c['diagnostics']; seed = settings.get('seed', c['seed'])
    plan, bank = load(c, arm)
    ids, splits = _panel(bank, settings, seed)
    donors = _shuffle_donors(bank, ids, splits, seed)
    reference = _reference_donors(bank, ids, settings['particles'], seed)
    increment_scale, reference_rows = _increment_scale(bank)
    identity = digest([plan['identity'], arm, settings, ids.tolist(), splits.tolist(), sha(__file__)])
    marker = technical/'diagnostics-complete.json'
    if marker.exists():
        if read(marker)['identity'] != identity:
            raise ValueError('Changed frozen diagnostics: '+arm['name'])
        return True
    write_json(folder/'panel.json', dict(identity=identity, indices=ids.tolist(),
        splits=splits.tolist(), shuffled_history_indices=donors.tolist(),
        training_reference_indices=reference.tolist(), increment_reference_indices=reference_rows,
        selection='Outcome-blind hash rank within ancestor/regime; train/test separate',
        shuffled_history='Different saved window from same split, ancestor and regime; complete four-frame history replaced',
        reference='Source-balanced complete saved training futures, no recentering; reference samples, not MD continuations',
        noise='Same particle seed for matching panel batches across models, fixed checkpoints, grids and shuffled histories',
        horizon='Native 0.1 ps; four past frames span0.3 ps; forecast60frames through6ps',
        checkpoint='Fixed updates and final, EMA for block predictor; no checkpoint or grid selection',
        normalization='Shared old16 mixed-cohort training reference for increment scale; original shared244 measurement normalization',
        probability='Source/cohort-enriched observational windows, not natural nucleation incidence'))
    weights = bank['weights'].cuda(); increment_scale = increment_scale.cuda()
    packets = []; model = None; previous = None
    for combination in _combinations(c, arm):
        name = '-'.join(str(combination[k]) for k in ('checkpoint', 'gridpoints', 'control'))
        path = folder/(name+'.pt')
        if time.time() > deadline-60:
            return False
        if previous != combination['checkpoint']:
            del model
            model, checksum, update = _checkpoint(c, arm, bank, plan, combination['checkpoint'])
            previous = combination['checkpoint']
        packet_identity = digest([identity, combination, checksum])
        packet = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else dict(
            identity=packet_identity, combination=combination, checkpoint_sha256=checksum,
            dynamics_update=update, offset=0, values={})
        if packet['identity'] != packet_identity:
            raise ValueError('Changed diagnostic combination: '+name)
        if arm['predictor'] == 'block':
            model.predictor.flow_steps = combination['gridpoints']
        for offset in range(packet['offset'], len(ids), settings['batch_size']):
            if time.time() > deadline-60:
                return False
            part = slice(offset, offset+settings['batch_size'])
            chunk = ids[part]
            observed = bank['x'][chunk].cuda()
            inputs = bank['input'][chunk].cuda()
            sample_seed = seed+970003+offset
            control = combination['control']
            failed = False
            try:
                if control == 'training_trajectory_reference':
                    samples = bank['x'][reference[part], 4:].cuda()
                elif control == 'persistence':
                    samples = observed[:, None, 3:4].expand(-1, 1, 60, -1)
                elif control == 'teacher_forced':
                    samples = _teacher_forced(model, inputs, settings['particles'], sample_seed)
                else:
                    history = (bank['input'][donors[part], :4].cuda() if control == 'shuffled_history'
                               else inputs[:, :4])
                    _, samples = model.rollout(history, 60, settings['particles'], sample_seed)
            except RuntimeError as error:
                if 'probability tensor contains either' not in str(error):
                    raise
                # An invalid MDN distribution is a scientific generation
                # failure. Preserve an undefined full batch, not survivors or
                # fallback predictions; independent combinations still run.
                failed = True
                samples = torch.full((len(chunk), settings['particles'], 60, 244),
                                     float('nan'), device=observed.device)
                write_json(folder/(name+f'-numerical-failure-{offset:05d}.json'), dict(
                    identity=packet_identity, combination=combination, indices=chunk.tolist(),
                    error=str(error), treatment='Whole affected batch undefined; no substitute samples'))
            values = _score(samples.float(), observed[:, 4:], observed[:, 3], weights,
                            increment_scale, teacher_forced=control == 'teacher_forced')
            for steps in (10, 30, 60):
                values[steps, 'generation_failure_fraction'] = np.full(len(chunk), float(failed))
                if failed:
                    values[steps, 'nonfinite_path_fraction'][:] = np.nan
            for key, value in values.items():
                packet['values'][key] = np.concatenate((packet['values'][key], value)) if key in packet['values'] else value
            packet['offset'] = offset+len(chunk)
            atomic_save(path, packet)
            write_json(folder/'progress.json', dict(combination=combination, rows=packet['offset'], total=len(ids)))
        packets.append(packet)
    records, sources = _tables(arm, bank, ids, splits, packets, settings, seed)
    write_metric_rows(records, out, family=family, name='diagnostics')
    write_metric_rows(sources, out, family=family, name='diagnostics_sources')
    write_json(marker, dict(identity=identity, state='complete', combinations=len(packets),
        panel_rows=len(ids), particles=settings['particles'], checkpoints='fixed, never selected',
        receipts=[dict(combination=p['combination'], checkpoint_sha256=p['checkpoint_sha256']) for p in packets]))
    return True
