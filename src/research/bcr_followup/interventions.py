"""Frozen-decoder mean, amplitude, donor and optimized-global-code interventions."""
import numpy as np
import torch

from src.training_methods.bcr.data import BalancedStream, pack, corrupt, balanced_subset
from src.training_methods.bcr.evaluate import matching_bins, derangement, paired_root_gain
from src.training_methods.bcr.objective import per_environment
from src.training_methods.bcr.runtime import save
from .common import fixed_batch, decode, remaining, write_json


@torch.no_grad()
def constant_score(study, decoder, code, indices, device, deadline):
    patches = [study.patches[i] for i in indices]
    values = []
    for k in range(len(study.manifest['noise_levels'])):
        for start in range(0, len(patches), 16):
            remaining(deadline)
            clean, noisy, epsilon, sigma = fixed_batch(patches, start, min(start+16, len(patches)), k, 0,
                study.manifest['noise_levels'], study.manifest['d0'], device, seed=8731)
            prediction = decode(decoder, noisy, code[None].expand(len(sigma), -1), sigma, study.manifest['d0'])
            values.extend(per_environment(prediction, epsilon, clean, study.manifest['radius_A']).cpu().tolist())
    return float(np.mean(values))


def fit_constant(study, model, codes, root, device, deadline):
    cfg = study.config['constant']; d0 = study.manifest['d0']; levels = study.manifest['noise_levels']
    mean = codes[study.split['train']].mean(0)
    code = torch.nn.Parameter(torch.as_tensor(mean, device=device).clone())
    # load_model froze both encoder and decoder: only this one vector is trainable.
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Constant-code diagnostic requires a frozen encoder and decoder')
    optimizer = torch.optim.Adam([code], lr=cfg['learning_rate'])
    fit = study.split['fit']
    tune = balanced_subset(study.records, study.split['tune'], cfg['tuning_anchors'])
    stream = BalancedStream([study.records[i] for i in fit], study.config['seed']+501)
    noise = torch.Generator().manual_seed(study.config['seed']+503)
    root.mkdir(parents=True, exist_ok=True)
    last = root/'constant-last.pt'; start = 0
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state['identity'] != study.identity:
            raise ValueError('Constant-code resume identity changed')
        with torch.no_grad(): code.copy_(state['code'])
        optimizer.load_state_dict(state['optimizer']); stream.load_state_dict(state['stream'])
        noise.set_state(state['noise_rng'].cpu()); start = state['step']
        best, best_code, best_step, trace = state['best'], state['best_code'], state['best_step'], state['trace']
    else:
        best = constant_score(study, model.decoder, code, tune, device, deadline)
        best_code, best_step, trace = code.detach().clone(), 0, [dict(step=0, tuning_nmse=best)]

    def persist(step):
        save(last, dict(identity=study.identity, step=step, code=code.detach(), optimizer=optimizer.state_dict(),
                        stream=stream.state_dict(), noise_rng=noise.get_state(), best=best, best_code=best_code,
                        best_step=best_step, trace=trace))

    # A deadline may interrupt tuning after the optimizer checkpoint was saved.
    # Finish that selection decision before taking another update or exporting.
    if start > trace[-1]['step'] and (start % cfg['evaluate_every'] == 0 or start == cfg['updates']):
        value = constant_score(study, model.decoder, code, tune, device, deadline)
        trace.append(dict(step=start, tuning_nmse=value))
        if value < best: best, best_code, best_step = value, code.detach().clone(), start
        persist(start)

    for step in range(start, cfg['updates']):
        try: remaining(deadline)
        except TimeoutError:
            persist(step); raise
        indices = [fit[i] for i in stream.draw(cfg['batch_size'])]
        clean = pack([study.patches[i] for i in indices], device)
        noisy, epsilon, sigma, _ = corrupt(clean, levels, d0, noise)
        optimizer.zero_grad()
        for first in range(0, len(indices), cfg['microbatch']):
            sl = slice(first, first+cfg['microbatch'])
            a, b = ({k: v[sl] for k, v in x.items()} for x in (clean, noisy))
            prediction = decode(model.decoder, b, code[None].expand(len(a['mask']), -1), sigma[sl], d0)
            loss = per_environment(prediction, epsilon[sl], a, study.manifest['radius_A']).sum()/len(indices)
            if not torch.isfinite(loss): raise FloatingPointError(f'Constant-code loss at {step}')
            loss.backward()
        torch.nn.utils.clip_grad_norm_([code], 1., error_if_nonfinite=True); optimizer.step()
        if (step+1) % cfg['evaluate_every'] == 0 or step+1 == cfg['updates']:
            # Save before evaluation too, so a deadline during tuning loses no updates.
            persist(step+1)
            value = constant_score(study, model.decoder, code, tune, device, deadline)
            trace.append(dict(step=step+1, tuning_nmse=value))
            if value < best: best, best_code, best_step = value, code.detach().clone(), step+1
            persist(step+1)
    write_json(root/'constant-fit.json', dict(identity=study.identity, selected_step=best_step,
        best_tuning_nmse=best, mean_code=mean.tolist(), optimized_code=best_code.cpu().tolist(),
        fit_roots=sorted({study.records[i]['root'] for i in fit}), tuning_roots=study.split['tuning_roots'], trace=trace))
    return best_code.detach(), torch.as_tensor(mean, device=device)


def intervention_codes(z, mean, optimized, alphas, mappings):
    result = {f'alpha_{a:g}': mean + a*(z-mean) for a in alphas}
    result['optimized_constant'] = optimized.expand_as(z)
    for key, ids in mappings.items():
        result[key] = z[torch.as_tensor(ids.clip(0), device=z.device)]
    return result


def run(study, device, deadline=None):
    _, covariates = study.descriptors()
    records = [study.records[i] for i in study.chosen]
    roots = np.array([r['root'] for r in records]); conditions = [r['temperature_K'] for r in records]
    patches = [study.patches[i] for i in study.chosen]
    bins = matching_bins(covariates[study.split['train']])
    liquid = covariates[study.chosen, 2] < .35
    mappings = {f'{name}_{j}': derangement(roots, conditions, covariates[study.chosen], bins, 731+13*j, relax)
                for name, relax in [('strict_donor', 0), ('unrestricted_donor', 2)]
                for j in range(study.pilot['evaluation_shuffles'])}
    levels = study.manifest['noise_levels']; d0 = study.manifest['d0']
    for step in study.config['intervention_steps']:
        root = study.technical/'interventions'/f'{step:06d}'
        if (root/'complete.json').exists(): continue
        remaining(deadline)
        model = study.model(step, device); all_codes = study.features(step)['exported']
        optimized, mean = fit_constant(study, model, all_codes, root, device, deadline)
        z = torch.as_tensor(all_codes[study.chosen], device=device)
        codes = intervention_codes(z, mean, optimized, study.config['alphas'], mappings)
        errors = {name: np.empty((len(levels), study.pilot['evaluation_draws'], len(patches))) for name in codes}
        with torch.no_grad():
            for k, level in enumerate(levels):
                for draw in range(study.pilot['evaluation_draws']):
                    for start in range(0, len(patches), 16):
                        remaining(deadline); stop = min(start+16, len(patches))
                        clean, noisy, epsilon, sigma = fixed_batch(patches, start, stop, k, draw, levels, d0, device)
                        for name, code in codes.items():
                            prediction = decode(model.decoder, noisy, code[start:stop], sigma, d0)
                            value = per_environment(prediction, epsilon, clean, study.manifest['radius_A']).cpu().numpy()
                            if name in mappings: value[mappings[name][start:stop] < 0] = np.nan
                            errors[name][k, draw, start:stop] = value
        # Exact reference-bank replay, including original draw/anchor ordering.
        import json
        original = json.loads((study.original/f'technical/evaluations/{step:06d}/bcr-reconstruction.json').read_text())
        for k, level in enumerate(levels):
            np.testing.assert_allclose(errors['alpha_1'][k].mean(0), original['levels'][str(level)]['per_anchor_nmse'],
                                       atol=3e-5, rtol=2e-4, err_msg='Correct-code corruption replay differs')
        np.savez(root/'errors.npz', **errors, roots=roots, indices=np.array(study.chosen), liquid=liquid,
                 **{f'map_{k}': v for k, v in mappings.items()})
        summary = {}
        for k, level in enumerate(levels):
            means = {name: values[k].mean(0) for name, values in errors.items() if name not in mappings}
            for kind in ('strict_donor', 'unrestricted_donor'):
                values = np.stack([errors[f'{kind}_{j}'][k] for j in range(study.pilot['evaluation_shuffles'])])
                count = np.isfinite(values).sum((0, 1))
                means[kind] = np.divide(np.nansum(values, axis=(0, 1)), count,
                                       out=np.full(len(patches), np.nan), where=count > 0)
            summary[str(level)] = {population: {name: paired_root_gain(means['alpha_1'][mask], other[mask], roots[mask])
                for name, other in means.items()} for population, mask in [('all', np.ones(len(roots), bool)), ('liquid', liquid)]}
        write_json(root/'complete.json', dict(identity=study.identity, encoder_step=step, levels=summary,
                    training_mean_roots=len({study.records[i]['root'] for i in study.split['train']}),
                    constant_selection='Two training tuning roots only; no development adaptation',
                    identical_all_liquid_population=bool(liquid.all()), corruption_reference_seed=731))
        print(f'Finished frozen-decoder interventions at step {step}', flush=True)
