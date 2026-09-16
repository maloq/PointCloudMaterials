"""Matched-budget native MACE ablations with validation-only checkpoint selection."""
from pathlib import Path
import time

import numpy as np
import torch

from src.experiment_runner.metric_docs import check_metric_docs
from src.experiment_runner.registry import sha256, write_json
from src.models.encoders.mace_causal import CausalMACEEncoder
from .data import PROTOCOL, atomic_save, load
from .evaluate import encode, export, extract, jump_metrics, physical_metrics
from .objective import PhysicalHeads, admissible, target_normalization, targets, task_objective
from .runtime import CausalRuntime


VARIANTS = ('A', 'B', 'C', 'D', 'repeated_anchor', 'E')


def variant_root(config, variant):
    """Each fit is output/<question>/<run-variant>/{tables,technical}."""
    base = Path(config['output'])
    return base.with_name(f'{base.name}-{variant}')


def initialize(config, variant, device):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown controlled ablation: {variant}')
    torch.manual_seed(config['seed'])
    kwargs = dict(config['encoder'], use_velocity=variant not in ('A', 'B'),
                  use_history=variant in ('D', 'E', 'repeated_anchor'), repeat_anchor=variant == 'repeated_anchor')
    model = CausalMACEEncoder(**kwargs).to(device)
    # Reset so task heads start identically across architecture switches.
    torch.manual_seed(config['seed']+1)
    heads = PhysicalHeads(model.invariant_dim, config['future_lags_ps'], **config['heads'],
        event_bins_ps=config['events']['bin_edges_ps'] if config['events']['enabled'] else ()).to(device)
    return model, heads


def load_encoder(checkpoint, device='cpu'):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['protocol'] != PROTOCOL:
        raise ValueError(f'Expected causal native MACE checkpoint: {checkpoint}')
    model, _ = initialize(saved['config'], saved['variant'], device)
    model.load_state_dict(saved['encoder_state'], strict=True)
    return model.eval(), saved['config']


def paired_indices(samples, lag):
    lookup = {(s['source_id'], s['center_atom_id'], round(s['anchor_ps'], 8)): i for i, s in enumerate(samples)}
    return {i: lookup[key] for i, s in enumerate(samples)
            if (key := (s['source_id'], s['center_atom_id'], round(s['anchor_ps']+lag, 8))) in lookup}


def validation_score(constraint, config, variant):
    def average(prefix):
        values = [v for k, v in constraint.items() if k.startswith(prefix)]
        if not values:
            raise ValueError(f'No validation tasks for {prefix}')
        return float(np.mean(values))
    score = average('present/')
    if variant != 'A':
        weights = config['loss']
        score += weights['future_weight']*(average('future/')+weights['delta_weight']*average('delta/')
                                           +weights['path_weight']*average('path/'))
        if 'hazard/nll' in constraint:
            score += weights['hazard_weight']*constraint['hazard/nll']
    return score


def run(config, args):
    torch.set_num_threads(config['cpu_threads'])
    samples, identity = load(config)
    contracts = check_metric_docs()['mace_causal']['files']
    identity['implementation'] = contracts
    variant, device = args.variant, args.device
    root = variant_root(config, variant)
    technical = root/'technical'
    if not args.resume and technical.exists():
        raise FileExistsError(f'Preserve completed/partial training: {technical}')
    technical.mkdir(parents=True, exist_ok=True)
    model, heads = initialize(config, variant, device)
    norm = target_normalization(samples)
    runtime = CausalRuntime(samples, norm, model, device, config.get('runtime'))
    write_json(technical/'runtime.json', runtime.summary)
    train = [s for s in samples if s['split'] == 'train']
    validation = [s for s in samples if s['split'] == 'val']
    by_source = {sid: [i for i, s in enumerate(train) if s['source_id'] == sid]
                 for sid in sorted({s['source_id'] for s in train})}
    rng = np.random.default_rng(config['seed'])
    optimizer = torch.optim.AdamW(list(model.parameters())+list(heads.parameters()),
                                 lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    smooth = config['smoothness']
    reference = None
    reference_trace = None
    if variant == 'E':
        baseline = torch.load(smooth['reference_checkpoint'], map_location='cpu', weights_only=False)
        if (baseline['protocol'] != PROTOCOL or baseline['variant'] != 'D'
                or baseline['identity']['cache_sha256'] != identity['cache_sha256']
                or baseline['identity']['implementation'] != identity['implementation']
                or baseline['config']['encoder'] != config['encoder']
                or baseline['config']['heads'] != config['heads'] or smooth['weight'] <= 0):
            raise ValueError('E requires a matched informative D checkpoint and positive slowness coefficient')
        model.load_state_dict(baseline['encoder_state'], strict=True)
        heads.load_state_dict(baseline['head_state'], strict=True)
        reference = baseline['validation_constraints']
        base_train = runtime.extract(model, heads, train)
        reference_trace = float(np.mean([np.var(base_train['z'][indices], axis=0).sum()
                                        for indices in by_source.values()]))
        if reference_trace <= 0:
            raise ValueError('Collapsed informative reference has no usable slowness scale')
        identity['reference_checkpoint_sha256'] = sha256(Path(smooth['reference_checkpoint']))
    pairs = paired_indices(train, smooth['lag_ps'])
    if variant == 'E' and not pairs:
        raise ValueError('No adjacent training windows at the declared smoothness lag')
    best, start, history, selected = float('inf'), 0, [], False
    if args.resume:
        saved = torch.load(technical/'last.pt', map_location='cpu', weights_only=False)
        if saved['config'] != config or saved['identity'] != identity or saved['variant'] != variant:
            raise ValueError('Exact resume requires unchanged config, data, variant and implementation')
        model.load_state_dict(saved['encoder_state'], strict=True)
        heads.load_state_dict(saved['head_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state'])
        rng.bit_generator.state = saved['numpy_rng_state']
        torch.set_rng_state(saved['torch_rng_state'])
        if saved['cuda_rng_state'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
        best, start, history, selected = saved['best_score'], saved['step']+1, saved['history'], saved['selected']
    else:
        for relative, digest in contracts.items():
            path = Path(relative)
            if sha256(path) != digest:
                raise ValueError(f'Implementation changed during snapshot: {relative}')
            destination = technical/'source-snapshot'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        write_json(technical/'config.json', config)
    started = time.monotonic()
    try:
        for step in range(start, config['training']['steps']+1):
            if step:
                chosen_sources = rng.choice(list(by_source), config['training']['batch_sources'], replace=True)
                chosen = [int(rng.choice(by_source[sid])) for sid in chosen_sources]
                batch = [train[i] for i in chosen]
                model.train(); heads.train(); optimizer.zero_grad(set_to_none=True)
                z = runtime.encode(model, batch)
                target = runtime.target(batch)
                pred = heads(z, target['temperature'])
                weights = dict(config['loss'])
                if variant == 'A':
                    weights.update(future_weight=0., hazard_weight=0.)
                value, metrics = task_objective(pred, target, **weights)
                if variant == 'E':
                    eligible = [j for j, i in enumerate(chosen) if i in pairs]
                    if eligible:
                        next_z = runtime.encode(model, [train[pairs[chosen[j]]] for j in eligible])
                        slowness = (next_z-z[eligible]).square().sum(-1).mean()/(2*reference_trace)
                        value = value+smooth['weight']*slowness
                        metrics['slowness'] = float(slowness.detach())
                value.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(heads.parameters()),
                                               config['training']['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                if step % config['training']['log_every'] == 0:
                    print('CAUSAL TRAIN', variant, step, metrics, flush=True)
            if step % config['training']['validation_every'] == 0 or step == config['training']['steps']:
                result = runtime.extract(model, heads, validation)
                _, constraint = physical_metrics(result, validation, norm, config['future_lags_ps'])
                informative = reference is None or admissible(constraint, reference,
                    smooth['relative_tolerance'], smooth['absolute_tolerance'])
                if variant == 'E':
                    jumps = [r['value'] for r in jump_metrics(result['z'], validation, smooth['lag_ps'])
                             if r['population'] == 'all' and r['metric'].endswith('/normalized_rms_jump')]
                    if not jumps or any(v is None for v in jumps):
                        raise ValueError('Validation has no defined smoothness pairs/variance')
                    score = float(np.mean(jumps))
                else:
                    score = validation_score(constraint, config, variant)
                improved = informative and score < best
                if improved:
                    best, selected = score, True
                history.append(dict(step=step, score=score, admissible=informative, constraints=constraint))
                payload = dict(protocol=PROTOCOL, config=config, variant=variant, step=step, identity=identity,
                    encoder_state=model.state_dict(), head_state=heads.state_dict(), optimizer_state=optimizer.state_dict(),
                    normalization=norm, validation_constraints=constraint, best_score=best, selected=selected, history=history,
                    numpy_rng_state=rng.bit_generator.state, torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
                if improved:
                    atomic_save(technical/'best.pt', payload)
                atomic_save(technical/'last.pt', payload)
                write_json(technical/'history.json', history)
                write_json(technical/'status.json', dict(state='training', step=step, admissible=informative, score=score))
                print('CAUSAL VALIDATION', variant, step, score, 'admissible=', informative, flush=True)
        if not selected:
            raise RuntimeError('No checkpoint met all validation information constraints; no best model selected')
        saved = torch.load(technical/'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['encoder_state'], strict=True)
        heads.load_state_dict(saved['head_state'], strict=True)
        thresholds = None
        if config['events']['enabled']:
            from .events import thresholds_from_validation
            val_result = runtime.extract(model, heads, validation)
            thresholds = thresholds_from_validation(val_result['hazard'], validation,
                                                     config['events']['maximum_false_alarm_rate'])
            write_json(technical/'event-thresholds.json', dict(thresholds=thresholds, fitted_split='val'))
        for split in ('train', 'val', 'test'):
            subset = [s for s in samples if s['split'] == split]
            result = runtime.extract(model, heads, subset)
            export(root, result, subset, norm, config, split, thresholds)
        write_json(technical/'status.json', dict(state='complete', steps=config['training']['steps'],
                   selected_step=saved['step'], elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        write_json(technical/'status.json', dict(state='failed', error=repr(error)))
        raise
