"""Matched-budget exploratory fits with exact checkpoint/resume state."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from src.data.predictive_memory.prepare import write_json
from src.data.predictive_memory.windows import MemoryDataset
from src.models.encoders.predictive_memory import PredictiveMemoryEncoder
from src.project_runtime.paths import load_json
from src.experiment_runner.metric_docs import write_metric_table
from .objective import PathHeads, fit_scaler, joint_nll, physical_scores


def bootstrap_sources(values, sources, draws, seed):
    unique = np.unique(sources)
    source_means = np.array([np.mean(values[np.asarray(sources) == s]) for s in unique])
    rng = np.random.default_rng(seed)
    estimate = source_means[rng.integers(len(unique), size=(draws, len(unique)))].mean(1)
    return dict(mean=float(source_means.mean()), ci95=np.quantile(estimate, [.025, .975]).tolist(), sources=len(unique))


@torch.no_grad()
def evaluate(model, dataset, indices, normalizer, device):
    model.eval()
    center, scale = normalizer
    result, embeddings = [], []
    for i in indices:
        present, future, condition = dataset.targets([i], device)
        z = model['encoder'](dataset.observation(i).to(device))
        prediction = model['heads'](z, condition)
        scores = physical_scores(prediction, (present-center)/scale, (future-center)/scale)
        row = dataset.rows[i]
        result.append(dict(source_id=row['source_id'], anchor=row['anchor'], center_id=row['center_id'],
                           **{k: float(v.item()) for k, v in scores.items()}))
        embeddings.append(z.cpu())
    return result, torch.cat(embeddings)


def save_checkpoint(path, model, optimizer, step, best, config, variant, normalizer, dataset, sampler):
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, best=best,
        config=config, variant=variant, normalizer=[v.cpu() for v in normalizer], release_sha256=dataset.release_sha256,
        sampler_state=sampler.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    temporary = path.with_suffix('.building')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, config, variant, dataset, sampler, device):
    checkpoint = torch.load(path, weights_only=False, map_location='cpu')
    if checkpoint['config'] != config or checkpoint['variant'] != variant or checkpoint['release_sha256'] != dataset.release_sha256:
        raise ValueError('Resume requires identical scientific configuration, variant and immutable release')
    model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
    sampler.bit_generator.state = checkpoint['sampler_state']
    torch.set_rng_state(checkpoint['torch_rng']); torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return checkpoint['step'], checkpoint['best'], [v.to(device) for v in checkpoint['normalizer']]


def fit(config, *, history_ps, velocity, repeat_anchor=False, resume=False, deadline_utc=None):
    if config['training']['batch_size'] != 1:
        raise ValueError('Initial memory pilot uses source-balanced batch size 1; no hidden batch approximation')
    torch.set_num_threads(2)
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda')
    variant = dict(history_ps=history_ps, velocity=velocity, repeat_anchor=repeat_anchor)
    name = f"{'xv' if velocity else 'x'}-H{history_ps:g}"+('-repeat' if repeat_anchor else '')
    root = Path(config['output'])/name
    technical = root/'technical'
    technical.mkdir(parents=True, exist_ok=True)
    (root/'tables').mkdir(exist_ok=True)
    latest = technical/'latest.pt'
    if latest.exists() and not resume:
        raise FileExistsError(f'Existing fit requires explicit --resume: {latest}')
    dataset = MemoryDataset(config, history_ps)
    print(f'Loaded {len(dataset.rows)} matched windows; variant={variant}', flush=True)
    model = nn.ModuleDict(dict(encoder=PredictiveMemoryEncoder(**config['encoder'], radius_A=config['radius_A'],
        cutoff_A=config['cutoff_A'], use_velocity=velocity, use_history=history_ps > 0, repeat_anchor=repeat_anchor),
        heads=PathHeads(config['encoder']['output_dim'], len(config['future_lags_ps']), **config['mixture']))).to(device)
    training = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], weight_decay=training['weight_decay'])
    present, future, _ = dataset.targets(dataset.indices['train'], device)
    normalizer = fit_scaler(present, future)
    sampler = np.random.default_rng(seed)
    step, best = 0, float('inf')
    if resume:
        step, best, normalizer = restore_checkpoint(latest, model, optimizer, config, variant, dataset, sampler, device)
    normal_center, normal_scale = normalizer
    # Equal numbers of anchors per source make uniform row sampling source-balanced.
    validation = [i for i in dataset.indices['val'] if dataset.rows[i]['anchor'] == config['anchor_frames'][1]]
    deadline = datetime.fromisoformat(deadline_utc).timestamp() if deadline_utc else float('inf')
    start = time.monotonic()
    log = (technical/'training.jsonl').open('a' if resume else 'x')
    for step in range(step+1, training['steps']+1):
        if time.time() > deadline-120:
            save_checkpoint(latest, model, optimizer, step-1, best, config, variant, normalizer, dataset, sampler)
            write_json(technical/'status.json', dict(state='paused_deadline', step=step-1, target_steps=training['steps']))
            raise TimeoutError('Allocation deadline reserve reached; exact resume retained')
        model.train(); optimizer.zero_grad(set_to_none=True)
        index = int(sampler.choice(dataset.indices['train']))
        present, future, condition = dataset.targets([index], device)
        state = model['encoder'](dataset.observation(index).to(device))
        prediction = model['heads'](state, condition)
        future_loss = joint_nll(prediction, (future-normal_center)/normal_scale).mean()
        present_loss = (prediction['present']-(present-normal_center)/normal_scale).square().mean()
        loss = future_loss+training['present_weight']*present_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss variant={variant} step={step} source={dataset.rows[index]["source_id"]}')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        if step % training['log_every'] == 0 or step == 1:
            record = dict(step=step, loss=float(loss.detach()), future_nll=float(future_loss.detach()),
                present_mse=float(present_loss.detach()), gradient_norm=float(grad), elapsed_seconds=time.monotonic()-start,
                max_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
            print(json.dumps(record), file=log, flush=True)
            print(f'{name} '+json.dumps(record), flush=True)
            write_json(technical/'status.json', dict(state='training', target_steps=training['steps'], **record))
        if step % training['validation_every'] == 0 or step == training['steps']:
            scores, _ = evaluate(model, dataset, validation, normalizer, device)
            selection = float(np.mean([r['joint_nll'] for r in scores]))
            improved = selection < best
            if improved:
                best = selection
                save_checkpoint(technical/'best.pt', model, optimizer, step, best, config, variant, normalizer, dataset, sampler)
            save_checkpoint(latest, model, optimizer, step, best, config, variant, normalizer, dataset, sampler)
            print(f'{name} validation step={step} joint_nll={selection:.6g} best={best:.6g}', flush=True)
    log.close()
    checkpoint = torch.load(technical/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    metrics = dict(selected_step=checkpoint['step'], trained_steps=training['steps'], history_ps=history_ps,
        velocity=int(velocity), repeat_anchor=int(repeat_anchor), parameters=sum(p.numel() for p in model.parameters()),
        max_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    exports = {}
    for split in ('train', 'val', 'test'):
        indices = dataset.indices[split]
        rows, embeddings = evaluate(model, dataset, indices, normalizer, device)
        exports[split] = dict(rows=rows, embeddings=embeddings, indices=indices)
        metrics[split] = {key: bootstrap_sources(np.array([r[key] for r in rows]),
            [r['source_id'] for r in rows], config['bootstrap_draws'], seed)
            for key in rows[0] if key not in ('source_id', 'anchor', 'center_id')}
    torch.save(exports, technical/'evaluation.pt')
    metrics['wall_seconds'] = time.monotonic()-start
    write_json(technical/'metrics.json', metrics)
    write_metric_table(metrics, root, family='predictive_memory')
    write_json(technical/'status.json', dict(state='complete', step=training['steps'], selected_step=checkpoint['step'],
                                          wall_seconds=metrics['wall_seconds']))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--history-ps', type=float, choices=(0., 12., 48.), required=True)
    parser.add_argument('--velocity', action='store_true')
    parser.add_argument('--repeat-anchor', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--deadline-utc')
    args = parser.parse_args()
    config = load_json(args.config)
    fit(config, history_ps=args.history_ps, velocity=args.velocity, repeat_anchor=args.repeat_anchor,
        resume=args.resume, deadline_utc=args.deadline_utc)


if __name__ == '__main__':
    main()
