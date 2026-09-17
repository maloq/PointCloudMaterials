"""Matched-budget exploratory fits with exact checkpoint/resume state."""
import argparse
from datetime import datetime
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
from src.models.encoders.mace_backend import mace_backend_metadata
from src.project_runtime.paths import load_json
from src.experiment_runner.metric_docs import write_metric_table
from .objective import PathHeads, fit_scaler, joint_nll, physical_scores
from .runtime import MemoryRuntime, batch_settings, sample_batch


def bootstrap_sources(values, sources, draws, seed):
    unique = np.unique(sources)
    source_means = np.array([np.mean(values[np.asarray(sources) == s]) for s in unique])
    rng = np.random.default_rng(seed)
    estimate = source_means[rng.integers(len(unique), size=(draws, len(unique)))].mean(1)
    return dict(mean=float(source_means.mean()), ci95=np.quantile(estimate, [.025, .975]).tolist(), sources=len(unique))


@torch.no_grad()
def evaluate(model, runtime, indices, batch_size):
    model.eval()
    result, embeddings = [], []
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start+batch_size]
        z = model['encoder'](runtime.observations(batch))
        prediction = model['heads'](z, runtime.condition[batch])
        scores = physical_scores(prediction, runtime.standard_present[batch], runtime.standard_future[batch])
        scores = {key: value.cpu().tolist() for key, value in scores.items()}
        for local, index in enumerate(batch):
            row = runtime.dataset.rows[index]
            result.append(dict(source_id=row['source_id'], anchor=row['anchor'], center_id=row['center_id'],
                               **{key: values[local] for key, values in scores.items()}))
        embeddings.append(z.cpu())
    return result, torch.cat(embeddings)


def training_update(model, runtime, indices, optimizer, training, micro_batch_size):
    """Average over windows; clip and update once, including an uneven last microbatch."""
    model.train(); optimizer.zero_grad(set_to_none=True)
    future_total = torch.zeros((), device=runtime.device)
    present_total = torch.zeros_like(future_total)
    for start in range(0, len(indices), micro_batch_size):
        batch = indices[start:start+micro_batch_size]
        state = model['encoder'](runtime.observations(batch))
        prediction = model['heads'](state, runtime.condition[batch])
        future_loss = joint_nll(prediction, runtime.standard_future[batch]).mean()
        present_loss = (prediction['present']-runtime.standard_present[batch]).square().mean()
        weight = len(batch)/len(indices)
        loss = (future_loss+training['present_weight']*present_loss)*weight
        if not torch.isfinite(loss):
            keys = [(runtime.dataset.rows[i]['source_id'], runtime.dataset.rows[i]['anchor']) for i in batch]
            raise FloatingPointError(f'Nonfinite predictive-memory loss on source/anchor rows {keys}')
        loss.backward()
        future_total += future_loss.detach()*weight
        present_total += present_loss.detach()*weight
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=future_total+training['present_weight']*present_total,
                future_nll=future_total, present_mse=present_total, gradient_norm=grad)


def save_checkpoint(path, model, optimizer, step, best, config, variant, normalizer, dataset, sampler):
    payload = dict(format_version=2, model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, best=best,
        config=config, variant=variant, normalizer=[v.cpu() for v in normalizer], release_sha256=dataset.release_sha256,
        sampler_state=sampler.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    temporary = path.with_suffix('.building')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, config, variant, dataset, sampler, device):
    checkpoint = torch.load(path, weights_only=False, map_location='cpu')
    if checkpoint.get('format_version') != 2:
        raise ValueError('This trainer requires checkpoint format 2; use the original commit for earlier runs')
    if checkpoint['config'] != config or checkpoint['variant'] != variant or checkpoint['release_sha256'] != dataset.release_sha256:
        raise ValueError('Resume requires identical scientific configuration, variant and immutable release')
    model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
    sampler.bit_generator.state = checkpoint['sampler_state']
    torch.set_rng_state(checkpoint['torch_rng']); torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return checkpoint['step'], checkpoint['best'], [v.to(device) for v in checkpoint['normalizer']]


def fit(config, *, history_ps, velocity, repeat_anchor=False, resume=False, deadline_utc=None):
    batch_size, micro_batch_size, evaluation_batch_size = batch_settings(config['training'])
    torch.set_num_threads(config.get('runtime', {}).get('torch_threads', 2))
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
    runtime = MemoryRuntime(dataset, device, config.get('runtime'))
    print(f'Loaded {len(dataset.rows)} matched windows; variant={variant}', flush=True)
    model = nn.ModuleDict(dict(encoder=PredictiveMemoryEncoder(**config['encoder'], radius_A=config['radius_A'],
        cutoff_A=config['cutoff_A'], use_velocity=velocity, use_history=history_ps > 0, repeat_anchor=repeat_anchor),
        heads=PathHeads(config['encoder']['output_dim'], len(config['future_lags_ps']), **config['mixture']))).to(device)
    training = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], weight_decay=training['weight_decay'])
    train_indices = dataset.indices['train']
    normalizer = fit_scaler(runtime.present[train_indices], runtime.future[train_indices])
    sampler = np.random.default_rng(seed)
    step, best = 0, float('inf')
    if resume:
        step, best, normalizer = restore_checkpoint(latest, model, optimizer, config, variant, dataset, sampler, device)
    runtime.normalize(normalizer)
    write_json(technical/'runtime.json', dict(batch_size=batch_size, micro_batch_size=micro_batch_size,
        evaluation_batch_size=evaluation_batch_size, **runtime.statistics(),
        mace_backend=mace_backend_metadata(model['encoder'].mace_backend),
        cuda_device=torch.cuda.get_device_name(device), torch_version=torch.__version__,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
    # Equal numbers of anchors per source make uniform row sampling source-balanced.
    validation = [i for i in dataset.indices['val'] if dataset.rows[i]['anchor'] == config['anchor_frames'][1]]
    deadline = datetime.fromisoformat(deadline_utc).timestamp() if deadline_utc else float('inf')
    start = time.monotonic()
    log = (technical/'training.jsonl').open('a' if resume else 'x')
    validation_log = (technical/'validation.jsonl').open('a' if resume else 'x')
    for step in range(step+1, training['steps']+1):
        if time.time() > deadline-120:
            save_checkpoint(latest, model, optimizer, step-1, best, config, variant, normalizer, dataset, sampler)
            write_json(technical/'status.json', dict(state='paused_deadline', step=step-1, target_steps=training['steps']))
            raise TimeoutError('Allocation deadline reserve reached; exact resume retained')
        indices = sample_batch(sampler, train_indices, batch_size)
        values = training_update(model, runtime, indices, optimizer, training, micro_batch_size)
        if step % training['log_every'] == 0 or step == 1:
            record = dict(step=step, **{key: float(value) for key, value in values.items()},
                windows_seen=step*batch_size, batch_size=batch_size, micro_batch_size=micro_batch_size,
                elapsed_seconds=time.monotonic()-start, max_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                **runtime.statistics())
            print(json.dumps(record), file=log, flush=True)
            print(f'{name} '+json.dumps(record), flush=True)
            write_json(technical/'status.json', dict(state='training', target_steps=training['steps'], **record))
        if step % training['validation_every'] == 0 or step == training['steps']:
            scores, _ = evaluate(model, runtime, validation, evaluation_batch_size)
            selection = float(np.mean([r['joint_nll'] for r in scores]))
            improved = selection < best
            if improved:
                best = selection
                save_checkpoint(technical/'best.pt', model, optimizer, step, best, config, variant, normalizer, dataset, sampler)
            save_checkpoint(latest, model, optimizer, step, best, config, variant, normalizer, dataset, sampler)
            print(json.dumps(dict(step=step, joint_nll=selection, best=best,
                present_mse=float(np.mean([r['present_mse'] for r in scores])),
                future_mse=float(np.mean([r['future_mse'] for r in scores])),
                elapsed_seconds=time.monotonic()-start)), file=validation_log, flush=True)
            print(f'{name} validation step={step} joint_nll={selection:.6g} best={best:.6g}', flush=True)
    log.close()
    validation_log.close()
    checkpoint = torch.load(technical/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    metrics = dict(selected_step=checkpoint['step'], trained_steps=training['steps'], history_ps=history_ps,
        velocity=int(velocity), repeat_anchor=int(repeat_anchor), parameters=sum(p.numel() for p in model.parameters()),
        max_allocated_GiB=torch.cuda.max_memory_allocated()/2**30, batch_size=batch_size,
        micro_batch_size=micro_batch_size, evaluation_batch_size=evaluation_batch_size,
        trained_windows=training['steps']*batch_size, selected_windows=checkpoint['step']*batch_size)
    exports = {}
    for split in ('train', 'val', 'test'):
        indices = dataset.indices[split]
        rows, embeddings = evaluate(model, runtime, indices, evaluation_batch_size)
        exports[split] = dict(rows=rows, embeddings=embeddings, indices=indices)
        metrics[split] = {key: bootstrap_sources(np.array([r[key] for r in rows]),
            [r['source_id'] for r in rows], config['bootstrap_draws'], seed)
            for key in rows[0] if key not in ('source_id', 'anchor', 'center_id')}
    torch.save(exports, technical/'evaluation.pt')
    metrics['wall_seconds'] = time.monotonic()-start
    metrics['runtime'] = runtime.statistics()
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
